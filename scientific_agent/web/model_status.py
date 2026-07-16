"""Bounded, read-only load probes for operator-configured model backends."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_STATUS_RESPONSE_BYTES = 512 * 1024
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^\r\n]*\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def validate_status_base_url(value: str, setting_name: str) -> str:
    """Validate a startup-only probe origin; request paths are never user supplied."""

    value = value.strip().rstrip("/")
    if not value:
        return ""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{setting_name} must be an HTTP(S) origin without credentials, path, "
            "query, or fragment"
        )
    return value


def _bounded_fetch(url: str, timeout: float) -> bytes:
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    request = Request(url, headers={"Accept": "application/json,text/plain"})
    try:
        with opener.open(request, timeout=timeout) as response:
            data = response.read(MAX_STATUS_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        raise OSError(f"status endpoint returned HTTP {exc.code}") from exc
    if len(data) > MAX_STATUS_RESPONSE_BYTES:
        raise OSError("status endpoint response exceeded the size limit")
    return data


def _metric_values(payload: str, name: str) -> list[float]:
    values: list[float] = []
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.fullmatch(line.strip())
        if match and match.group("name") == name:
            value = float(match.group("value"))
            if math.isfinite(value):
                values.append(value)
    return values


def _metric_total(payload: str, name: str) -> float | None:
    values = _metric_values(payload, name)
    return sum(values) if values else None


@dataclass(frozen=True)
class ModelStatusTarget:
    role: str
    model: str
    provider: str
    base_url: str


class ModelStatusMonitor:
    """Cache small snapshots from fixed vLLM/llama.cpp observability endpoints."""

    def __init__(
        self,
        targets: tuple[ModelStatusTarget, ...],
        *,
        timeout_seconds: float = 2.0,
        cache_seconds: float = 3.0,
        fetch: Callable[[str, float], bytes] = _bounded_fetch,
    ) -> None:
        self.targets = targets
        self.timeout_seconds = timeout_seconds
        self.cache_seconds = cache_seconds
        self.fetch = fetch
        self._lock = threading.Lock()
        self._cached_at = 0.0
        self._cached: dict | None = None

    def snapshot(self) -> dict:
        now = time.monotonic()
        with self._lock:
            if self._cached is not None and now - self._cached_at < self.cache_seconds:
                return self._cached
            models = [self._probe(target) for target in self.targets]
            payload = {
                "updated_at": datetime.now(UTC).isoformat(),
                "models": models,
                "summary": self._summary(models),
            }
            self._cached = payload
            self._cached_at = time.monotonic()
            return payload

    def _probe(self, target: ModelStatusTarget) -> dict:
        base = target.base_url
        if not base:
            return self._result(
                target,
                state="unconfigured",
                reachable=False,
                message="Load monitoring is not configured",
            )
        try:
            metrics = self.fetch(f"{base}/metrics", self.timeout_seconds).decode(
                "utf-8", errors="strict"
            )
            if target.provider == "vllm":
                return self._vllm(target, metrics)
            if target.provider == "llama_cpp":
                try:
                    slots = self._llama_slots(base)
                except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
                    slots = None
                return self._llama_cpp(target, metrics, slots)
            raise ValueError("unsupported model status provider")
        except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
            return self._result(
                target,
                state="offline",
                reachable=False,
                message="Backend status is unavailable",
            )

    def _llama_slots(self, base: str) -> tuple[int, int] | None:
        raw = self.fetch(f"{base}/slots", self.timeout_seconds)
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("llama.cpp slot status must be a list")
        total = len(payload)
        busy = sum(
            1
            for item in payload
            if isinstance(item, dict) and item.get("is_processing") is True
        )
        return total, busy

    def _vllm(self, target: ModelStatusTarget, metrics: str) -> dict:
        active = self._required_count(metrics, "vllm:num_requests_running")
        queued = self._required_count(metrics, "vllm:num_requests_waiting")
        cache_values = _metric_values(metrics, "vllm:kv_cache_usage_perc")
        cache = sum(cache_values) / len(cache_values) if cache_values else None
        state, message = self._state_message(active, queued)
        return self._result(
            target,
            state=state,
            reachable=True,
            active_requests=active,
            queued_requests=queued,
            cache_usage_percent=(round(cache * 100, 1) if cache is not None else None),
            message=message,
        )

    def _llama_cpp(
        self, target: ModelStatusTarget, metrics: str, slots: tuple[int, int] | None
    ) -> dict:
        active = self._required_count(metrics, "llamacpp:requests_processing")
        queued = self._required_count(metrics, "llamacpp:requests_deferred")
        total_slots = slots[0] if slots else None
        busy_slots = min(total_slots, active) if total_slots is not None else None
        if queued:
            state = "queued"
            message = f"{queued} request{'s' if queued != 1 else ''} waiting"
        elif total_slots and active >= total_slots:
            state = "saturated"
            message = "All inference slots are busy"
        else:
            state, message = self._state_message(active, queued)
        return self._result(
            target,
            state=state,
            reachable=True,
            active_requests=active,
            queued_requests=queued,
            slots_total=total_slots,
            slots_busy=busy_slots,
            message=message,
        )

    @staticmethod
    def _required_count(metrics: str, name: str) -> int:
        value = _metric_total(metrics, name)
        if value is None or value < 0:
            raise ValueError(f"missing or invalid metric: {name}")
        return int(value)

    @staticmethod
    def _state_message(active: int, queued: int) -> tuple[str, str]:
        if queued:
            return "queued", f"{queued} request{'s' if queued != 1 else ''} waiting"
        if active:
            return "busy", "Processing now; queue is clear"
        return "ready", "Available now"

    @staticmethod
    def _result(
        target: ModelStatusTarget,
        *,
        state: str,
        reachable: bool,
        message: str,
        active_requests: int = 0,
        queued_requests: int = 0,
        slots_total: int | None = None,
        slots_busy: int | None = None,
        cache_usage_percent: float | None = None,
    ) -> dict:
        return {
            "role": target.role,
            "model": target.model,
            "provider": target.provider,
            "reachable": reachable,
            "state": state,
            "active_requests": active_requests,
            "queued_requests": queued_requests,
            "slots_total": slots_total,
            "slots_busy": slots_busy,
            "cache_usage_percent": cache_usage_percent,
            "message": message,
        }

    @staticmethod
    def _summary(models: list[dict]) -> dict:
        queued = sum(model["queued_requests"] for model in models)
        saturated = sum(model["state"] == "saturated" for model in models)
        offline = sum(model["state"] == "offline" for model in models)
        unconfigured = sum(model["state"] == "unconfigured" for model in models)
        if queued:
            message = "Shared model queue is active; new work may wait"
        elif saturated:
            message = "A model is at capacity; its next step may wait"
        elif offline:
            message = "One or more model backends cannot be reached"
        elif unconfigured:
            message = "Backend load monitoring is not fully configured"
        elif any(model["active_requests"] for model in models):
            message = "Models are processing work; current queues are clear"
        else:
            message = "Both scientific models are available now"
        return {
            "queued_requests": queued,
            "saturated_models": saturated,
            "offline_models": offline,
            "unconfigured_models": unconfigured,
            "message": message,
        }
