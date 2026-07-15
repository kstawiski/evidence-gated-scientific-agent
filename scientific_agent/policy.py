"""Deterministic gate for every model-requested tool call."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import stat
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .provenance import EventLedger, utc_now
from .schemas import RetrievalEvidence


READ_ONLY_TOOLS = {
    "list_workspace",
    "read_text_file",
    "search_workspace",
    "resolve-library-id",
    "query-docs",
    "brave_web_search",
    "brave_llm_context",
    "new_page",
    "navigate_page",
    "take_snapshot",
    "take_screenshot",
    "wait_for",
}

URL_ARGUMENT_NAMES = {"url", "uri", "target_url", "browser_url"}
MAX_ARGUMENT_BYTES = 64 * 1024
MAX_RESULT_ARTIFACT_BYTES = 2 * 1024 * 1024
MAX_MODEL_TOOL_RESULT_BYTES = 64 * 1024
MAX_MODEL_TOOL_RESULTS_TOTAL_BYTES = 256 * 1024
RETRIEVAL_TOOLS = {
    "query-docs",
    "brave_web_search",
    "brave_llm_context",
    "search_pubmed",
    "acquire_pubmed_article",
    "import_browser_downloaded_pdf",
}
LITERATURE_TOOLS = {
    "search_pubmed",
    "acquire_pubmed_article",
    "search_acquired_article",
    "list_browser_downloads",
    "import_browser_downloaded_pdf",
}
EXECUTION_TOOLS = {"run_python_analysis", "run_r_analysis"}
INSTALLATION_TOOLS = {"install_python_packages", "install_r_packages"}
_URL = re.compile(r"https?://[^\s<>\"']+")


def _extract_urls(value: Any) -> set[str]:
    if isinstance(value, dict):
        return (
            set().union(*(_extract_urls(item) for item in value.values()))
            if value
            else set()
        )
    if isinstance(value, list | tuple):
        return set().union(*(_extract_urls(item) for item in value)) if value else set()
    if not isinstance(value, str):
        return set()
    return {item.rstrip(".,;:)]}") for item in _URL.findall(value)}


def _is_public_http_url(value: str) -> bool:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return False
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            }
        except OSError:
            return False
    return bool(addresses) and all(
        address.is_global and not address.is_multicast for address in addresses
    )


def _find_url_arguments(value: Any, key: str = "") -> list[str]:
    urls: list[str] = []
    if isinstance(value, dict):
        for nested_key, nested_value in value.items():
            urls.extend(_find_url_arguments(nested_value, str(nested_key).lower()))
    elif isinstance(value, list):
        for nested_value in value:
            urls.extend(_find_url_arguments(nested_value, key))
    elif isinstance(value, str) and (key in URL_ARGUMENT_NAMES or key.endswith("url")):
        urls.append(value)
    return urls


@dataclass
class ToolPolicy:
    ledger: EventLedger
    allowed_tools: set[str]
    chrome_context: str = ""
    chrome_ready: bool = False
    successful_retrievals: int = 0
    retrieval_tools: set[str] | None = None
    retrieved_urls: set[str] | None = None
    retrieval_dates: set[str] | None = None
    retrieval_artifacts: list[str] | None = None
    retrieval_artifact_roots: tuple[Path, ...] = ()
    evidence_dir: Path | None = None
    observer: Callable[[str, str, str], None] | None = None
    _evidence_artifact_count: int = 0
    _model_observation_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.chrome_context:
            self.chrome_context = f"scientific-agent-{uuid.uuid4().hex[:12]}"
        if self.retrieval_tools is None:
            self.retrieval_tools = set()
        if self.retrieved_urls is None:
            self.retrieved_urls = set()
        if self.retrieval_dates is None:
            self.retrieval_dates = set()
        if self.retrieval_artifacts is None:
            self.retrieval_artifacts = []

    def retrieval_evidence(self) -> RetrievalEvidence:
        return RetrievalEvidence(
            successful_calls=self.successful_retrievals,
            tools=sorted(self.retrieval_tools or set()),
            urls=sorted(self.retrieved_urls or set()),
            retrieval_dates=sorted(self.retrieval_dates or set()),
            artifacts=list(self.retrieval_artifacts or []),
        )

    def _observe(self, event_type: str, tool_name: str, status: str) -> None:
        if self.observer is None:
            return
        try:
            self.observer(event_type, tool_name, status)
        except Exception:
            # Monitoring is observational and may not alter a scientific result.
            pass

    def evaluate(self, tool_name: str, arguments: dict[str, Any]) -> tuple[bool, str]:
        if tool_name not in self.allowed_tools:
            return False, f"tool is not allow-listed: {tool_name}"
        encoded = json.dumps(arguments, default=str).encode("utf-8")
        if len(encoded) > MAX_ARGUMENT_BYTES:
            return False, f"tool arguments exceed {MAX_ARGUMENT_BYTES} bytes"
        for url in _find_url_arguments(arguments):
            if not _is_public_http_url(url):
                return False, f"URL is not an allowed public HTTP(S) target: {url}"
        return True, (
            "sandboxed execution allow-list"
            if tool_name in EXECUTION_TOOLS
            else "isolated canonical package registry allow-list"
            if tool_name in INSTALLATION_TOOLS
            else "fixed-host path-confined literature acquisition allow-list"
            if tool_name in LITERATURE_TOOLS
            else "read-only allow-list"
        )

    def _returned_artifacts(self, result: Any) -> tuple[list[str], str | None]:
        if not isinstance(result, dict) or "artifacts" not in result:
            return [], None
        values = result.get("artifacts")
        if not isinstance(values, list) or not all(
            isinstance(item, str) for item in values
        ):
            return [], "literature tool returned an invalid artifact list"
        roots = tuple(root.resolve() for root in self.retrieval_artifact_roots)
        accepted: list[str] = []
        for value in values:
            try:
                path = Path(value).resolve(strict=True)
                info = path.lstat()
            except (OSError, RuntimeError):
                return [], "literature tool returned a missing artifact"
            if not stat.S_ISREG(info.st_mode):
                return [], "literature tool artifact is not a regular file"
            if not any(path == root or root in path.parents for root in roots):
                return [], "literature tool artifact escaped its workspace"
            accepted.append(str(path))
        return accepted, None

    def _preserve_tool_result(self, tool_name: str, encoded: bytes) -> Path | None:
        if self.evidence_dir is None:
            return None
        self.evidence_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._evidence_artifact_count += 1
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tool_name).strip("-.")
        artifact = self.evidence_dir / (
            f"tool-{self._evidence_artifact_count:03d}-{safe_name or 'result'}.json"
        )
        artifact.write_bytes(encoded + b"\n")
        artifact.chmod(0o600)
        return artifact

    def _model_observation(
        self,
        tool_name: str,
        result: Any,
        encoded: bytes,
        artifact: Path | None,
    ) -> dict | None:
        """Bound model context while preserving the complete result as evidence."""

        remaining = max(
            0, MAX_MODEL_TOOL_RESULTS_TOTAL_BYTES - self._model_observation_bytes
        )
        if len(encoded) <= MAX_MODEL_TOOL_RESULT_BYTES and len(encoded) <= remaining:
            self._model_observation_bytes += len(encoded)
            return None

        compact: dict[str, Any] = {
            "status": "completed",
            "result_compacted": True,
            "reason": (
                "The complete tool result was preserved as a run artifact but "
                "was not injected into model context in full."
            ),
            "result_bytes": len(encoded),
            "result_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        if artifact is not None:
            compact["full_result_artifact"] = str(artifact)

        # Browser screenshots are encoded binary data, not useful text previews.
        # Snapshot/search/document results retain a bounded leading preview.
        preview_budget = min(MAX_MODEL_TOOL_RESULT_BYTES, remaining) - 1_024
        if tool_name != "take_screenshot" and preview_budget > 1_024:
            preview = encoded[:preview_budget].decode("utf-8", errors="replace")
            compact["result_preview"] = preview
            while (
                len(json.dumps(compact, ensure_ascii=False).encode("utf-8"))
                > min(MAX_MODEL_TOOL_RESULT_BYTES, remaining)
                and compact["result_preview"]
            ):
                compact["result_preview"] = compact["result_preview"][
                    : len(compact["result_preview"]) // 2
                ]

        compact_bytes = len(json.dumps(compact, ensure_ascii=False).encode("utf-8"))
        self._model_observation_bytes += compact_bytes
        return compact

    @staticmethod
    def _logged_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        logged = dict(arguments)
        code = logged.get("code")
        if isinstance(code, str):
            encoded = code.encode("utf-8")
            logged["code"] = {
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
            }
        return logged

    def before_tool(
        self, tool, args: dict[str, Any], tool_context
    ) -> dict | None:  # ADK callback
        name = getattr(tool, "name", type(tool).__name__)
        arguments = args
        chrome_page_tools = {
            "navigate_page",
            "take_snapshot",
            "take_screenshot",
            "wait_for",
        }
        if name == "new_page":
            arguments["isolatedContext"] = self.chrome_context
        elif name in chrome_page_tools and not self.chrome_ready:
            self.ledger.append(
                "tool_policy",
                {
                    "tool_name": name,
                    "arguments": self._logged_arguments(arguments),
                    "decision": "deny",
                    "reason": "open an isolated new_page before using Chrome page tools",
                },
            )
            self._observe("tool_policy", name, "denied")
            return {
                "error": "POLICY_DENIED",
                "reason": "open an isolated new_page before using Chrome page tools",
            }
        allowed, reason = self.evaluate(name, arguments)
        action_id = hashlib.sha256(
            json.dumps(
                {"tool": name, "arguments": arguments}, sort_keys=True, default=str
            ).encode()
        ).hexdigest()[:16]
        self.ledger.append(
            "tool_policy",
            {
                "action_id": action_id,
                "tool_name": name,
                "arguments": self._logged_arguments(arguments),
                "decision": "allow" if allowed else "deny",
                "reason": reason,
            },
        )
        if allowed:
            self._observe("tool_policy", name, "allowed")
            return None
        self._observe("tool_policy", name, "denied")
        return {"error": "POLICY_DENIED", "reason": reason, "action_id": action_id}

    def after_tool(
        self, tool, args: dict[str, Any], tool_context, tool_response: Any
    ) -> dict | None:  # ADK callback
        name = getattr(tool, "name", type(tool).__name__)
        result = tool_response
        if name == "new_page" and not (
            isinstance(result, dict) and result.get("error")
        ):
            self.chrome_ready = True
        encoded = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        reported_error = result.get("error") if isinstance(result, dict) else None
        if (
            not reported_error
            and isinstance(result, dict)
            and result.get("status")
            in {"failed", "timed_out", "cancelled", "policy_denied"}
        ):
            reported_error = str(result["status"]).upper()
        if not reported_error and len(encoded) > MAX_RESULT_ARTIFACT_BYTES:
            reported_error = "RESULT_TOO_LARGE"
        artifact_error: str | None = None
        if name in RETRIEVAL_TOOLS and not reported_error:
            returned_artifacts, artifact_error = self._returned_artifacts(result)
            if artifact_error:
                reported_error = "INVALID_RETRIEVAL_ARTIFACT"
        else:
            returned_artifacts = []
        if name in RETRIEVAL_TOOLS and not reported_error:
            self.successful_retrievals += 1
            assert self.retrieval_tools is not None
            assert self.retrieved_urls is not None
            assert self.retrieval_dates is not None
            self.retrieval_tools.add(name)
            self.retrieved_urls.update(_extract_urls(result))
            self.retrieval_dates.add(utc_now()[:10])
            artifact = self._preserve_tool_result(name, encoded)
            if artifact is not None:
                assert self.retrieval_artifacts is not None
                self.retrieval_artifacts.append(str(artifact))
            self.retrieval_artifacts.extend(returned_artifacts)
        elif not reported_error and (
            len(encoded) > MAX_MODEL_TOOL_RESULT_BYTES
            or self._model_observation_bytes + len(encoded)
            > MAX_MODEL_TOOL_RESULTS_TOTAL_BYTES
        ):
            artifact = self._preserve_tool_result(name, encoded)
        else:
            artifact = None
        compact_result = (
            self._model_observation(name, result, encoded, artifact)
            if not reported_error
            else None
        )
        self.ledger.append(
            "tool_result",
            {
                "tool_name": name,
                "result_sha256": hashlib.sha256(encoded).hexdigest(),
                "result_bytes": len(encoded),
                "reported_error": reported_error,
                "retrieved_url_count": len(self.retrieved_urls or set()),
                "model_result_bytes": (
                    len(json.dumps(compact_result, default=str).encode("utf-8"))
                    if compact_result is not None
                    else len(encoded)
                ),
                "result_compacted": compact_result is not None,
            },
        )
        self._observe("tool_result", name, "failed" if reported_error else "completed")
        if reported_error == "RESULT_TOO_LARGE":
            return {
                "error": "RESULT_TOO_LARGE",
                "reason": (f"tool result exceeds {MAX_RESULT_ARTIFACT_BYTES} bytes"),
            }
        if reported_error == "INVALID_RETRIEVAL_ARTIFACT":
            return {
                "error": "INVALID_RETRIEVAL_ARTIFACT",
                "reason": artifact_error,
            }
        return compact_result


def default_allowed_tools(
    include_chrome: bool,
    enable_code: bool = False,
    enable_packages: bool = False,
) -> set[str]:
    allowed = set(READ_ONLY_TOOLS)
    allowed |= LITERATURE_TOOLS
    if not include_chrome:
        allowed -= {
            "new_page",
            "navigate_page",
            "take_snapshot",
            "take_screenshot",
            "wait_for",
        }
    if enable_code:
        allowed |= EXECUTION_TOOLS
    if enable_packages:
        allowed |= INSTALLATION_TOOLS
    return allowed
