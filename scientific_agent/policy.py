"""Deterministic gate for every model-requested tool call."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import socket
import uuid
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
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
RETRIEVAL_TOOLS = {
    "query-docs",
    "brave_web_search",
    "brave_llm_context",
}
EXECUTION_TOOLS = {"run_python_analysis", "run_r_analysis"}
_URL = re.compile(r"https?://[^\s<>\"']+")


def _extract_urls(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(*(_extract_urls(item) for item in value.values())) if value else set()
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
    evidence_dir: Path | None = None

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
            else "read-only allow-list"
        )

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
            return {
                "error": "POLICY_DENIED",
                "reason": "open an isolated new_page before using Chrome page tools",
            }
        allowed, reason = self.evaluate(name, arguments)
        action_id = hashlib.sha256(
            json.dumps({"tool": name, "arguments": arguments}, sort_keys=True, default=str).encode()
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
            return None
        return {"error": "POLICY_DENIED", "reason": reason, "action_id": action_id}

    def after_tool(
        self, tool, args: dict[str, Any], tool_context, tool_response: Any
    ) -> dict | None:  # ADK callback
        name = getattr(tool, "name", type(tool).__name__)
        arguments = args
        result = tool_response
        if name == "new_page" and not (isinstance(result, dict) and result.get("error")):
            self.chrome_ready = True
        encoded = json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        reported_error = result.get("error") if isinstance(result, dict) else None
        if name in RETRIEVAL_TOOLS and not reported_error and len(encoded) > MAX_RESULT_ARTIFACT_BYTES:
            reported_error = "RESULT_TOO_LARGE"
        if name in RETRIEVAL_TOOLS and not reported_error:
            self.successful_retrievals += 1
            assert self.retrieval_tools is not None
            assert self.retrieved_urls is not None
            assert self.retrieval_dates is not None
            self.retrieval_tools.add(name)
            self.retrieved_urls.update(_extract_urls(result))
            self.retrieval_dates.add(utc_now()[:10])
            if self.evidence_dir is not None:
                self.evidence_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
                assert self.retrieval_artifacts is not None
                artifact = self.evidence_dir / (
                    f"tool-{len(self.retrieval_artifacts) + 1:03d}-{name}.json"
                )
                artifact.write_bytes(encoded + b"\n")
                artifact.chmod(0o600)
                self.retrieval_artifacts.append(str(artifact))
        self.ledger.append(
            "tool_result",
            {
                "tool_name": name,
                "result_sha256": hashlib.sha256(encoded).hexdigest(),
                "result_bytes": len(encoded),
                "reported_error": reported_error,
                "retrieved_url_count": len(self.retrieved_urls or set()),
            },
        )
        if reported_error == "RESULT_TOO_LARGE":
            return {
                "error": "RESULT_TOO_LARGE",
                "reason": (
                    f"retrieval result exceeds {MAX_RESULT_ARTIFACT_BYTES} bytes"
                ),
            }
        return None


def default_allowed_tools(include_chrome: bool, enable_code: bool = False) -> set[str]:
    allowed = set(READ_ONLY_TOOLS)
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
    return allowed
