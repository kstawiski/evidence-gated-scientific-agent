#!/usr/bin/env python3
"""Run a token-safe A2A 1.0 release gate against Evidence Bench."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from typing import IO, Mapping, Sequence
from urllib.parse import urlsplit

import httpx


DEFAULT_PROMPT = (
    "Prepare a concise evidence-gated scientific methods report comparing Welch's "
    "and Student's independent two-sample t-tests. Distinguish documented facts "
    "from interpretation and state the limitations of the comparison."
)
SCIENTIFIC_SUCCESS_STATES = {
    "supported",
    "supported_with_comments",
}
IMRAD = ("Introduction", "Methods", "Results", "Discussion", "Conclusions")
DEFAULT_MCP_SERVERS = ("context7", "brave-search", "chrome-devtools")


class GateError(RuntimeError):
    """A deployed A2A release-gate check failed."""


@dataclass(frozen=True)
class GateConfig:
    base_url: str
    token: str
    timeout_seconds: float
    prompt: str
    mcp_servers: tuple[str, ...] = DEFAULT_MCP_SERVERS
    enable_code: bool = False


def _boolean(value: str | None) -> bool:
    if value is None or value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    if value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    raise GateError("A2A_ENABLE_CODE must be a boolean")


def _mcp_names(values: Sequence[str] | None, env_value: str) -> tuple[str, ...]:
    if values is None and not env_value.strip():
        return DEFAULT_MCP_SERVERS
    source = values if values is not None else (env_value,)
    names = [name.strip() for value in source for name in value.split(",")]
    return tuple(dict.fromkeys(name for name in names if name))


def config_from_args(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> GateConfig:
    env = os.environ if environ is None else environ
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="env: A2A_BASE_URL")
    parser.add_argument(
        "--timeout-seconds", type=float, help="env: A2A_TIMEOUT_SECONDS"
    )
    parser.add_argument("--prompt", help="env: A2A_PROMPT")
    parser.add_argument(
        "--mcp-server",
        "--mcp",
        action="append",
        dest="mcp_servers",
        help="repeat or comma-separate; env: A2A_MCP_SERVERS",
    )
    parser.add_argument(
        "--enable-code",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="env: A2A_ENABLE_CODE",
    )
    args = parser.parse_args(argv)

    token = env.get("A2A_TOKEN", "")
    if not token:
        raise GateError("A2A_TOKEN is required")
    base_url = (
        args.base_url or env.get("A2A_BASE_URL", "http://127.0.0.1:8080")
    ).rstrip("/")
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise GateError("A2A base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise GateError("A2A base URL must not contain credentials")
    try:
        timeout = args.timeout_seconds or float(env.get("A2A_TIMEOUT_SECONDS", "2700"))
    except ValueError as exc:
        raise GateError("A2A_TIMEOUT_SECONDS must be numeric") from exc
    if timeout <= 0:
        raise GateError("A2A timeout must be positive")
    prompt = (args.prompt or env.get("A2A_PROMPT", DEFAULT_PROMPT)).strip()
    if not prompt:
        raise GateError("A2A prompt must not be empty")
    enable_code = (
        args.enable_code
        if args.enable_code is not None
        else _boolean(env.get("A2A_ENABLE_CODE"))
    )
    return GateConfig(
        base_url=base_url,
        token=token,
        timeout_seconds=timeout,
        prompt=prompt,
        mcp_servers=_mcp_names(args.mcp_servers, env.get("A2A_MCP_SERVERS", "")),
        enable_code=enable_code,
    )


def _rpc_result(response: httpx.Response) -> dict:
    try:
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise GateError(f"invalid A2A response: {exc}") from exc
    if not isinstance(payload, dict):
        raise GateError("A2A response is not a JSON object")
    if "error" in payload:
        error = payload["error"]
        message = error.get("message") if isinstance(error, dict) else None
        raise GateError(message or "A2A JSON-RPC error")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise GateError("A2A response has no object result")
    return result


def run_gate(
    config: GateConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict:
    """Execute the gate and return only non-secret verification evidence."""

    timeout = httpx.Timeout(
        config.timeout_seconds, connect=min(30.0, config.timeout_seconds)
    )
    with httpx.Client(timeout=timeout, transport=transport) as client:
        try:
            card_response = client.get(f"{config.base_url}/.well-known/agent-card.json")
            card_response.raise_for_status()
            card = card_response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise GateError(f"invalid Agent Card response: {exc}") from exc
        interfaces = (
            card.get("supportedInterfaces", []) if isinstance(card, dict) else []
        )
        interface = next(
            (
                item
                for item in interfaces
                if isinstance(item, dict)
                and item.get("protocolBinding") == "JSONRPC"
                and item.get("protocolVersion") == "1.0"
            ),
            None,
        )
        if interface is None:
            raise GateError("Agent Card does not advertise JSONRPC A2A 1.0")
        if (card.get("capabilities") or {}).get("streaming") is not True:
            raise GateError("Agent Card does not advertise streaming")

        headers = {
            "Authorization": f"Bearer {config.token}",
            "A2A-Version": "1.0",
            "Content-Type": "application/json",
        }
        stream_request = {
            "jsonrpc": "2.0",
            "id": "a2a-live-gate-stream",
            "method": "SendStreamingMessage",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "ROLE_USER",
                    "parts": [{"text": config.prompt}],
                    "metadata": {
                        "enable_code": config.enable_code,
                        "mcp_servers": list(config.mcp_servers),
                    },
                },
                "configuration": {},
            },
        }
        states: list[str] = []
        streamed_artifacts: set[str] = set()
        task_id: str | None = None
        try:
            with client.stream(
                "POST",
                f"{config.base_url}/a2a",
                headers={**headers, "Accept": "text/event-stream"},
                json=stream_request,
            ) as response:
                response.raise_for_status()
                if not response.headers.get("content-type", "").startswith(
                    "text/event-stream"
                ):
                    raise GateError("SendStreamingMessage did not return SSE")
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    try:
                        envelope = json.loads(line.removeprefix("data: "))
                    except ValueError as exc:
                        raise GateError("A2A stream emitted invalid JSON") from exc
                    if "error" in envelope:
                        error = envelope["error"]
                        message = (
                            error.get("message") if isinstance(error, dict) else None
                        )
                        raise GateError(message or "A2A stream emitted an error")
                    result = envelope.get("result", {})
                    if isinstance(result.get("task"), dict):
                        event_task = result["task"]
                        assigned = event_task.get("id")
                        if isinstance(assigned, str):
                            if task_id is not None and assigned != task_id:
                                raise GateError("A2A stream changed task ID")
                            task_id = assigned
                        state = (event_task.get("status") or {}).get("state")
                        if isinstance(state, str):
                            states.append(state)
                    elif isinstance(result.get("statusUpdate"), dict):
                        state = (result["statusUpdate"].get("status") or {}).get(
                            "state"
                        )
                        if isinstance(state, str):
                            states.append(state)
                    elif isinstance(result.get("artifactUpdate"), dict):
                        name = (result["artifactUpdate"].get("artifact") or {}).get(
                            "name"
                        )
                        if isinstance(name, str):
                            streamed_artifacts.add(name)
        except httpx.HTTPError as exc:
            raise GateError(f"A2A stream failed: {exc}") from exc

        if task_id is None:
            raise GateError("server did not assign a task ID")
        if not {"TASK_STATE_SUBMITTED", "TASK_STATE_WORKING"}.issubset(states):
            raise GateError(f"A2A stream is missing lifecycle states: {states}")
        if states[-1:] != ["TASK_STATE_COMPLETED"]:
            raise GateError(f"A2A stream did not complete successfully: {states}")
        required_artifacts = {"run-summary.json", "report.md"}
        if not required_artifacts.issubset(streamed_artifacts):
            raise GateError("A2A stream did not announce both required artifacts")

        task = _rpc_result(
            client.post(
                f"{config.base_url}/a2a",
                headers=headers,
                json={
                    "jsonrpc": "2.0",
                    "id": "a2a-live-gate-get-task",
                    "method": "GetTask",
                    "params": {"id": task_id},
                },
            )
        )
        if task.get("id") != task_id:
            raise GateError("GetTask returned a different task ID")
        if (task.get("status") or {}).get("state") != "TASK_STATE_COMPLETED":
            raise GateError("GetTask did not return a completed task")
        artifacts = {
            item.get("name"): item
            for item in task.get("artifacts", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        if not required_artifacts.issubset(artifacts):
            raise GateError("GetTask is missing run-summary.json or report.md")
        try:
            summary = next(
                part["data"]
                for part in artifacts["run-summary.json"]["parts"]
                if "data" in part
            )
            report = next(
                part["text"]
                for part in artifacts["report.md"]["parts"]
                if "text" in part
            )
        except (KeyError, StopIteration, TypeError) as exc:
            raise GateError("required A2A artifact has no usable content") from exc
        if not isinstance(summary, dict) or not isinstance(report, str):
            raise GateError("required A2A artifact has the wrong content type")
        scientific_status = summary.get("status")
        if scientific_status not in SCIENTIFIC_SUCCESS_STATES:
            raise GateError(f"non-success scientific status: {scientific_status}")
        missing = [
            heading
            for heading in IMRAD
            if not re.search(
                rf"^#{{1,6}}\s+{heading}\s*$",
                report,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        ]
        if missing:
            raise GateError(f"report.md is missing IMRaD headings: {missing}")

    return {
        "passed": True,
        "agent_version": card.get("version"),
        "protocol_binding": interface["protocolBinding"],
        "protocol_version": interface["protocolVersion"],
        "task_id": task_id,
        "run_id": summary.get("run_id"),
        "workspace_id": summary.get("workspace_id"),
        "scientific_status": scientific_status,
        "stream_states": states,
        "artifacts": sorted(artifacts),
        "mcp_servers": list(config.mcp_servers),
        "enable_code": config.enable_code,
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    stdout: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> int:
    stdout, stderr = stdout or sys.stdout, stderr or sys.stderr
    token = ""
    try:
        config = config_from_args(argv, environ)
        token = config.token
        result = run_gate(config, transport=transport)
    except (GateError, httpx.HTTPError) as exc:
        message = str(exc).replace(token, "<redacted>") if token else str(exc)
        print(json.dumps({"passed": False, "error": message}), file=stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), file=stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
