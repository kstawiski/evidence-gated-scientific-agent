#!/usr/bin/env python3
"""Submit one text task to an Evidence Bench A2A 1.0 endpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


DEFAULT_MCPS = ("context7", "brave-search", "chrome-devtools")


class ClientError(RuntimeError):
    """A local configuration or A2A transport error."""


def _base_url(value: str) -> str:
    value = value.rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ClientError("base URL must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ClientError("credentials must not be embedded in the base URL")
    return value


def _objective(args: argparse.Namespace) -> str:
    value = args.objective
    if args.objective_file:
        value = Path(args.objective_file).read_text(encoding="utf-8")
    value = (value or "").strip()
    if not value:
        raise ClientError("--objective or --objective-file is required")
    return value


def _json_request(url: str, token: str, payload: dict, timeout: float) -> dict:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "A2A-Version": "1.0",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except HTTPError as exc:
        detail = exc.read(1000).decode("utf-8", errors="replace")
        raise ClientError(f"A2A request failed with HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, ValueError) as exc:
        raise ClientError(f"A2A request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise ClientError("A2A response is not a JSON object")
    if "error" in result:
        error = result["error"]
        message = error.get("message") if isinstance(error, dict) else None
        raise ClientError(message or "A2A returned an error")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVIDENCE_BENCH_URL", "http://127.0.0.1:8080"),
    )
    objective = parser.add_mutually_exclusive_group(required=True)
    objective.add_argument("--objective")
    objective.add_argument("--objective-file")
    parser.add_argument("--context-id", default=None)
    parser.add_argument("--enable-code", action="store_true")
    parser.add_argument("--mcp", action="append", choices=DEFAULT_MCPS)
    parser.add_argument("--no-research", action="store_true")
    parser.add_argument("--timeout", type=float, default=7200.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.timeout <= 0:
            raise ClientError("timeout must be positive")
        token = os.environ.get("A2A_TOKEN", "")
        if not token:
            raise ClientError("A2A_TOKEN is required")
        base_url = _base_url(args.base_url)
        mcps = [] if args.no_research else (args.mcp or list(DEFAULT_MCPS))
        message = {
            "messageId": str(uuid.uuid4()),
            "role": "ROLE_USER",
            "parts": [{"text": _objective(args)}],
            "metadata": {
                "enable_code": args.enable_code,
                "mcp_servers": mcps,
            },
        }
        if args.context_id:
            message["contextId"] = args.context_id
        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "SendMessage",
            "params": {
                "message": message,
                "configuration": {"returnImmediately": False},
            },
        }
        result = _json_request(f"{base_url}/a2a", token, payload, args.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (ClientError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
