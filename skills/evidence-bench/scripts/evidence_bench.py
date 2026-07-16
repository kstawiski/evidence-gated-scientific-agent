#!/usr/bin/env python3
"""Small, dependency-free client for the internal Evidence Bench web API."""

from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://10.20.102.122"
DEFAULT_MCPS = ("context7", "brave-search", "chrome-devtools")
TERMINAL_STATES = {
    "supported",
    "supported_with_comments",
    "contradicted",
    "inconclusive",
    "requires_more_evidence",
    "requires_human_decision",
    "failed",
    "cancelled",
    "interrupted",
}


class ClientError(RuntimeError):
    pass


class Client:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ClientError("base URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ClientError("credentials are not accepted in the base URL")
        self._parsed = parsed

    def _url(self, path: str) -> str:
        prefix = self._parsed.path.rstrip("/")
        return f"{self._parsed.scheme}://{self._parsed.netloc}{prefix}{path}"

    def json_request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self._url(path), data=data, headers=headers, method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except Exception as exc:
            raise ClientError(f"{method} {path} failed: {exc}") from exc

    def download(self, path: str, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        request = Request(
            self._url(path), headers={"Accept": "application/octet-stream"}
        )
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.part")
        try:
            with urlopen(request, timeout=max(self.timeout, 300.0)) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            temporary.replace(output)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            raise ClientError(f"download failed: {exc}") from exc

    def upload(self, workspace_id: str, source: Path) -> dict[str, Any]:
        source = source.expanduser()
        if source.is_symlink() or not source.is_file():
            raise ClientError(
                f"upload input must be a regular non-symlink file: {source}"
            )
        boundary = f"evidence-bench-{uuid.uuid4().hex}"
        filename = source.name.replace('"', "_")
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        preamble = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="upload"; filename="{filename}"\r\n'
            f"Content-Type: {media_type}\r\n\r\n"
        ).encode("utf-8")
        closing = f"\r\n--{boundary}--\r\n".encode("ascii")
        connection_type = (
            http.client.HTTPSConnection
            if self._parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_type(
            self._parsed.hostname,
            self._parsed.port,
            timeout=max(self.timeout, 300.0),
        )
        prefix = self._parsed.path.rstrip("/")
        endpoint = f"{prefix}/api/workspaces/{quote(workspace_id)}/files"
        try:
            connection.putrequest("POST", endpoint)
            connection.putheader(
                "Content-Type", f"multipart/form-data; boundary={boundary}"
            )
            connection.putheader(
                "Content-Length",
                str(len(preamble) + source.stat().st_size + len(closing)),
            )
            connection.putheader("Accept", "application/json")
            connection.endheaders()
            connection.send(preamble)
            with source.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(closing)
            response = connection.getresponse()
            body = response.read()
            if response.status >= 400:
                raise ClientError(
                    f"upload failed with HTTP {response.status}: "
                    f"{body[:500].decode('utf-8', errors='replace')}"
                )
            return json.loads(body)
        except ClientError:
            raise
        except Exception as exc:
            raise ClientError(f"upload failed: {exc}") from exc
        finally:
            connection.close()


def _objective(args: argparse.Namespace, field: str) -> str:
    value = getattr(args, field)
    file_value = getattr(args, f"{field}_file")
    if file_value:
        value = Path(file_value).read_text(encoding="utf-8")
    value = (value or "").strip()
    if not value:
        raise ClientError(
            f"--{field.replace('_', '-')} or --{field.replace('_', '-')}-file is required"
        )
    return value


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _wait(
    client: Client,
    run_id: str,
    poll_seconds: float,
    download_dir: Path | None,
) -> dict[str, Any]:
    after_id = 0
    while True:
        events = client.json_request(
            "GET",
            f"/api/runs/{quote(run_id)}/events?{urlencode({'after_id': after_id})}",
        )
        for event in events:
            after_id = max(after_id, int(event.get("id", 0)))
            created = str(event.get("created_at", ""))[11:19]
            actor = event.get("actor", "Controller")
            phase = event.get("phase", "")
            message = event.get("message", "")
            print(f"[{created}] {actor} · {phase}: {message}", file=sys.stderr)
        detail = client.json_request("GET", f"/api/runs/{quote(run_id)}")
        if detail.get("status") in TERMINAL_STATES:
            if download_dir is not None:
                download_dir.mkdir(parents=True, exist_ok=True)
                bundle = download_dir / f"evidence-bench-{run_id}.zip"
                client.download(f"/api/runs/{quote(run_id)}/bundle", bundle)
                detail["downloaded_bundle"] = str(bundle.resolve())
                report_artifact = next(
                    (
                        item.get("path")
                        for item in detail.get("artifacts", [])
                        if item.get("path") == "report.md"
                    ),
                    None,
                )
                if report_artifact:
                    report = download_dir / "report.md"
                    query = urlencode({"path": report_artifact})
                    client.download(
                        f"/api/runs/{quote(run_id)}/artifacts?{query}", report
                    )
                    detail["downloaded_report"] = str(report.resolve())
            return detail
        time.sleep(poll_seconds)


def _add_text_input(parser: argparse.ArgumentParser, name: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(f"--{name.replace('_', '-')}")
    group.add_argument(f"--{name.replace('_', '-')}-file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVIDENCE_BENCH_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="create a workspace and run an analysis")
    _add_text_input(run, "objective")
    run.add_argument("--workspace-id")
    run.add_argument("--workspace-name", default="Agent scientific task")
    run.add_argument("--file", action="append", default=[])
    run.add_argument("--no-code", action="store_true")
    run.add_argument("--mcp", action="append", choices=DEFAULT_MCPS)
    run.add_argument("--no-research", action="store_true")
    run.add_argument("--wait", action="store_true")
    run.add_argument("--poll-seconds", type=float, default=3.0)
    run.add_argument("--download-dir", type=Path)

    follow = commands.add_parser("follow-up", help="create an audited child revision")
    follow.add_argument("--run-id", required=True)
    _add_text_input(follow, "request")
    follow.add_argument("--enable-code", action="store_true")
    follow.add_argument("--wait", action="store_true")
    follow.add_argument("--poll-seconds", type=float, default=3.0)
    follow.add_argument("--download-dir", type=Path)

    discuss = commands.add_parser("discuss", help="ask Gemma about a completed report")
    discuss.add_argument("--run-id", required=True)
    _add_text_input(discuss, "message")

    status = commands.add_parser("status", help="show a run")
    status.add_argument("--run-id", required=True)

    cancel = commands.add_parser("cancel", help="request cancellation")
    cancel.add_argument("--run-id", required=True)

    download = commands.add_parser("download", help="download a provenance bundle")
    download.add_argument("--run-id", required=True)
    download.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.timeout <= 0:
            raise ClientError("--timeout must be positive")
        client = Client(args.base_url, args.timeout)
        if args.command == "run":
            workspace_id = args.workspace_id
            if not workspace_id:
                workspace = client.json_request(
                    "POST", "/api/workspaces", {"name": args.workspace_name}
                )
                workspace_id = workspace["id"]
            for filename in args.file:
                client.upload(workspace_id, Path(filename))
            mcps = (
                []
                if args.no_research
                else list(dict.fromkeys(args.mcp or DEFAULT_MCPS))
            )
            run = client.json_request(
                "POST",
                f"/api/workspaces/{quote(workspace_id)}/runs",
                {
                    "objective": _objective(args, "objective"),
                    "enable_code": not args.no_code,
                    "mcp_servers": mcps,
                },
            )
            print(
                f"Submitted workspace={workspace_id} run={run['id']}", file=sys.stderr
            )
            result = (
                _wait(client, run["id"], args.poll_seconds, args.download_dir)
                if args.wait
                else run
            )
        elif args.command == "follow-up":
            run = client.json_request(
                "POST",
                f"/api/runs/{quote(args.run_id)}/follow-ups",
                {
                    "request": _objective(args, "request"),
                    "enable_code": args.enable_code,
                },
            )
            print(f"Submitted revision run={run['id']}", file=sys.stderr)
            result = (
                _wait(client, run["id"], args.poll_seconds, args.download_dir)
                if args.wait
                else run
            )
        elif args.command == "discuss":
            result = client.json_request(
                "POST",
                f"/api/runs/{quote(args.run_id)}/discussion",
                {"message": _objective(args, "message")},
            )
        elif args.command == "status":
            result = client.json_request("GET", f"/api/runs/{quote(args.run_id)}")
        elif args.command == "cancel":
            result = client.json_request(
                "POST", f"/api/runs/{quote(args.run_id)}/cancel"
            )
        else:
            client.download(
                f"/api/runs/{quote(args.run_id)}/bundle", args.output.expanduser()
            )
            result = {"run_id": args.run_id, "bundle": str(args.output.resolve())}
        _emit(result)
        return 0
    except (ClientError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
