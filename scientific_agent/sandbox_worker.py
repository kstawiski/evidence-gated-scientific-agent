"""Internal, token-authenticated Python/R sandbox worker for Docker deployments."""

from __future__ import annotations

import json
import os
import secrets
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import SandboxSettings
from .execution import AnalysisExecutor


class ExecuteRequest(BaseModel):
    workspace: str
    computation_root: str
    language: Literal["python", "r"]
    code: str = Field(max_length=128 * 1024)
    timeout_seconds: int = Field(ge=1, le=3600)


@dataclass
class WorkerState:
    data_dir: Path
    token: str
    settings: SandboxSettings
    environments_dir: Path | None = None
    output_uid: int = 10001
    output_gid: int = 10001
    executors: dict[str, AnalysisExecutor] = field(default_factory=dict)
    staging_roots: dict[str, Path] = field(default_factory=dict)
    executor_locks: dict[str, threading.Lock] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def confined_paths(self, request: ExecuteRequest) -> tuple[Path, Path]:
        data = self.data_dir.resolve()
        workspace = Path(request.workspace).resolve()
        root = Path(request.computation_root).resolve()
        try:
            workspace_parts = workspace.relative_to(data).parts
            root_parts = root.relative_to(data).parts
        except ValueError as exc:
            raise ValueError("worker paths must remain below the data directory") from exc
        valid_workspace = (
            len(workspace_parts) == 3
            and workspace_parts[0] == "workspaces"
            and workspace_parts[2] == "files"
        )
        valid_root = (
            len(root_parts) >= 4
            and root_parts[0] == "workspaces"
            and root_parts[1] == workspace_parts[1]
            and root_parts[2] == "runs"
        )
        try:
            uuid.UUID(workspace_parts[1] if valid_workspace else "")
        except ValueError as exc:
            raise ValueError("invalid workspace path") from exc
        if not valid_workspace or not valid_root or not workspace.is_dir():
            raise ValueError("invalid workspace or computation path")
        return workspace, root

    def execute(self, request: ExecuteRequest) -> dict:
        workspace, root = self.confined_paths(request)
        # The first attempt has no prior computation directory yet, but bubblewrap
        # still needs an existing read-only bind source for /history.
        root.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        key = str(root)
        with self.lock:
            executor = self.executors.get(key)
            if executor is None:
                staging = Path(tempfile.mkdtemp(prefix="evidence-worker-"))
                staged_workspace = staging / "workspace"
                staged_root = staging / "computations"
                staged_workspace.mkdir(mode=0o700)
                for source in workspace.iterdir():
                    if source.is_symlink() or not source.is_file():
                        raise ValueError("workspace inputs must be flat regular files")
                    shutil.copy2(source, staged_workspace / source.name)
                environment_dir = None
                if self.environments_dir is not None:
                    candidate = (
                        self.environments_dir.resolve() / workspace.parent.name
                    ).resolve()
                    if candidate.parent != self.environments_dir.resolve():
                        raise ValueError("invalid workspace environment path")
                    environment_dir = candidate
                executor = AnalysisExecutor(
                    staged_workspace,
                    staged_root,
                    self.settings,
                    environment_dir=environment_dir,
                    history_dir=root.parent,
                )
                self.executors[key] = executor
                self.staging_roots[key] = staged_root
                self.executor_locks[key] = threading.Lock()
            executor_lock = self.executor_locks[key]
        with executor_lock:
            result = executor.execute(
                request.language,
                request.code,
                request.timeout_seconds,
            )
            staged_root = self.staging_roots[key]
            shutil.copytree(staged_root, root, dirs_exist_ok=True)
            self._handoff_ownership(root)
            result = self._rewrite_paths(result, staged_root, root)
            evidence = self._rewrite_paths(
                executor.evidence().model_dump(mode="json"), staged_root, root
            )
            return {
                "result": result,
                "evidence": evidence,
            }

    def _handoff_ownership(self, root: Path) -> None:
        """Return worker-created provenance to the unprivileged web service."""

        for path in [*root.rglob("*"), root]:
            if not path.is_symlink():
                try:
                    os.chown(path, self.output_uid, self.output_gid)
                except PermissionError:
                    # Root-squashed NFS can map both container UIDs to the same
                    # NAS owner while denying explicit chown. The web service's
                    # later artifact read remains the effective access check.
                    continue

    def _rewrite_paths(self, value, source: Path, destination: Path):
        if isinstance(value, dict):
            return {
                key: self._rewrite_paths(item, source, destination)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self._rewrite_paths(item, source, destination) for item in value]
        if isinstance(value, str) and value.startswith(f"{source}/"):
            return f"{destination}/{value.removeprefix(f'{source}/')}"
        return value


def _handler(state: WorkerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvidenceBenchSandbox/0.3"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._json(404, {"detail": "not found"})
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/execute":
                self._json(404, {"detail": "not found"})
                return
            expected = f"Bearer {state.token}"
            actual = self.headers.get("Authorization", "")
            if not secrets.compare_digest(actual, expected):
                self._json(401, {"detail": "valid worker token required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"detail": "invalid content length"})
                return
            if size < 1 or size > state.settings.max_code_bytes + 32 * 1024:
                self._json(413, {"detail": "request body exceeds worker limit"})
                return
            try:
                request = ExecuteRequest.model_validate_json(self.rfile.read(size))
                payload = state.execute(request)
            except (ValidationError, ValueError) as exc:
                self._json(400, {"detail": str(exc)})
                return
            except Exception as exc:
                self._json(500, {"detail": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, payload)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            del format, args

    return Handler


def main() -> None:
    token = os.environ.get("SCIENTIFIC_AGENT_SANDBOX_WORKER_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("SCIENTIFIC_AGENT_SANDBOX_WORKER_TOKEN must be at least 24 characters")
    data_dir = Path(os.environ.get("SCIENTIFIC_AGENT_DATA_DIR", "/data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    environments_dir = Path(
        os.environ.get("SCIENTIFIC_AGENT_ENVIRONMENTS_DIR", "/environments")
    ).resolve()
    settings = replace(SandboxSettings(), worker_url="", worker_token="")
    state = WorkerState(
        data_dir,
        token,
        settings,
        environments_dir=environments_dir,
        output_uid=int(os.environ.get("SANDBOX_OUTPUT_UID", "10001")),
        output_gid=int(os.environ.get("SANDBOX_OUTPUT_GID", "10001")),
    )
    host = os.environ.get("SANDBOX_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("SANDBOX_WORKER_PORT", "8090"))
    server = ThreadingHTTPServer((host, port), _handler(state))
    server.serve_forever()


if __name__ == "__main__":
    main()
