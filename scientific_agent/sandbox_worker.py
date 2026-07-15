"""Internal, token-authenticated Python/R sandbox worker for Docker deployments."""

from __future__ import annotations

import json
import os
import secrets
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import SandboxSettings
from .execution import AnalysisExecutor
from .provenance import sha256_file
from .reporting import _parse_tesseract_tsv


PDF_TEXT_MAX_BYTES = 16 * 1024 * 1024
PDF_PARSE_TIMEOUT_SECONDS = 90
OCR_TEXT_MAX_BYTES = 4 * 1024 * 1024
OCR_PARSE_TIMEOUT_SECONDS = 30


class ExecuteRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    workspace: str
    computation_root: str
    language: Literal["python", "r"]
    code: str = Field(max_length=128 * 1024)
    timeout_seconds: int = Field(ge=1, le=3600)
    max_calls_per_attempt: int | None = Field(default=None, ge=1)


class CancelRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")


class ReleaseRequest(BaseModel):
    workspace: str
    computation_root: str


class PdfExtractRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    pdf_path: str = Field(min_length=1, max_length=4096)


class FigureOcrRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
    figure_path: str = Field(min_length=1, max_length=4096)


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
    cancellation_events: dict[str, threading.Event] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def confined_paths(
        self, request: ExecuteRequest | ReleaseRequest
    ) -> tuple[Path, Path]:
        data = self.data_dir.resolve()
        workspace = Path(request.workspace).resolve()
        root = Path(request.computation_root).resolve()
        try:
            workspace_parts = workspace.relative_to(data).parts
            root_parts = root.relative_to(data).parts
        except ValueError as exc:
            raise ValueError(
                "worker paths must remain below the data directory"
            ) from exc
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

    def confined_pdf_path(self, request: PdfExtractRequest) -> Path:
        """Accept exactly one regular PDF in a UUID workspace reference directory."""

        try:
            uuid.UUID(request.request_id)
        except ValueError as exc:
            raise ValueError("invalid PDF extraction request ID") from exc
        data = self.data_dir.resolve()
        raw = Path(request.pdf_path)
        if not raw.is_absolute() or ".." in raw.parts:
            raise ValueError("PDF path must be absolute and traversal-free")
        try:
            parts = raw.relative_to(data).parts
        except ValueError as exc:
            raise ValueError("PDF path must remain below the data directory") from exc
        valid_shape = (
            len(parts) == 6
            and parts[0] == "workspaces"
            and parts[2:5] == ("files", "references", "pdfs")
            and Path(parts[5]).name == parts[5]
            and Path(parts[5]).suffix.lower() == ".pdf"
        )
        try:
            uuid.UUID(parts[1] if valid_shape else "")
        except ValueError as exc:
            raise ValueError("invalid workspace PDF path") from exc
        if not valid_shape:
            raise ValueError("invalid workspace PDF path")
        candidate = data
        for part in parts:
            candidate = candidate / part
            try:
                info = candidate.lstat()
            except OSError as exc:
                raise ValueError("workspace PDF path does not exist") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("workspace PDF path must not contain symlinks")
        info = raw.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("workspace PDF must be a regular file")
        if info.st_size > self.settings.max_file_bytes:
            raise ValueError("workspace PDF exceeds the worker size limit")
        resolved = raw.resolve(strict=True)
        expected_parent = (
            data / "workspaces" / parts[1] / "files" / "references" / "pdfs"
        )
        if resolved.parent != expected_parent or resolved != raw:
            raise ValueError("workspace PDF path escaped its reference directory")
        return resolved

    def confined_figure_path(self, request: FigureOcrRequest) -> Path:
        """Accept one generated raster below a UUID workspace run."""

        try:
            uuid.UUID(request.request_id)
        except ValueError as exc:
            raise ValueError("invalid OCR request ID") from exc
        data = self.data_dir.resolve()
        raw = Path(request.figure_path)
        if not raw.is_absolute() or ".." in raw.parts:
            raise ValueError("figure path must be absolute and traversal-free")
        try:
            parts = raw.relative_to(data).parts
        except ValueError as exc:
            raise ValueError(
                "figure path must remain below the data directory"
            ) from exc
        output_indexes = [
            index
            for index in range(len(parts) - 1)
            if parts[index : index + 2] == ("output", "figures")
        ]
        valid_shape = (
            len(parts) >= 9
            and parts[0] == "workspaces"
            and parts[2] == "runs"
            and parts[4] == "computations"
            and bool(output_indexes)
            and Path(parts[-1]).name == parts[-1]
            and Path(parts[-1]).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        try:
            uuid.UUID(parts[1] if valid_shape else "")
        except ValueError as exc:
            raise ValueError("invalid workspace figure path") from exc
        if not valid_shape:
            raise ValueError("invalid workspace figure path")
        candidate = data
        for part in parts:
            candidate /= part
            try:
                info = candidate.lstat()
            except OSError as exc:
                raise ValueError("workspace figure path does not exist") from exc
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("workspace figure path must not contain symlinks")
        info = raw.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("workspace figure must be a regular file")
        if info.st_size > self.settings.max_file_bytes:
            raise ValueError("workspace figure exceeds the worker size limit")
        resolved = raw.resolve(strict=True)
        if resolved != raw:
            raise ValueError("workspace figure path escaped its run directory")
        return resolved

    def _pdf_bwrap_command(self, pdf: Path, output_dir: Path) -> list[str]:
        pdftotext = Path("/usr/bin/pdftotext")
        required = [
            self.settings.bwrap,
            self.settings.prlimit,
            pdftotext,
            Path("/usr/lib"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                f"PDF sandbox runtime paths are missing: {', '.join(missing)}"
            )
        sandbox = [
            str(self.settings.bwrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--dir",
            "/usr",
            "--dir",
            "/usr/bin",
            "--ro-bind",
            str(pdftotext),
            "/usr/bin/pdftotext",
            "--ro-bind",
            "/usr/lib",
            "/usr/lib",
            "--symlink",
            "usr/lib",
            "/lib",
        ]
        if Path("/usr/lib64").exists():
            sandbox.extend(
                [
                    "--ro-bind",
                    "/usr/lib64",
                    "/usr/lib64",
                    "--symlink",
                    "usr/lib64",
                    "/lib64",
                ]
            )
        sandbox.extend(["--dir", "/etc"])
        created_dirs = {"/usr", "/usr/bin", "/usr/lib", "/usr/lib64", "/etc"}
        for source in (
            "/etc/ld.so.cache",
            "/etc/fonts",
            "/usr/share/fonts",
            "/usr/share/poppler",
            "/var/cache/fontconfig",
        ):
            path = Path(source)
            if not path.exists():
                continue
            parent = path.parent
            # Bubblewrap creates only the narrow parent hierarchy needed for the
            # allow-listed runtime data; no host /etc, /usr/bin, or /var is exposed.
            current = Path("/")
            for part in parent.parts[1:]:
                current /= part
                if str(current) not in created_dirs:
                    sandbox.extend(["--dir", str(current)])
                    created_dirs.add(str(current))
            sandbox.extend(["--ro-bind", source, source])
        sandbox.extend(
            [
                "--dir",
                "/dev",
                "--dev-bind",
                "/dev/null",
                "/dev/null",
                "--dir",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
                "--dir",
                "/input",
                "--ro-bind",
                str(pdf),
                "/input/article.pdf",
                "--dir",
                "/output",
                "--bind",
                str(output_dir),
                "/output",
                "--clearenv",
                "--setenv",
                "HOME",
                "/tmp/home",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "PATH",
                "/usr/bin",
                "--chdir",
                "/output",
                "/usr/bin/pdftotext",
                "-layout",
                "/input/article.pdf",
                "/output/article.txt",
            ]
        )
        memory_limit = min(self.settings.max_memory_bytes, 1024**3)
        file_limit = min(self.settings.max_file_bytes, PDF_TEXT_MAX_BYTES)
        return [
            str(self.settings.prlimit),
            f"--cpu={PDF_PARSE_TIMEOUT_SECONDS + 1}",
            f"--as={memory_limit}",
            f"--fsize={file_limit}",
            "--nofile=128",
            "--nproc=16",
            "--core=0",
            "--",
            *sandbox,
        ]

    def extract_pdf_text(self, request: PdfExtractRequest) -> dict:
        pdf = self.confined_pdf_path(request)
        with tempfile.TemporaryDirectory(prefix="evidence-pdf-parser-") as temporary:
            output_dir = Path(temporary)
            output_dir.chmod(0o700)
            stderr_path = output_dir / "parser.stderr"
            command = self._pdf_bwrap_command(pdf, output_dir)
            with stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    start_new_session=True,
                )
                deadline = time.monotonic() + PDF_PARSE_TIMEOUT_SECONDS
                while process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2)
                    raise ValueError("PDF parser exceeded its wall-time limit")
                process.wait()
            output = output_dir / "article.txt"
            if process.returncode != 0:
                detail = stderr_path.read_bytes()[:8_192].decode(
                    "utf-8", errors="replace"
                )
                raise ValueError(f"isolated PDF parser failed: {detail.strip()[:500]}")
            try:
                info = output.lstat()
            except OSError as exc:
                raise ValueError("isolated PDF parser produced no text") from exc
            if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
                raise ValueError("isolated PDF parser produced an invalid output")
            if info.st_size > PDF_TEXT_MAX_BYTES:
                raise ValueError("isolated PDF text exceeds the output limit")
            text = output.read_text(encoding="utf-8", errors="replace")
            return {
                "text": text,
                "bytes": info.st_size,
                "pdf_sha256": sha256_file(pdf),
            }

    def _ocr_bwrap_command(self, figure: Path, output_dir: Path) -> list[str]:
        tesseract = Path("/usr/bin/tesseract")
        required = [
            self.settings.bwrap,
            self.settings.prlimit,
            tesseract,
            Path("/usr/lib"),
            Path("/usr/share/tesseract-ocr"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise RuntimeError(
                f"OCR sandbox runtime paths are missing: {', '.join(missing)}"
            )
        suffix = figure.suffix.lower()
        sandbox = [
            str(self.settings.bwrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--dir",
            "/usr",
            "--dir",
            "/usr/bin",
            "--ro-bind",
            str(tesseract),
            "/usr/bin/tesseract",
            "--dir",
            "/usr/lib",
            "--ro-bind",
            "/usr/lib",
            "/usr/lib",
            "--symlink",
            "usr/lib",
            "/lib",
        ]
        if Path("/usr/lib64").exists():
            sandbox.extend(
                [
                    "--ro-bind",
                    "/usr/lib64",
                    "/usr/lib64",
                    "--symlink",
                    "usr/lib64",
                    "/lib64",
                ]
            )
        sandbox.extend(
            [
                "--dir",
                "/usr/share",
                "--ro-bind",
                "/usr/share/tesseract-ocr",
                "/usr/share/tesseract-ocr",
                "--dir",
                "/dev",
                "--dev-bind",
                "/dev/null",
                "/dev/null",
                "--dir",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/tmp/home",
                "--dir",
                "/input",
                "--ro-bind",
                str(figure),
                f"/input/figure{suffix}",
                "--dir",
                "/output",
                "--bind",
                str(output_dir),
                "/output",
                "--clearenv",
                "--setenv",
                "HOME",
                "/tmp/home",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "LANG",
                "C.UTF-8",
                "--setenv",
                "PATH",
                "/usr/bin",
                "--setenv",
                "OMP_THREAD_LIMIT",
                "1",
                "--chdir",
                "/output",
                "/usr/bin/tesseract",
                f"/input/figure{suffix}",
                "/output/figure",
                "-l",
                "eng",
                "--psm",
                "3",
                "tsv",
            ]
        )
        memory_limit = min(self.settings.max_memory_bytes, 1024**3)
        return [
            str(self.settings.prlimit),
            f"--cpu={OCR_PARSE_TIMEOUT_SECONDS + 1}",
            f"--as={memory_limit}",
            f"--fsize={OCR_TEXT_MAX_BYTES}",
            "--nofile=128",
            "--nproc=16",
            "--core=0",
            "--",
            *sandbox,
        ]

    def extract_figure_ocr(self, request: FigureOcrRequest) -> dict:
        figure = self.confined_figure_path(request)
        with tempfile.TemporaryDirectory(prefix="evidence-figure-ocr-") as temporary:
            output_dir = Path(temporary)
            output_dir.chmod(0o700)
            completed = subprocess.run(
                self._ocr_bwrap_command(figure, output_dir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=False,
                timeout=OCR_PARSE_TIMEOUT_SECONDS + 2,
            )
            output = output_dir / "figure.tsv"
            if completed.returncode != 0:
                detail = completed.stderr[:8_192].decode("utf-8", errors="replace")
                raise ValueError(f"isolated OCR failed: {detail.strip()[:500]}")
            try:
                info = output.lstat()
            except OSError as exc:
                raise ValueError("isolated OCR produced no output") from exc
            if (
                not stat.S_ISREG(info.st_mode)
                or stat.S_ISLNK(info.st_mode)
                or info.st_size > OCR_TEXT_MAX_BYTES
            ):
                raise ValueError("isolated OCR produced invalid output")
            return {
                "ocr": _parse_tesseract_tsv(output.read_bytes()),
                "figure_sha256": sha256_file(figure),
            }

    def execute(self, request: ExecuteRequest) -> dict:
        workspace, root = self.confined_paths(request)
        cancellation = threading.Event()
        with self.lock:
            if request.request_id in self.cancellation_events:
                raise ValueError("duplicate worker request ID")
            self.cancellation_events[request.request_id] = cancellation
        # The first attempt has no prior computation directory yet, but bubblewrap
        # still needs an existing read-only bind source for /history.
        root.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        key = str(root)
        requested_call_budget = (
            request.max_calls_per_attempt
            if request.max_calls_per_attempt is not None
            else self.settings.max_calls_per_attempt
        )
        effective_call_budget = min(
            requested_call_budget, self.settings.max_calls_per_attempt
        )
        try:
            with self.lock:
                executor = self.executors.get(key)
                if executor is None:
                    staging = Path(tempfile.mkdtemp(prefix="evidence-worker-"))
                    staged_workspace = staging / "workspace"
                    staged_root = staging / "computations"
                    staged_workspace.mkdir(mode=0o700)
                    self._stage_workspace_inputs(workspace, staged_workspace)
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
                        replace(
                            self.settings,
                            max_calls_per_attempt=effective_call_budget,
                        ),
                        environment_dir=environment_dir,
                        history_dir=root.parent,
                    )
                    self.executors[key] = executor
                    self.staging_roots[key] = staged_root
                    self.executor_locks[key] = threading.Lock()
                elif executor.settings.max_calls_per_attempt != effective_call_budget:
                    raise ValueError(
                        "analysis call budget cannot change within one attempt"
                    )
                executor_lock = self.executor_locks[key]
            with executor_lock:
                executor.cancel_event = cancellation
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
        finally:
            with self.lock:
                self.cancellation_events.pop(request.request_id, None)

    @staticmethod
    def _stage_workspace_inputs(workspace: Path, staged_workspace: Path) -> None:
        """Copy uploaded root files while excluding controller-managed directories."""

        for source in workspace.iterdir():
            if source.is_symlink():
                raise ValueError("workspace inputs must not contain symlinks")
            if source.is_dir():
                continue
            if not source.is_file():
                raise ValueError("workspace inputs must be regular files")
            shutil.copy2(source, staged_workspace / source.name)

    def cancel(self, request_id: str) -> bool:
        with self.lock:
            event = self.cancellation_events.get(request_id)
            if event is None:
                return False
            event.set()
            return True

    def release(self, request: ReleaseRequest) -> bool:
        _, root = self.confined_paths(request)
        key = str(root)
        with self.lock:
            executor_lock = self.executor_locks.get(key)
        if executor_lock is None:
            return False
        with executor_lock:
            with self.lock:
                if self.executor_locks.get(key) is not executor_lock:
                    return False
                self.executors.pop(key, None)
                staged_root = self.staging_roots.pop(key, None)
                self.executor_locks.pop(key, None)
            if staged_root is not None:
                shutil.rmtree(staged_root.parent, ignore_errors=True)
        return True

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


def _worker_authorized(actual: str, token: str) -> bool:
    expected = f"Bearer {token}"
    return secrets.compare_digest(actual.encode("utf-8"), expected.encode("utf-8"))


def _handler(state: WorkerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvidenceBenchSandbox/0.4"

        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self._json(404, {"detail": "not found"})
                return
            self._json(200, {"status": "ok"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {
                "/execute",
                "/cancel",
                "/release",
                "/extract-pdf-text",
                "/extract-figure-ocr",
            }:
                self._json(404, {"detail": "not found"})
                return
            actual = self.headers.get("Authorization", "")
            if not _worker_authorized(actual, state.token):
                self._json(401, {"detail": "valid worker token required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"detail": "invalid content length"})
                return
            max_size = (
                state.settings.max_code_bytes + 32 * 1024
                if self.path == "/execute"
                else 32 * 1024
                if self.path in {"/release", "/extract-pdf-text", "/extract-figure-ocr"}
                else 4096
            )
            if size < 1 or size > max_size:
                self._json(413, {"detail": "request body exceeds worker limit"})
                return
            try:
                body = self.rfile.read(size)
                if self.path == "/cancel":
                    cancel = CancelRequest.model_validate_json(body)
                    accepted = state.cancel(cancel.request_id)
                    self._json(
                        202 if accepted else 404,
                        {"accepted": accepted, "request_id": cancel.request_id},
                    )
                    return
                if self.path == "/release":
                    release = ReleaseRequest.model_validate_json(body)
                    released = state.release(release)
                    self._json(200, {"released": released})
                    return
                if self.path == "/extract-pdf-text":
                    extraction = PdfExtractRequest.model_validate_json(body)
                    payload = state.extract_pdf_text(extraction)
                    self._json(200, payload)
                    return
                if self.path == "/extract-figure-ocr":
                    extraction = FigureOcrRequest.model_validate_json(body)
                    payload = state.extract_figure_ocr(extraction)
                    self._json(200, payload)
                    return
                request = ExecuteRequest.model_validate_json(body)
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
        raise RuntimeError(
            "SCIENTIFIC_AGENT_SANDBOX_WORKER_TOKEN must be at least 24 characters"
        )
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
