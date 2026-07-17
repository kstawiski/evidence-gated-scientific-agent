"""Typed, resource-bounded Python and R execution through bubblewrap."""

from __future__ import annotations

import ast
import os
import re
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

import httpx

from .config import SandboxSettings
from .provenance import sha256_bytes, sha256_file, utc_now, write_json
from .schemas import ArtifactRef, ComputationEvidence, ComputationRecord


Language = Literal["python", "r"]
RETURN_TEXT_BYTES = 32 * 1024
PREVIEW_TEXT_BYTES = 8 * 1024
PREVIEW_SUFFIXES = {".csv", ".json", ".md", ".tsv", ".txt"}
PRIOR_EXECUTION_REFERENCE = re.compile(r"/prior/(?P<execution_id>exec-[0-9]{3})/")


def _python_static_violations(code: str) -> list[str]:
    """Reject a small set of unambiguously invalid scientific API calls."""

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    violations: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "errorbar":
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        if "linewidths" in keywords:
            violations.add(
                "Matplotlib errorbar rejects linewidths=; use linewidth= or elinewidth="
            )
        scalar_x = bool(node.args and isinstance(node.args[0], ast.Constant))
        if not scalar_x:
            continue
        for name in ("xerr", "yerr"):
            value = keywords.get(name)
            if not isinstance(value, (ast.List, ast.Tuple)) or len(value.elts) != 2:
                continue
            if any(isinstance(item, (ast.List, ast.Tuple)) for item in value.elts):
                continue
            violations.add(
                f"Matplotlib singleton asymmetric {name} must have shape (2, 1); "
                f"use [[lower], [upper]] rather than [lower, upper]"
            )
    return sorted(violations)


def _unavailable_prior_reference_violations(
    code: str, successful_execution_ids: set[str]
) -> list[str]:
    referenced = {
        match.group("execution_id")
        for match in PRIOR_EXECUTION_REFERENCE.finditer(code)
    }
    return [
        (
            f"/prior/{execution_id} is not a successful execution in the current "
            "attempt; use /history/attempt-N/exec-ID/output only for the exact "
            "registered prior-attempt artifact"
        )
        for execution_id in sorted(referenced - successful_execution_ids)
    ]


def _read_bounded(path: Path) -> str:
    data = path.read_bytes()[:RETURN_TEXT_BYTES]
    text = data.decode("utf-8", errors="replace")
    if path.stat().st_size > RETURN_TEXT_BYTES:
        text += f"\n...[truncated; full log: {path}]"
    return text


def _artifact(path: Path, description: str) -> ArtifactRef:
    return ArtifactRef(
        path=str(path.resolve()),
        sha256=sha256_file(path),
        description=description,
    )


def _output_previews(artifacts: list[ArtifactRef]) -> dict[str, str]:
    previews: dict[str, str] = {}
    for artifact in artifacts:
        path = Path(artifact.path)
        if path.suffix.lower() not in PREVIEW_SUFFIXES:
            continue
        data = path.read_bytes()[:PREVIEW_TEXT_BYTES]
        preview = data.decode("utf-8", errors="replace")
        if path.stat().st_size > PREVIEW_TEXT_BYTES:
            preview += "\n...[truncated]"
        previews[artifact.path] = preview
    return previews


@dataclass
class AnalysisExecutor:
    workspace: Path
    root: Path
    settings: SandboxSettings
    environment_dir: Path | None = None
    history_dir: Path | None = None
    cancel_event: threading.Event | None = None
    _records: list[ComputationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.root = self.root.resolve()
        if self.environment_dir is not None:
            self.environment_dir = self.environment_dir.resolve()
        self.history_dir = (self.history_dir or self.root.parent).resolve()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)

    def _required_paths(self, language: Language) -> list[Path]:
        common = [self.settings.bwrap, self.settings.prlimit, Path("/usr")]
        if language == "python":
            return [
                *common,
                self.settings.python,
                self.settings.python_prefix,
                self.settings.python_packages,
            ]
        return [*common, self.settings.rscript, self.settings.r_library, Path("/etc/R")]

    def _validate_runtime(self, language: Language) -> None:
        missing = [
            str(path) for path in self._required_paths(language) if not path.exists()
        ]
        if missing:
            raise RuntimeError(
                f"sandbox runtime paths are missing: {', '.join(missing)}"
            )

    def _bwrap_command(
        self,
        language: Language,
        script: Path,
        output_dir: Path,
        workspace_packages: Path | None = None,
    ) -> list[str]:
        bwrap = [
            str(self.settings.bwrap),
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--symlink",
            "usr/bin",
            "/bin",
            "--symlink",
            "usr/lib",
            "/lib",
            "--symlink",
            "usr/lib64",
            "/lib64",
            "--dir",
            "/etc",
            "--ro-bind",
            "/etc/alternatives",
            "/etc/alternatives",
            "--ro-bind",
            "/etc/ld.so.cache",
            "/etc/ld.so.cache",
            "--dir",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/tmp/home",
            "--dir",
            "/workspace",
            "--ro-bind",
            str(self.workspace),
            "/workspace",
            "--dir",
            "/prior",
            "--ro-bind",
            str(self.root),
            "/prior",
            "--ro-bind",
            str(self.history_dir),
            "/history",
            "--dir",
            "/analysis",
            "--ro-bind",
            str(script),
            f"/analysis/{script.name}",
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
            "OMP_NUM_THREADS",
            "1",
            "--setenv",
            "OPENBLAS_NUM_THREADS",
            "1",
            "--setenv",
            "MKL_NUM_THREADS",
            "1",
            "--chdir",
            "/output",
        ]
        if language == "python":
            try:
                python_rel = self.settings.python.resolve().relative_to(
                    self.settings.python_prefix.resolve()
                )
            except ValueError as exc:
                raise RuntimeError(
                    "sandbox Python executable must be inside its configured prefix"
                ) from exc
            sandbox_python = Path("/opt/python-runtime") / python_rel
            python_path_setup = "sys.path.insert(0,'/opt/python-packages');"
            bwrap.extend(["--dir", "/opt"])
            if workspace_packages is not None and workspace_packages.is_dir():
                bwrap.extend(
                    [
                        "--ro-bind",
                        str(workspace_packages),
                        "/opt/workspace-python-packages",
                    ]
                )
                python_path_setup += (
                    "sys.path.insert(0,'/opt/workspace-python-packages');"
                )
            bwrap.extend(
                [
                    "--ro-bind",
                    str(self.settings.python_prefix),
                    "/opt/python-runtime",
                    "--ro-bind",
                    str(self.settings.python_packages),
                    "/opt/python-packages",
                    "--setenv",
                    "PATH",
                    "/opt/python-runtime/bin:/usr/bin",
                    "--setenv",
                    "PYTHONHASHSEED",
                    "0",
                    "--setenv",
                    "MPLCONFIGDIR",
                    "/tmp/matplotlib",
                    str(sandbox_python),
                    "-I",
                    "-c",
                    (
                        "import runpy,sys;"
                        "import resource;"
                        f"resource.setrlimit(resource.RLIMIT_NPROC,({self.settings.max_processes},{self.settings.max_processes}));"
                        f"{python_path_setup}"
                        f"runpy.run_path('/analysis/{script.name}',run_name='__main__')"
                    ),
                ]
            )
        else:
            r_libraries = "/opt/R-library"
            bwrap.extend(["--dir", "/opt"])
            if workspace_packages is not None and workspace_packages.is_dir():
                bwrap.extend(
                    [
                        "--ro-bind",
                        str(workspace_packages),
                        "/opt/workspace-R-library",
                    ]
                )
                r_libraries = "/opt/workspace-R-library:/opt/R-library"
            bwrap.extend(
                [
                    "--ro-bind",
                    "/etc/R",
                    "/etc/R",
                    "--ro-bind",
                    str(self.settings.r_library),
                    "/opt/R-library",
                    "--setenv",
                    "PATH",
                    "/usr/bin",
                    "--setenv",
                    "R_LIBS_USER",
                    r_libraries,
                    "/usr/bin/bash",
                    "-c",
                    (
                        f"ulimit -u {self.settings.max_processes}; "
                        f"exec /usr/bin/Rscript --vanilla /analysis/{script.name}"
                    ),
                ]
            )
        return bwrap

    def _snapshot_environment(
        self,
        language: Language,
        call_dir: Path,
    ) -> tuple[Path | None, dict[str, str], list[ArtifactRef]]:
        if self.environment_dir is None:
            return None, {}, []
        current = self.environment_dir / language
        if not current.exists():
            return None, {}, []
        generation = current.resolve()
        generations = (self.environment_dir / ".generations").resolve()
        if generation.parent != generations:
            raise RuntimeError("workspace package generation escaped its environment")
        packages = generation / "packages"
        lock = generation / "lock.json"
        if not packages.is_dir() or not lock.is_file():
            raise RuntimeError("workspace package generation is incomplete")
        lock_copy = call_dir / f"environment-{language}-lock.json"
        lock_copy.write_bytes(lock.read_bytes())
        lock_copy.chmod(0o600)
        digest = sha256_file(lock_copy)
        return (
            packages,
            {language: digest},
            [_artifact(lock_copy, "workspace package environment lock")],
        )

    def _limited_command(self, command: list[str], timeout_seconds: int) -> list[str]:
        return [
            str(self.settings.prlimit),
            f"--cpu={timeout_seconds + 1}",
            f"--as={self.settings.max_memory_bytes}",
            f"--fsize={self.settings.max_file_bytes}",
            "--nofile=1024",
            "--core=0",
            "--",
            *command,
        ]

    def _inspect_outputs(self, output_dir: Path) -> tuple[list[ArtifactRef], list[str]]:
        artifacts: list[ArtifactRef] = []
        violations: list[str] = []
        total_bytes = 0
        for path in sorted(output_dir.rglob("*")):
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode):
                violations.append(f"non-regular output rejected: {path}")
                continue
            total_bytes += info.st_size
            if info.st_size > self.settings.max_file_bytes:
                violations.append(f"output file exceeds per-file limit: {path}")
                continue
            artifacts.append(_artifact(path, "sandbox-generated analysis artifact"))
        if total_bytes > self.settings.max_output_bytes:
            violations.append(
                f"total output exceeds {self.settings.max_output_bytes} bytes"
            )
            artifacts = []
        return artifacts, violations

    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict:
        """Execute one Python or R script inside the fixed sandbox profile."""

        execution_id = f"exec-{len(self._records) + 1:03d}"
        started_at = utc_now()
        started = time.monotonic()
        code_bytes = code.encode("utf-8")
        extension = "py" if language == "python" else "R"
        call_dir = self.root / execution_id
        output_dir = call_dir / "output"
        call_dir.mkdir(parents=True, mode=0o700, exist_ok=False)
        output_dir.mkdir(mode=0o700)
        script = call_dir / f"analysis.{extension}"
        stdout_path = call_dir / "stdout.txt"
        stderr_path = call_dir / "stderr.txt"
        script.write_bytes(code_bytes)
        script.chmod(0o600)
        environment_locks: dict[str, str] = {}
        environment_artifacts: list[ArtifactRef] = []
        workspace_packages: Path | None = None

        status: Literal[
            "succeeded", "failed", "timed_out", "cancelled", "policy_denied"
        ]
        exit_code: int | None = None
        violations: list[str] = []
        if not code.strip():
            violations.append("code must not be empty")
        if len(code_bytes) > self.settings.max_code_bytes:
            violations.append(f"code exceeds {self.settings.max_code_bytes} byte limit")
        if len(self._records) >= self.settings.max_calls_per_attempt:
            violations.append("analysis call budget exhausted")
        if language == "python":
            violations.extend(_python_static_violations(code))
        violations.extend(
            _unavailable_prior_reference_violations(
                code,
                {
                    record.execution_id
                    for record in self._records
                    if record.status == "succeeded"
                },
            )
        )
        try:
            (
                workspace_packages,
                environment_locks,
                environment_artifacts,
            ) = self._snapshot_environment(language, call_dir)
        except Exception as exc:
            violations.append(str(exc))
        try:
            self._validate_runtime(language)
        except Exception as exc:
            violations.append(str(exc))

        timeout_seconds = max(1, min(timeout_seconds, self.settings.max_wall_seconds))
        timed_out = False
        cancelled = False
        output_artifacts: list[ArtifactRef] = []
        if violations:
            status = "policy_denied"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("\n".join(violations) + "\n", encoding="utf-8")
        else:
            command = self._limited_command(
                self._bwrap_command(
                    language,
                    script,
                    output_dir,
                    workspace_packages=workspace_packages,
                ),
                timeout_seconds,
            )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
                deadline = time.monotonic() + timeout_seconds + 2
                while process.poll() is None:
                    if self.cancel_event is not None and self.cancel_event.is_set():
                        cancelled = True
                        break
                    if time.monotonic() >= deadline:
                        timed_out = True
                        break
                    time.sleep(0.05)
                if cancelled or timed_out:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=2)
                else:
                    process.wait()
                exit_code = process.returncode
            output_artifacts, output_violations = self._inspect_outputs(output_dir)
            violations.extend(output_violations)
            if cancelled:
                status = "cancelled"
            elif timed_out:
                status = "timed_out"
            elif exit_code == 0 and not violations:
                status = "succeeded"
            else:
                status = "policy_denied" if violations else "failed"

        if status != "succeeded" and output_artifacts:
            output_root = output_dir.resolve()
            rejected = [
                (Path(artifact.path).relative_to(output_root), artifact)
                for artifact in output_artifacts
            ]
            rejected_dir = call_dir / "rejected_output"
            output_dir.rename(rejected_dir)
            output_artifacts = [
                ArtifactRef(
                    path=str((rejected_dir / relative).resolve()),
                    sha256=artifact.sha256,
                    description="rejected sandbox output (not evidence)",
                )
                for relative, artifact in rejected
            ]

        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        artifacts = [
            _artifact(script, f"{language} analysis source"),
            _artifact(stdout_path, "captured standard output"),
            _artifact(stderr_path, "captured standard error"),
            *environment_artifacts,
            *output_artifacts,
        ]
        record = ComputationRecord(
            execution_id=execution_id,
            language=language,
            code_sha256=sha256_bytes(code_bytes),
            started_at=started_at,
            duration_seconds=round(time.monotonic() - started, 3),
            exit_code=exit_code,
            status=status,
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            environment_locks=environment_locks,
            artifacts=artifacts,
        )
        self._records.append(record)
        calls_used = len(self._records)
        calls_remaining = max(0, self.settings.max_calls_per_attempt - calls_used)
        write_json(
            call_dir / "execution.json",
            {
                **record.model_dump(mode="json"),
                "violations": sorted(set(violations)),
                "limits": {
                    "wall_seconds": timeout_seconds,
                    "memory_bytes": self.settings.max_memory_bytes,
                    "processes": self.settings.max_processes,
                    "file_bytes": self.settings.max_file_bytes,
                    "total_output_bytes": self.settings.max_output_bytes,
                },
            },
        )
        return {
            "execution_id": execution_id,
            "language": language,
            "status": status,
            "exit_code": exit_code,
            "duration_seconds": record.duration_seconds,
            "environment_locks": environment_locks,
            "stdout": _read_bounded(stdout_path),
            "stderr": _read_bounded(stderr_path),
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
            "output_previews": (
                _output_previews(output_artifacts) if status == "succeeded" else {}
            ),
            "violations": sorted(set(violations)),
            "calls_used": calls_used,
            "calls_remaining": calls_remaining,
            "stop_required": "analysis call budget exhausted" in violations,
            "workspace_path": "/workspace",
            "prior_outputs_path": "/prior",
            "attempt_history_path": "/history",
            "output_path": "/output",
        }

    def evidence(self) -> ComputationEvidence:
        successful = [
            record for record in self._records if record.status == "succeeded"
        ]
        artifacts = [
            artifact
            for record in successful
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
        ]
        return ComputationEvidence(
            successful_calls=len(successful),
            records=list(self._records),
            artifacts=artifacts,
        )

    def close(self) -> None:
        """Release executor-scoped resources (none for in-process execution)."""


def build_analysis_tools(executor: AnalysisExecutor):
    def run_python_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run Python scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Earlier calls are
        read-only at /prior/<execution-id>/output, and prior repair attempts at
        /history/attempt-N/<execution-id>/output. Write all tables,
        figures, and machine-readable results below /output. NumPy, pandas,
        SciPy, statsmodels, scikit-learn, and Matplotlib are available. This
        runtime uses Matplotlib 3.10+; Axes.boxplot uses tick_labels rather than
        the removed labels keyword.

        Args:
            code: Complete Python script to execute.
            timeout_seconds: Wall-time request, capped by controller policy.
        """

        return executor.execute("python", code, timeout_seconds)

    def run_r_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run R scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Earlier calls are
        read-only at /prior/<execution-id>/output, and prior repair attempts at
        /history/attempt-N/<execution-id>/output. Write all tables,
        figures, and machine-readable results below /output. ggplot2, dplyr,
        survival, data.table, and the installed R libraries are available.

        Args:
            code: Complete R script to execute.
            timeout_seconds: Wall-time request, capped by controller policy.
        """

        return executor.execute("r", code, timeout_seconds)

    return [run_python_analysis, run_r_analysis]


def sandbox_preflight(settings: SandboxSettings, workspace: Path | None = None) -> dict:
    """Check runtime paths and execute fixed Python/R isolation probes."""

    paths = {
        "bwrap": settings.bwrap,
        "prlimit": settings.prlimit,
        "python": settings.python,
        "python_prefix": settings.python_prefix,
        "python_packages": settings.python_packages,
        "rscript": settings.rscript,
        "r_library": settings.r_library,
    }
    # A remote worker owns its runtime paths; the execution probe below is the
    # authoritative check. Local path existence matters only for in-process mode.
    missing = (
        []
        if settings.worker_url
        else [name for name, path in paths.items() if not path.exists()]
    )
    result = {
        "paths": {name: str(path) for name, path in paths.items()},
        "missing_required": missing,
        "network": "unshared",
        "workspace": "read-only",
        "output": "per-call writable directory",
    }
    if missing:
        result["probes"] = {}
        return result
    managed_base: Path | None = None
    temporary_root = None
    if settings.worker_url:
        data = Path(os.environ.get("SCIENTIFIC_AGENT_DATA_DIR", "/data")).resolve()
        managed_base = data / "workspaces" / str(uuid.uuid4())
        probe_workspace = managed_base / "files"
        probe_root = managed_base / "runs" / "preflight" / "computations"
        probe_workspace.mkdir(parents=True, mode=0o700)
        probe_root.mkdir(parents=True, mode=0o700)
        executor: AnalysisRunner = RemoteAnalysisExecutor(
            probe_workspace, probe_root, settings
        )
    else:
        temporary_root = tempfile.TemporaryDirectory(
            prefix="scientific-agent-preflight-"
        )
        executor = AnalysisExecutor(
            (workspace or Path.cwd()).resolve(),
            Path(temporary_root.name),
            settings,
        )
    try:
        python_probe = executor.execute(
            "python",
            (
                "import matplotlib,numpy,pandas,scipy,sklearn,statsmodels;"
                "open('/output/python-ok.txt', 'w').write('ok')"
            ),
            timeout_seconds=15,
        )
        r_probe = executor.execute(
            "r",
            (
                "stopifnot(all(vapply(c('ggplot2','dplyr','survival','data.table'), "
                "requireNamespace, logical(1), quietly=TRUE)));"
                "writeLines('ok', '/output/r-ok.txt')"
            ),
            timeout_seconds=15,
        )
    finally:
        executor.close()
        if temporary_root is not None:
            temporary_root.cleanup()
        if managed_base is not None:
            shutil.rmtree(managed_base, ignore_errors=True)
    result["probes"] = {
        "python": python_probe["status"],
        "r": r_probe["status"],
    }
    return result


class AnalysisRunner(Protocol):
    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict: ...

    def evidence(self) -> ComputationEvidence: ...

    def close(self) -> None: ...


@dataclass
class RemoteAnalysisExecutor:
    """Typed client for the isolated, non-published container worker."""

    workspace: Path
    root: Path
    settings: SandboxSettings
    cancel_event: threading.Event | None = None
    _evidence: ComputationEvidence = field(default_factory=ComputationEvidence)

    def __post_init__(self) -> None:
        if not self.settings.worker_url or not self.settings.worker_token:
            raise RuntimeError("sandbox worker URL and token are both required")

    def execute(
        self,
        language: Language,
        code: str,
        timeout_seconds: int = 120,
    ) -> dict:
        timeout = max(1, min(timeout_seconds, self.settings.max_wall_seconds))
        request_id = str(uuid.uuid4())
        finished = threading.Event()

        def cancel_remote() -> None:
            if self.cancel_event is None:
                return
            while not finished.is_set():
                if not self.cancel_event.wait(timeout=0.1):
                    continue
                try:
                    response = httpx.post(
                        f"{self.settings.worker_url}/cancel",
                        headers={
                            "Authorization": f"Bearer {self.settings.worker_token}"
                        },
                        json={"request_id": request_id},
                        timeout=5,
                    )
                    if response.status_code in {200, 202}:
                        return
                except httpx.HTTPError:
                    pass
                finished.wait(timeout=0.2)

        watcher = threading.Thread(
            target=cancel_remote,
            name=f"sandbox-cancel-{request_id[:8]}",
            daemon=True,
        )
        watcher.start()
        try:
            response = httpx.post(
                f"{self.settings.worker_url}/execute",
                headers={"Authorization": f"Bearer {self.settings.worker_token}"},
                json={
                    "request_id": request_id,
                    "workspace": str(self.workspace),
                    "computation_root": str(self.root),
                    "language": language,
                    "code": code,
                    "timeout_seconds": timeout,
                    "max_calls_per_attempt": self.settings.max_calls_per_attempt,
                },
                timeout=timeout + 15,
            )
        finally:
            finished.set()
        if response.is_error:
            try:
                detail = response.json().get("detail", "worker request failed")
            except ValueError:
                detail = "worker returned a non-JSON error"
            raise RuntimeError(
                f"sandbox worker returned HTTP {response.status_code}: {detail}"
            )
        payload = response.json()
        self._evidence = ComputationEvidence.model_validate(payload["evidence"])
        return payload["result"]

    def evidence(self) -> ComputationEvidence:
        return self._evidence

    def close(self) -> None:
        response = httpx.post(
            f"{self.settings.worker_url}/release",
            headers={"Authorization": f"Bearer {self.settings.worker_token}"},
            json={
                "workspace": str(self.workspace),
                "computation_root": str(self.root),
            },
            timeout=10,
        )
        if response.is_error:
            try:
                detail = response.json().get("detail", "worker release failed")
            except ValueError:
                detail = "worker returned a non-JSON release error"
            raise RuntimeError(
                f"sandbox worker release returned HTTP {response.status_code}: {detail}"
            )


def create_analysis_executor(
    workspace: Path,
    root: Path,
    settings: SandboxSettings,
    *,
    cancel_event: threading.Event | None = None,
) -> AnalysisRunner:
    if settings.worker_url:
        return RemoteAnalysisExecutor(workspace, root, settings, cancel_event)
    return AnalysisExecutor(workspace, root, settings, cancel_event=cancel_event)
