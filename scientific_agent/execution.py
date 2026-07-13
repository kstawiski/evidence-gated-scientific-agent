"""Typed, resource-bounded Python and R execution through bubblewrap."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .config import SandboxSettings
from .provenance import sha256_bytes, sha256_file, utc_now, write_json
from .schemas import ArtifactRef, ComputationEvidence, ComputationRecord


Language = Literal["python", "r"]
RETURN_TEXT_BYTES = 32 * 1024
PREVIEW_TEXT_BYTES = 8 * 1024
PREVIEW_SUFFIXES = {".csv", ".json", ".md", ".tsv", ".txt"}


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
    _records: list[ComputationRecord] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.root = self.root.resolve()
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
        missing = [str(path) for path in self._required_paths(language) if not path.exists()]
        if missing:
            raise RuntimeError(f"sandbox runtime paths are missing: {', '.join(missing)}")

    def _bwrap_command(
        self,
        language: Language,
        script: Path,
        output_dir: Path,
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
            bwrap.extend(
                [
                    "--dir",
                    "/opt",
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
                        "sys.path.insert(0,'/opt/python-packages');"
                        f"runpy.run_path('/analysis/{script.name}',run_name='__main__')"
                    ),
                ]
            )
        else:
            bwrap.extend(
                [
                    "--ro-bind",
                    "/etc/R",
                    "/etc/R",
                    "--dir",
                    "/opt",
                    "--ro-bind",
                    str(self.settings.r_library),
                    "/opt/R-library",
                    "--setenv",
                    "PATH",
                    "/usr/bin",
                    "--setenv",
                    "R_LIBS_USER",
                    "/opt/R-library",
                    "/usr/bin/bash",
                    "-c",
                    (
                        f"ulimit -u {self.settings.max_processes}; "
                        f"exec /usr/bin/Rscript --vanilla /analysis/{script.name}"
                    ),
                ]
            )
        return bwrap

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

        status: Literal["succeeded", "failed", "timed_out", "policy_denied"]
        exit_code: int | None = None
        violations: list[str] = []
        if not code.strip():
            violations.append("code must not be empty")
        if len(code_bytes) > self.settings.max_code_bytes:
            violations.append(
                f"code exceeds {self.settings.max_code_bytes} byte limit"
            )
        if len(self._records) >= self.settings.max_calls_per_attempt:
            violations.append("analysis call budget exhausted")
        try:
            self._validate_runtime(language)
        except Exception as exc:
            violations.append(str(exc))

        timeout_seconds = max(1, min(timeout_seconds, self.settings.max_wall_seconds))
        timed_out = False
        output_artifacts: list[ArtifactRef] = []
        if violations:
            status = "policy_denied"
            stdout_path.write_text("", encoding="utf-8")
            stderr_path.write_text("\n".join(violations) + "\n", encoding="utf-8")
        else:
            command = self._limited_command(
                self._bwrap_command(language, script, output_dir), timeout_seconds
            )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    completed = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout_seconds + 2,
                        check=False,
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
            output_artifacts, output_violations = self._inspect_outputs(output_dir)
            violations.extend(output_violations)
            if timed_out:
                status = "timed_out"
            elif exit_code == 0 and not violations:
                status = "succeeded"
            else:
                status = "policy_denied" if violations else "failed"

        stdout_path.chmod(0o600)
        stderr_path.chmod(0o600)
        artifacts = [
            _artifact(script, f"{language} analysis source"),
            _artifact(stdout_path, "captured standard output"),
            _artifact(stderr_path, "captured standard error"),
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
            artifacts=artifacts,
        )
        self._records.append(record)
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
            "stdout": _read_bounded(stdout_path),
            "stderr": _read_bounded(stderr_path),
            "artifacts": [item.model_dump(mode="json") for item in artifacts],
            "output_previews": _output_previews(output_artifacts),
            "violations": sorted(set(violations)),
            "workspace_path": "/workspace",
            "output_path": "/output",
        }

    def evidence(self) -> ComputationEvidence:
        successful = [record for record in self._records if record.status == "succeeded"]
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


def build_analysis_tools(executor: AnalysisExecutor):
    def run_python_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run Python scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Write all tables,
        figures, and machine-readable results below /output. NumPy, pandas,
        SciPy, statsmodels, scikit-learn, and matplotlib are available.

        Args:
            code: Complete Python script to execute.
            timeout_seconds: Wall-time request, capped by controller policy.
        """

        return executor.execute("python", code, timeout_seconds)

    def run_r_analysis(code: str, timeout_seconds: int = 120) -> dict:
        """Run R scientific analysis in an offline sandbox.

        The assigned project is read-only at /workspace. Write all tables,
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
    missing = [name for name, path in paths.items() if not path.exists()]
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
    with tempfile.TemporaryDirectory(prefix="scientific-agent-preflight-") as root:
        executor = AnalysisExecutor(
            (workspace or Path.cwd()).resolve(),
            Path(root),
            settings,
        )
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
    result["probes"] = {
        "python": python_probe["status"],
        "r": r_probe["status"],
    }
    return result
