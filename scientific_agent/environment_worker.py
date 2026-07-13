"""Network-enabled package installer isolated from research data and secrets."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import secrets
import shutil
import stat
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import BaseModel, Field, ValidationError, model_validator

from .environment import validate_python_packages, validate_r_packages
from .provenance import utc_now


RETURN_BYTES = 24 * 1024
PYPI_INDEX = "https://pypi.org/simple"
CRAN_REPOSITORY = "https://cloud.r-project.org"
INSTALL_UID = 10001
INSTALL_GID = 10001


class InstallRequest(BaseModel):
    workspace_id: str
    language: Literal["python", "r"]
    repository: Literal["pypi", "cran", "bioconductor"]
    packages: list[str] = Field(min_length=1, max_length=24)
    timeout_seconds: int = Field(ge=30, le=3600)

    @model_validator(mode="after")
    def repository_matches_language(self) -> "InstallRequest":
        if self.language == "python" and self.repository != "pypi":
            raise ValueError("Python packages must use PyPI")
        if self.language == "r" and self.repository == "pypi":
            raise ValueError("R packages must use CRAN or Bioconductor")
        return self


def _bounded(path: Path) -> str:
    data = path.read_bytes()[:RETURN_BYTES]
    value = data.decode("utf-8", errors="replace")
    if path.stat().st_size > RETURN_BYTES:
        value += "\n...[truncated]"
    return value


@dataclass
class EnvironmentWorkerState:
    environments_dir: Path
    token: str
    max_packages: int = 24
    max_environment_bytes: int = 20 * 1024**3
    locks: dict[str, threading.Lock] = field(default_factory=dict)
    state_lock: threading.Lock = field(default_factory=threading.Lock)

    def workspace_root(self, workspace_id: str) -> Path:
        try:
            uuid.UUID(workspace_id)
        except ValueError as exc:
            raise ValueError("invalid workspace ID") from exc
        root = (self.environments_dir / workspace_id).resolve()
        if root.parent != self.environments_dir.resolve():
            raise ValueError("invalid workspace environment path")
        root.mkdir(parents=True, exist_ok=True)
        return root

    def install(self, request: InstallRequest) -> dict:
        packages = (
            validate_python_packages(request.packages, self.max_packages)
            if request.language == "python"
            else validate_r_packages(request.packages, self.max_packages)
        )
        root = self.workspace_root(request.workspace_id)
        key = f"{request.workspace_id}:{request.language}"
        with self.state_lock:
            lock = self.locks.setdefault(key, threading.Lock())
        with lock:
            return self._install_locked(root, request, packages)

    def _install_locked(
        self, root: Path, request: InstallRequest, packages: list[str]
    ) -> dict:
        destination = root / request.language
        generations = root / ".generations"
        generations.mkdir(mode=0o700, exist_ok=True)
        staging = generations / f"{request.language}-{uuid.uuid4().hex}"
        package_dir = staging / "packages"
        previous_lock: dict = {}
        if destination.exists() or destination.is_symlink():
            previous = destination.resolve()
            if previous.parent != generations.resolve():
                raise RuntimeError("workspace package environment target is invalid")
            if not (previous / "packages").is_dir():
                raise RuntimeError("workspace package generation is incomplete")
            shutil.copytree(previous / "packages", package_dir, symlinks=True)
            try:
                previous_lock = json.loads(
                    (previous / "lock.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                previous_lock = {}
        else:
            package_dir.mkdir(parents=True, mode=0o700)
        with tempfile.TemporaryDirectory(prefix="evidence-package-") as temporary:
            temp = Path(temporary)
            if os.geteuid() == 0:
                self._chown_tree(staging, INSTALL_UID, INSTALL_GID)
                os.chown(temp, INSTALL_UID, INSTALL_GID)
            stdout = temp / "stdout.txt"
            stderr = temp / "stderr.txt"
            command, environment = self._command(
                request, packages, package_dir, temp
            )
            timed_out = False
            with stdout.open("wb") as out, stderr.open("wb") as err:
                try:
                    completed = subprocess.run(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=out,
                        stderr=err,
                        env=environment,
                        timeout=request.timeout_seconds,
                        check=False,
                    )
                    exit_code = completed.returncode
                except subprocess.TimeoutExpired:
                    timed_out = True
                    exit_code = None
            if exit_code != 0 or timed_out:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": "timed_out" if timed_out else "failed",
                    "exit_code": exit_code,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr),
                    "installed": [],
                }

            inventory = self._inventory(request.language, package_dir)
            unsafe_entries = self._unsafe_entries(package_dir)
            if unsafe_entries:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr)
                    + "\nUnsafe package filesystem entry: "
                    + ", ".join(unsafe_entries[:8]),
                    "installed": inventory,
                }
            package_tree_sha256, package_tree_bytes = self._tree_identity(package_dir)
            if package_tree_bytes > self.max_environment_bytes:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr)
                    + f"\nEnvironment exceeds {self.max_environment_bytes} bytes",
                    "installed": inventory,
                }
            missing = self._missing_requested(request.language, packages, inventory)
            if missing:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr)
                    + "\nRequested package(s) absent after install: "
                    + ", ".join(missing),
                    "installed": inventory,
                }

            requested = list(packages)
            requested = list(
                dict.fromkeys([*previous_lock.get("requested", []), *packages])
            )
            lock_path = staging / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "created_at": utc_now(),
                        "language": request.language,
                        "repository": request.repository,
                        "repository_url": (
                            PYPI_INDEX
                            if request.repository == "pypi"
                            else CRAN_REPOSITORY
                            if request.repository == "cran"
                            else "https://bioconductor.org"
                        ),
                        "requested": requested,
                        "installed": inventory,
                        "package_tree_sha256": package_tree_sha256,
                        "package_tree_bytes": package_tree_bytes,
                        "worker_image": os.environ.get(
                            "SCIENTIFIC_AGENT_WORKER_IMAGE_ID", "unknown"
                        ),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            link = root / f".{request.language}-link-{uuid.uuid4().hex}"
            link.symlink_to(Path(".generations") / staging.name)
            os.replace(link, destination)
            return {
                "status": "succeeded",
                "exit_code": 0,
                "stdout": _bounded(stdout),
                "stderr": _bounded(stderr),
                "installed": inventory,
                "lock_file": str(lock_path),
                "generation": staging.name,
                "package_tree_sha256": package_tree_sha256,
                "package_tree_bytes": package_tree_bytes,
            }

    @staticmethod
    def _chown_tree(path: Path, uid: int, gid: int) -> None:
        for item in [path, *path.rglob("*")]:
            if not item.is_symlink():
                try:
                    os.chown(item, uid, gid)
                except PermissionError:
                    # Root-squashed NFS may map the controller and installer UID
                    # to the same NAS identity while refusing explicit chown.
                    # The subsequent unprivileged install remains the authority
                    # on whether the directory is actually writable.
                    continue

    @staticmethod
    def _missing_requested(
        language: str,
        packages: list[str],
        inventory: list[dict[str, str]],
    ) -> list[str]:
        if language == "python":
            installed = {
                canonicalize_name(item["name"]): item["version"]
                for item in inventory
                if item.get("name")
            }
            missing = []
            for package in packages:
                requirement = Requirement(package)
                if requirement.marker is not None and not requirement.marker.evaluate():
                    continue
                version = installed.get(canonicalize_name(requirement.name))
                if version is None or (
                    requirement.specifier
                    and not requirement.specifier.contains(version, prereleases=True)
                ):
                    missing.append(package)
            return missing
        installed = {item["name"] for item in inventory if item.get("name")}
        return [package for package in packages if package not in installed]

    @staticmethod
    def _unsafe_entries(package_dir: Path) -> list[str]:
        unsafe = []
        root = package_dir.resolve()
        for path in package_dir.rglob("*"):
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = (path.parent / os.readlink(path)).resolve(strict=False)
                if target != root and root not in target.parents:
                    unsafe.append(str(path.relative_to(package_dir)))
                continue
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                unsafe.append(str(path.relative_to(package_dir)))
        return unsafe

    @staticmethod
    def _tree_identity(package_dir: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        total = 0
        for path in sorted(package_dir.rglob("*")):
            relative = path.relative_to(package_dir).as_posix().encode("utf-8")
            if path.is_symlink():
                digest.update(b"L\0" + relative + b"\0" + os.readlink(path).encode())
            elif path.is_file():
                size = path.stat().st_size
                total += size
                digest.update(b"F\0" + relative + b"\0" + str(size).encode() + b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
            elif path.is_dir():
                digest.update(b"D\0" + relative + b"\0")
        return digest.hexdigest(), total

    def _command(
        self,
        request: InstallRequest,
        packages: list[str],
        package_dir: Path,
        temporary: Path,
    ) -> tuple[list[str], dict[str, str]]:
        environment = {
            "HOME": "/build/home",
            "TMPDIR": "/build",
            "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
        }
        if request.language == "python":
            environment.update(
                {
                    "UV_CACHE_DIR": "/build/uv-cache",
                    "UV_NO_CONFIG": "1",
                    "UV_DEFAULT_INDEX": PYPI_INDEX,
                }
            )
            inner = [
                "/usr/local/bin/uv",
                "pip",
                "install",
                "--target",
                "/target",
                "--python",
                "/usr/local/bin/python3",
                "--upgrade",
                "--strict",
                "--no-sources",
                "--no-progress",
                *packages,
            ]
            return self._sandbox_command(inner, environment, package_dir, temporary)

        quoted = ",".join(json.dumps(package) for package in packages)
        repository = request.repository
        script = (
            "lib <- '/target'; dir.create(lib, recursive=TRUE, showWarnings=FALSE); "
            f".libPaths(c(lib, .libPaths())); options(repos=c(CRAN={json.dumps(CRAN_REPOSITORY)})); "
            f"pkgs <- c({quoted}); "
        )
        if repository == "cran":
            script += "install.packages(pkgs, lib=lib, Ncpus=2);"
        else:
            script += (
                "if (!requireNamespace('BiocManager', quietly=TRUE)) "
                "install.packages('BiocManager', lib=lib); "
                "BiocManager::install(pkgs, lib=lib, ask=FALSE, update=FALSE, Ncpus=2);"
            )
        inner = ["/usr/bin/Rscript", "--vanilla", "-e", script]
        return self._sandbox_command(inner, environment, package_dir, temporary)

    @staticmethod
    def _sandbox_command(
        inner: list[str],
        environment: dict[str, str],
        package_dir: Path,
        temporary: Path,
    ) -> tuple[list[str], dict[str, str]]:
        command = [
            "/usr/bin/bwrap",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
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
            "--ro-bind",
            "/etc",
            "/etc",
            "--ro-bind",
            "/opt/venv",
            "/opt/venv",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--bind",
            str(package_dir),
            "/target",
            "--bind",
            str(temporary),
            "/build",
            "--clearenv",
        ]
        for name, value in environment.items():
            command.extend(["--setenv", name, value])
        command.extend(
            [
                "--chdir",
                "/build",
                "/usr/bin/setpriv",
                f"--reuid={INSTALL_UID}",
                f"--regid={INSTALL_GID}",
                "--clear-groups",
                "--no-new-privs",
                *inner,
            ]
        )
        return command, {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"}

    @staticmethod
    def _inventory(language: str, path: Path) -> list[dict[str, str]]:
        if language == "python":
            values = {
                distribution.metadata.get("Name", ""): distribution.version
                for distribution in importlib.metadata.distributions(path=[str(path)])
                if distribution.metadata.get("Name")
            }
            return [
                {"name": name, "version": version}
                for name, version in sorted(values.items(), key=lambda item: item[0].lower())
            ]
        expression = (
            f"x <- installed.packages(lib.loc={json.dumps(str(path))}); "
            "if (nrow(x)) write.table(x[,c('Package','Version'),drop=FALSE], "
            "row.names=FALSE, col.names=FALSE, quote=FALSE, sep='\\t')"
        )
        completed = subprocess.run(
            ["/usr/bin/Rscript", "--vanilla", "-e", expression],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return [
            {"name": name, "version": version}
            for line in completed.stdout.splitlines()
            if "\t" in line
            for name, version in [line.split("\t", 1)]
        ]


def _handler(state: EnvironmentWorkerState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "EvidenceBenchPackages/0.3"

        def do_GET(self) -> None:  # noqa: N802
            self._json(200, {"status": "ok"}) if self.path == "/healthz" else self._json(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/install":
                self._json(404, {"detail": "not found"})
                return
            expected = f"Bearer {state.token}"
            if not secrets.compare_digest(self.headers.get("Authorization", ""), expected):
                self._json(401, {"detail": "valid package worker token required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"detail": "invalid content length"})
                return
            if size < 1 or size > 32 * 1024:
                self._json(413, {"detail": "request body exceeds package worker limit"})
                return
            try:
                request = InstallRequest.model_validate_json(self.rfile.read(size))
                payload = state.install(request)
            except (ValidationError, ValueError) as exc:
                self._json(400, {"detail": str(exc)})
                return
            except Exception as exc:
                self._json(500, {"detail": f"{type(exc).__name__}: {exc}"})
                return
            self._json(200, payload)

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            del format, args

    return Handler


def main() -> None:
    if os.geteuid() != 0:
        raise RuntimeError(
            "package worker must start as container root so bubblewrap can "
            "confine installer hooks and drop their UID"
        )
    token = os.environ.get("SCIENTIFIC_AGENT_PACKAGE_WORKER_TOKEN", "")
    if len(token) < 24:
        raise RuntimeError("SCIENTIFIC_AGENT_PACKAGE_WORKER_TOKEN must be at least 24 characters")
    environments = Path(os.environ.get("SCIENTIFIC_AGENT_ENVIRONMENTS_DIR", "/environments")).resolve()
    environments.mkdir(parents=True, exist_ok=True)
    state = EnvironmentWorkerState(
        environments,
        token,
        max_packages=int(os.environ.get("SCIENTIFIC_AGENT_MAX_PACKAGES_PER_CALL", "24")),
        max_environment_bytes=int(
            os.environ.get(
                "SCIENTIFIC_AGENT_MAX_ENVIRONMENT_BYTES", str(20 * 1024**3)
            )
        ),
    )
    host = os.environ.get("PACKAGE_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("PACKAGE_WORKER_PORT", "8091"))
    ThreadingHTTPServer((host, port), _handler(state)).serve_forever()


if __name__ == "__main__":
    main()
