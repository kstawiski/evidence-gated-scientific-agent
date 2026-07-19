"""Network-enabled package installer isolated from research data and secrets."""

from __future__ import annotations

import importlib.metadata
import hashlib
import json
import os
import re
import secrets
import signal
import shutil
import stat
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

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
ACTIVE_QUOTA_POLL_SECONDS = 0.1


class InstallRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")
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


class CancelRequest(BaseModel):
    request_id: str = Field(pattern=r"^[0-9a-f-]{36}$")


class CleanupRequest(BaseModel):
    workspace_id: str


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
    max_workspace_bytes: int = 40 * 1024**3
    max_total_bytes: int = 200 * 1024**3
    max_environment_entries: int = 250_000
    max_workspace_entries: int = 500_000
    max_total_entries: int = 2_500_000
    locks: dict[str, threading.Lock] = field(default_factory=dict)
    cancellation_events: dict[str, threading.Event] = field(default_factory=dict)
    state_lock: threading.Lock = field(default_factory=threading.Lock)
    quota_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        for name, value in (
            ("max_environment_bytes", self.max_environment_bytes),
            ("max_workspace_bytes", self.max_workspace_bytes),
            ("max_total_bytes", self.max_total_bytes),
            ("max_environment_entries", self.max_environment_entries),
            ("max_workspace_entries", self.max_workspace_entries),
            ("max_total_entries", self.max_total_entries),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive")

    def workspace_root(self, workspace_id: str, *, create: bool = True) -> Path:
        try:
            parsed = uuid.UUID(workspace_id)
        except ValueError as exc:
            raise ValueError("invalid workspace ID") from exc
        if str(parsed) != workspace_id:
            raise ValueError("invalid workspace ID")
        environments = self.environments_dir.resolve()
        candidate = environments / workspace_id
        if candidate.is_symlink():
            raise ValueError("invalid workspace environment path")
        root = candidate.resolve(strict=False)
        if root != candidate or root.parent != environments:
            raise ValueError("invalid workspace environment path")
        if create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    @staticmethod
    def _generation_root(root: Path) -> Path:
        generations = root / ".generations"
        if generations.is_symlink():
            raise RuntimeError("workspace package generation root is invalid")
        generations.mkdir(mode=0o700, exist_ok=True)
        if generations.resolve() != generations or not generations.is_dir():
            raise RuntimeError("workspace package generation root is invalid")
        return generations

    @staticmethod
    def _directory_usage(
        root: Path,
        cancellation: threading.Event | None = None,
        *,
        stop_after_bytes: int | None = None,
        stop_after_entries: int | None = None,
    ) -> tuple[int, int]:
        if not root.exists():
            return 0, 0
        total_bytes = 0
        total_entries = 0
        for path in root.rglob("*"):
            if cancellation is not None and cancellation.is_set():
                raise InterruptedError("package installation cancelled")
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            total_entries += 1
            if stat.S_ISREG(info.st_mode):
                total_bytes += info.st_size
            if (stop_after_bytes is not None and total_bytes > stop_after_bytes) or (
                stop_after_entries is not None and total_entries > stop_after_entries
            ):
                break
        return total_bytes, total_entries

    @classmethod
    def _directory_bytes(
        cls,
        root: Path,
        cancellation: threading.Event | None = None,
    ) -> int:
        return cls._directory_usage(root, cancellation)[0]

    @classmethod
    def _directory_entries(
        cls,
        root: Path,
        cancellation: threading.Event | None = None,
    ) -> int:
        return cls._directory_usage(root, cancellation)[1]

    @staticmethod
    def health() -> dict:
        return {"status": "ok"}

    def _cumulative_quota_failure(
        self,
        root: Path,
        *,
        additional_bytes: int = 0,
        additional_entries: int = 0,
        cancellation: threading.Event | None = None,
    ) -> str | None:
        workspace_bytes, workspace_entries = self._directory_usage(root, cancellation)
        if workspace_bytes + additional_bytes > self.max_workspace_bytes:
            return "Workspace package-generation quota would be exceeded"
        if workspace_entries + additional_entries > self.max_workspace_entries:
            return "Workspace package-entry quota would be exceeded"
        total_bytes, total_entries = self._directory_usage(
            self.environments_dir, cancellation
        )
        if total_bytes + additional_bytes > self.max_total_bytes:
            return "Global package-environment quota would be exceeded"
        if total_entries + additional_entries > self.max_total_entries:
            return "Global package-entry quota would be exceeded"
        return None

    def _active_generation_allowances(
        self,
        root: Path,
        package_dir: Path,
    ) -> tuple[int, str, int, str]:
        """Return strictest active byte and entry allowances with their scopes."""

        package_bytes, package_entries = self._directory_usage(package_dir)
        workspace_bytes, workspace_entries = self._directory_usage(root)
        total_bytes, total_entries = self._directory_usage(self.environments_dir)
        workspace_byte_base = max(0, workspace_bytes - package_bytes)
        global_byte_base = max(
            0,
            total_bytes - package_bytes,
        )
        workspace_entry_base = max(0, workspace_entries - package_entries)
        global_entry_base = max(0, total_entries - package_entries)
        byte_allowances = (
            (self.max_environment_bytes, "per-generation"),
            (self.max_workspace_bytes - workspace_byte_base, "workspace"),
            (self.max_total_bytes - global_byte_base, "global"),
        )
        entry_allowances = (
            (self.max_environment_entries, "per-generation"),
            (
                self.max_workspace_entries - workspace_entry_base,
                "workspace",
            ),
            (self.max_total_entries - global_entry_base, "global"),
        )
        byte_allowance, byte_scope = min(byte_allowances, key=lambda item: item[0])
        entry_allowance, entry_scope = min(entry_allowances, key=lambda item: item[0])
        return (
            max(0, byte_allowance),
            byte_scope,
            max(0, entry_allowance),
            entry_scope,
        )

    @classmethod
    def _active_quota_failure(
        cls,
        package_dir: Path,
        cancellation: threading.Event,
        *,
        byte_allowance: int,
        byte_scope: str,
        entry_allowance: int,
        entry_scope: str,
    ) -> str | None:
        active_bytes, active_entries = cls._directory_usage(
            package_dir,
            cancellation,
            stop_after_bytes=byte_allowance,
            stop_after_entries=entry_allowance,
        )
        if active_bytes > byte_allowance:
            return (
                "Active package installation exceeded "
                f"{byte_scope} quota: staging package tree uses "
                f"{active_bytes} bytes; active allowance is {byte_allowance} bytes"
            )
        if active_entries > entry_allowance:
            return (
                "Active package installation exceeded "
                f"{entry_scope} entry quota: staging package tree uses "
                f"{active_entries} entries; active allowance is "
                f"{entry_allowance} entries"
            )
        return None

    @staticmethod
    def _quota_failure_result(message: str, *, stderr: str = "") -> dict:
        return {
            "status": "failed",
            "exit_code": 1,
            "stdout": "",
            "stderr": f"{stderr}\n{message}".lstrip(),
            "installed": [],
        }

    def cleanup(self, workspace_id: str) -> dict:
        root = self.workspace_root(workspace_id, create=False)
        keys = [f"{workspace_id}:python", f"{workspace_id}:r"]
        with self.state_lock:
            locks = [self.locks.setdefault(key, threading.Lock()) for key in keys]
        with locks[0], locks[1], self.quota_lock:
            removed_bytes = self._directory_bytes(root)
            if root.exists():
                if root.is_symlink() or not root.is_dir():
                    raise RuntimeError("workspace package environment is invalid")
                shutil.rmtree(root)
            return {
                "status": "deleted",
                "workspace_id": workspace_id,
                "removed_bytes": removed_bytes,
            }

    def install(self, request: InstallRequest) -> dict:
        cancellation = threading.Event()
        with self.state_lock:
            if request.request_id in self.cancellation_events:
                raise ValueError("duplicate package request ID")
            self.cancellation_events[request.request_id] = cancellation
        try:
            packages = (
                validate_python_packages(request.packages, self.max_packages)
                if request.language == "python"
                else validate_r_packages(request.packages, self.max_packages)
            )
            root = self.workspace_root(request.workspace_id, create=False)
            key = f"{request.workspace_id}:{request.language}"
            with self.state_lock:
                lock = self.locks.setdefault(key, threading.Lock())
            with lock:
                with self.quota_lock:
                    return self._install_locked(root, request, packages, cancellation)
        finally:
            with self.state_lock:
                self.cancellation_events.pop(request.request_id, None)

    def cancel(self, request_id: str) -> bool:
        with self.state_lock:
            event = self.cancellation_events.get(request_id)
            if event is None:
                return False
            event.set()
            return True

    def _install_locked(
        self,
        root: Path,
        request: InstallRequest,
        packages: list[str],
        cancellation: threading.Event,
    ) -> dict:
        if cancellation.is_set():
            return self._cancelled_result()
        root_exists = root.exists()
        destination = root / request.language
        if root_exists:
            reused = self._reuse_active_generation(
                root,
                destination,
                request.language,
                packages,
            )
            if reused is not None:
                return reused
        generations_exist = (root / ".generations").exists() if root_exists else False
        metadata_entries = int(not root_exists) + int(not generations_exist)
        try:
            quota_failure = self._cumulative_quota_failure(
                root,
                additional_entries=metadata_entries,
                cancellation=cancellation,
            )
        except InterruptedError:
            return self._cancelled_result()
        if quota_failure:
            return self._quota_failure_result(quota_failure)
        if not root_exists:
            root.mkdir(parents=True, exist_ok=False)
        replaces_existing_destination = destination.exists() or destination.is_symlink()
        try:
            quota_failure = self._cumulative_quota_failure(
                root, cancellation=cancellation
            )
        except InterruptedError:
            return self._cancelled_result()
        if quota_failure:
            return self._quota_failure_result(quota_failure)
        generations = self._generation_root(root)
        staging = generations / f"{request.language}-{uuid.uuid4().hex}"
        package_dir = staging / "packages"
        previous_lock: dict = {}
        if replaces_existing_destination:
            previous = destination.resolve()
            if previous.parent != generations.resolve():
                raise RuntimeError("workspace package environment target is invalid")
            if not (previous / "packages").is_dir():
                raise RuntimeError("workspace package generation is incomplete")
            copied_bytes, copied_entries = self._directory_usage(previous / "packages")
            if copied_bytes > self.max_environment_bytes:
                return self._quota_failure_result(
                    "Existing package generation exceeds per-generation quota"
                )
            if copied_entries > self.max_environment_entries:
                return self._quota_failure_result(
                    "Existing package generation exceeds per-generation entry quota"
                )
            try:
                quota_failure = self._cumulative_quota_failure(
                    root,
                    additional_bytes=copied_bytes,
                    additional_entries=copied_entries + 2,
                    cancellation=cancellation,
                )
            except InterruptedError:
                return self._cancelled_result()
            if quota_failure:
                return self._quota_failure_result(quota_failure)
            shutil.copytree(previous / "packages", package_dir, symlinks=True)
            try:
                previous_lock = json.loads(
                    (previous / "lock.json").read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                previous_lock = {}
        else:
            package_dir.mkdir(parents=True, mode=0o700)
        if cancellation.is_set():
            shutil.rmtree(staging, ignore_errors=True)
            return self._cancelled_result()
        with tempfile.TemporaryDirectory(prefix="evidence-package-") as temporary:
            temp = Path(temporary)
            if os.geteuid() == 0:
                self._chown_tree(staging, INSTALL_UID, INSTALL_GID)
                os.chown(temp, INSTALL_UID, INSTALL_GID)
            stdout = temp / "stdout.txt"
            stderr = temp / "stderr.txt"
            command, environment = self._command(request, packages, package_dir, temp)
            timed_out = False
            cancelled = False
            active_quota_failure: str | None = None
            (
                active_byte_allowance,
                active_byte_scope,
                active_entry_allowance,
                active_entry_scope,
            ) = self._active_generation_allowances(root, package_dir)
            with stdout.open("wb") as out, stderr.open("wb") as err:
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=out,
                    stderr=err,
                    env=environment,
                    start_new_session=True,
                )
                deadline = time.monotonic() + request.timeout_seconds
                next_quota_check = time.monotonic()
                while process.poll() is None:
                    if cancellation.is_set():
                        cancelled = True
                        break
                    now = time.monotonic()
                    if now >= deadline:
                        timed_out = True
                        break
                    if now >= next_quota_check:
                        try:
                            active_quota_failure = self._active_quota_failure(
                                package_dir,
                                cancellation,
                                byte_allowance=active_byte_allowance,
                                byte_scope=active_byte_scope,
                                entry_allowance=active_entry_allowance,
                                entry_scope=active_entry_scope,
                            )
                        except InterruptedError:
                            cancelled = True
                            break
                        if active_quota_failure is not None:
                            break
                        next_quota_check = now + ACTIVE_QUOTA_POLL_SECONDS
                    time.sleep(0.05)
                if cancelled or timed_out or active_quota_failure is not None:
                    self._terminate_process_group(process)
                else:
                    process.wait()
                exit_code = process.returncode
            if (
                exit_code == 0
                and not timed_out
                and not cancelled
                and active_quota_failure is None
            ):
                try:
                    active_quota_failure = self._active_quota_failure(
                        package_dir,
                        cancellation,
                        byte_allowance=active_byte_allowance,
                        byte_scope=active_byte_scope,
                        entry_allowance=active_entry_allowance,
                        entry_scope=active_entry_scope,
                    )
                except InterruptedError:
                    cancelled = True
            if active_quota_failure is not None:
                shutil.rmtree(staging, ignore_errors=True)
                return self._quota_failure_result(
                    active_quota_failure,
                    stderr=_bounded(stderr),
                )
            if exit_code != 0 or timed_out or cancelled:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": (
                        "cancelled"
                        if cancelled
                        else "timed_out"
                        if timed_out
                        else "failed"
                    ),
                    "exit_code": exit_code,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr),
                    "installed": [],
                }

            if cancellation.is_set():
                shutil.rmtree(staging, ignore_errors=True)
                return self._cancelled_result(_bounded(stdout), _bounded(stderr))
            inventory = self._inventory(request.language, package_dir)
            if cancellation.is_set():
                shutil.rmtree(staging, ignore_errors=True)
                return self._cancelled_result(_bounded(stdout), _bounded(stderr))
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
            try:
                (
                    package_tree_sha256,
                    package_tree_bytes,
                    package_tree_entries,
                ) = self._tree_identity(package_dir, cancellation)
            except InterruptedError:
                shutil.rmtree(staging, ignore_errors=True)
                return self._cancelled_result(_bounded(stdout), _bounded(stderr))
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
            if package_tree_entries > self.max_environment_entries:
                shutil.rmtree(staging, ignore_errors=True)
                return {
                    "status": "failed",
                    "exit_code": 1,
                    "stdout": _bounded(stdout),
                    "stderr": _bounded(stderr)
                    + "\nEnvironment exceeds "
                    + f"{self.max_environment_entries} entries",
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
                        "package_tree_entries": package_tree_entries,
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
            try:
                quota_failure = self._cumulative_quota_failure(
                    root,
                    additional_entries=0 if replaces_existing_destination else 1,
                    cancellation=cancellation,
                )
            except InterruptedError:
                shutil.rmtree(staging, ignore_errors=True)
                return self._cancelled_result(_bounded(stdout), _bounded(stderr))
            if quota_failure:
                shutil.rmtree(staging, ignore_errors=True)
                return self._quota_failure_result(
                    quota_failure,
                    stderr=_bounded(stderr),
                )
            link = root / f".{request.language}-link-{uuid.uuid4().hex}"
            link.symlink_to(Path(".generations") / staging.name)
            with self.state_lock:
                if cancellation.is_set():
                    link.unlink(missing_ok=True)
                    shutil.rmtree(staging, ignore_errors=True)
                    return self._cancelled_result(_bounded(stdout), _bounded(stderr))
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
                "package_tree_entries": package_tree_entries,
            }

    def _reuse_active_generation(
        self,
        root: Path,
        destination: Path,
        language: str,
        packages: list[str],
    ) -> dict | None:
        """Return an existing immutable generation when it satisfies the request."""

        if not (destination.exists() or destination.is_symlink()):
            return None
        generations = root / ".generations"
        if generations.is_symlink() or not generations.is_dir():
            raise RuntimeError("workspace package generation root is invalid")
        previous = destination.resolve()
        if previous.parent != generations.resolve():
            raise RuntimeError("workspace package environment target is invalid")
        package_dir = previous / "packages"
        lock_path = previous / "lock.json"
        if not package_dir.is_dir() or not lock_path.is_file():
            raise RuntimeError("workspace package generation is incomplete")
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        inventory = lock.get("installed")
        package_tree_sha256 = lock.get("package_tree_sha256")
        package_tree_bytes = lock.get("package_tree_bytes")
        package_tree_entries = lock.get("package_tree_entries")
        if (
            lock.get("language") != language
            or not isinstance(inventory, list)
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not isinstance(item.get("version"), str)
                for item in inventory
            )
            or not isinstance(package_tree_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", package_tree_sha256) is None
            or not isinstance(package_tree_bytes, int)
            or package_tree_bytes < 0
            or not isinstance(package_tree_entries, int)
            or package_tree_entries < 0
        ):
            return None
        if (
            package_tree_bytes > self.max_environment_bytes
            or package_tree_entries > self.max_environment_entries
            or self._missing_requested(language, packages, inventory)
        ):
            return None
        return {
            "status": "succeeded",
            "exit_code": 0,
            "stdout": (
                "All requested packages are already present in the active "
                "locked generation."
            ),
            "stderr": "",
            "installed": inventory,
            "lock_file": str(lock_path),
            "generation": previous.name,
            "package_tree_sha256": package_tree_sha256,
            "package_tree_bytes": package_tree_bytes,
            "package_tree_entries": package_tree_entries,
            "reused": True,
        }

    @staticmethod
    def _cancelled_result(stdout: str = "", stderr: str = "") -> dict:
        return {
            "status": "cancelled",
            "exit_code": None,
            "stdout": stdout,
            "stderr": stderr,
            "installed": [],
        }

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
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
    def _tree_identity(
        package_dir: Path,
        cancellation: threading.Event | None = None,
    ) -> tuple[str, int, int]:
        digest = hashlib.sha256()
        total = 0
        paths = sorted(package_dir.rglob("*"))
        for path in paths:
            if cancellation is not None and cancellation.is_set():
                raise InterruptedError("package installation cancelled")
            relative = path.relative_to(package_dir).as_posix().encode("utf-8")
            if path.is_symlink():
                digest.update(b"L\0" + relative + b"\0" + os.readlink(path).encode())
            elif path.is_file():
                size = path.stat().st_size
                total += size
                digest.update(b"F\0" + relative + b"\0" + str(size).encode() + b"\0")
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        if cancellation is not None and cancellation.is_set():
                            raise InterruptedError("package installation cancelled")
                        digest.update(chunk)
            elif path.is_dir():
                digest.update(b"D\0" + relative + b"\0")
        return digest.hexdigest(), total, len(paths)

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
        proxy_url = os.environ.get("SCIENTIFIC_AGENT_PACKAGE_PROXY_URL", "").strip()
        if proxy_url:
            parsed_proxy = urlparse(proxy_url)
            if (
                parsed_proxy.scheme != "http"
                or not parsed_proxy.hostname
                or parsed_proxy.username
                or parsed_proxy.password
                or parsed_proxy.path not in {"", "/"}
                or parsed_proxy.query
                or parsed_proxy.fragment
            ):
                raise ValueError("package proxy must be an unauthenticated HTTP origin")
            environment.update(
                {
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                    "http_proxy": proxy_url,
                    "https_proxy": proxy_url,
                    "NO_PROXY": "",
                    "no_proxy": "",
                }
            )
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
                for name, version in sorted(
                    values.items(), key=lambda item: item[0].lower()
                )
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
        server_version = "EvidenceBenchPackages/0.4"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._json(200, state.health())
            else:
                self._json(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in {"/install", "/cancel", "/cleanup"}:
                self._json(404, {"detail": "not found"})
                return
            expected = f"Bearer {state.token}"
            if not secrets.compare_digest(
                self.headers.get("Authorization", ""), expected
            ):
                self._json(401, {"detail": "valid package worker token required"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"detail": "invalid content length"})
                return
            max_size = 32 * 1024 if self.path == "/install" else 4096
            if size < 1 or size > max_size:
                self._json(413, {"detail": "request body exceeds package worker limit"})
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
                if self.path == "/cleanup":
                    cleanup = CleanupRequest.model_validate_json(body)
                    self._json(200, state.cleanup(cleanup.workspace_id))
                    return
                request = InstallRequest.model_validate_json(body)
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
        raise RuntimeError(
            "SCIENTIFIC_AGENT_PACKAGE_WORKER_TOKEN must be at least 24 characters"
        )
    environments = Path(
        os.environ.get("SCIENTIFIC_AGENT_ENVIRONMENTS_DIR", "/environments")
    ).resolve()
    environments.mkdir(parents=True, exist_ok=True)
    state = EnvironmentWorkerState(
        environments,
        token,
        max_packages=int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_PACKAGES_PER_CALL", "24")
        ),
        max_environment_bytes=int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_ENVIRONMENT_BYTES", str(20 * 1024**3))
        ),
        max_workspace_bytes=int(
            os.environ.get(
                "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_BYTES",
                str(40 * 1024**3),
            )
        ),
        max_total_bytes=int(
            os.environ.get(
                "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_BYTES",
                str(200 * 1024**3),
            )
        ),
        max_environment_entries=int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_ENVIRONMENT_ENTRIES", "250000")
        ),
        max_workspace_entries=int(
            os.environ.get(
                "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_ENTRIES", "500000"
            )
        ),
        max_total_entries=int(
            os.environ.get("SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_ENTRIES", "2500000")
        ),
    )
    host = os.environ.get("PACKAGE_WORKER_HOST", "0.0.0.0")
    port = int(os.environ.get("PACKAGE_WORKER_PORT", "8091"))
    ThreadingHTTPServer((host, port), _handler(state)).serve_forever()


if __name__ == "__main__":
    main()
