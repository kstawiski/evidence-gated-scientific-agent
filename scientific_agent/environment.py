"""Typed client and ADK tools for isolated per-workspace package environments."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from packaging.requirements import InvalidRequirement, Requirement

from .config import EnvironmentSettings
from .provenance import canonical_json


R_PACKAGE = re.compile(r"^[A-Za-z][A-Za-z0-9.]{0,79}$")


def validate_python_packages(packages: list[str], maximum: int) -> list[str]:
    if not 1 <= len(packages) <= maximum:
        raise ValueError(f"request 1-{maximum} Python packages at a time")
    normalized = []
    for value in packages:
        if not isinstance(value, str) or len(value) > 180 or value.startswith("-"):
            raise ValueError("invalid Python package requirement")
        try:
            requirement = Requirement(value)
        except InvalidRequirement as exc:
            raise ValueError(f"invalid Python package requirement: {value}") from exc
        if requirement.url is not None:
            raise ValueError("direct URL or VCS Python requirements are not allowed")
        normalized.append(str(requirement))
    return list(dict.fromkeys(normalized))


def validate_r_packages(packages: list[str], maximum: int) -> list[str]:
    if not 1 <= len(packages) <= maximum:
        raise ValueError(f"request 1-{maximum} R packages at a time")
    normalized = []
    for value in packages:
        if not isinstance(value, str) or not R_PACKAGE.fullmatch(value):
            raise ValueError(f"invalid R package name: {value}")
        normalized.append(value)
    return list(dict.fromkeys(normalized))


def workspace_id_from_path(workspace: Path) -> str:
    parts = workspace.resolve().parts
    if len(parts) < 3 or parts[-1] != "files":
        raise ValueError("package environments require a managed web workspace")
    workspace_id = parts[-2]
    try:
        uuid.UUID(workspace_id)
    except ValueError as exc:
        raise ValueError("package environments require a UUID workspace") from exc
    return workspace_id


def cleanup_workspace_environment(
    settings: EnvironmentSettings,
    workspace_id: str,
) -> dict:
    """Delete package generations only after the web store has locked deletion."""

    try:
        parsed = uuid.UUID(workspace_id)
    except ValueError as exc:
        raise ValueError("invalid workspace ID") from exc
    if str(parsed) != workspace_id:
        raise ValueError("invalid workspace ID")
    if not settings.worker_url or not settings.worker_token:
        raise RuntimeError("package worker URL and token are both required")
    try:
        response = httpx.post(
            f"{settings.worker_url}/cleanup",
            headers={"Authorization": f"Bearer {settings.worker_token}"},
            json={"workspace_id": workspace_id},
            timeout=max(60, settings.install_timeout_seconds + 30),
        )
    except httpx.HTTPError as exc:
        raise RuntimeError("package worker cleanup request failed") from exc
    if response.is_error:
        try:
            detail = response.json().get("detail", "package worker cleanup failed")
        except ValueError:
            detail = "package worker returned a non-JSON cleanup error"
        raise RuntimeError(
            f"package worker returned HTTP {response.status_code}: {detail}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "package worker returned a non-JSON cleanup response"
        ) from exc
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "deleted"
        or payload.get("workspace_id") != workspace_id
        or not isinstance(payload.get("removed_bytes"), int)
        or payload["removed_bytes"] < 0
    ):
        raise RuntimeError("package worker returned an invalid cleanup response")
    return payload


@dataclass
class EnvironmentManager:
    workspace: Path
    settings: EnvironmentSettings
    evidence_path: Path
    cancel_event: threading.Event | None = None
    _records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.settings.worker_url or not self.settings.worker_token:
            raise RuntimeError("package worker URL and token are both required")
        self.workspace_id = workspace_id_from_path(self.workspace)

    def records(self) -> list[dict]:
        """Return controller-recorded installation results for bounded handoffs."""

        return list(self._records)

    def install(
        self,
        language: str,
        packages: list[str],
        repository: str,
    ) -> dict:
        if language == "python":
            packages = validate_python_packages(
                packages, self.settings.max_packages_per_call
            )
            repository = "pypi"
        elif language == "r":
            packages = validate_r_packages(
                packages, self.settings.max_packages_per_call
            )
            if repository not in {"cran", "bioconductor"}:
                raise ValueError("R repository must be cran or bioconductor")
        else:
            raise ValueError("unsupported package language")
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
            name=f"package-cancel-{request_id[:8]}",
            daemon=True,
        )
        watcher.start()
        try:
            response = httpx.post(
                f"{self.settings.worker_url}/install",
                headers={"Authorization": f"Bearer {self.settings.worker_token}"},
                json={
                    "request_id": request_id,
                    "workspace_id": self.workspace_id,
                    "language": language,
                    "repository": repository,
                    "packages": packages,
                    "timeout_seconds": self.settings.install_timeout_seconds,
                },
                timeout=self.settings.install_timeout_seconds + 30,
            )
        finally:
            finished.set()
        if response.is_error:
            try:
                detail = response.json().get("detail", "package worker request failed")
            except ValueError:
                detail = "package worker returned a non-JSON error"
            raise RuntimeError(
                f"package worker returned HTTP {response.status_code}: {detail}"
            )
        result = response.json()
        record = {
            "language": language,
            "repository": repository,
            "requested": packages,
            **result,
        }
        self._records.append(record)
        self.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with self.evidence_path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")
        return result


def build_environment_tools(manager: EnvironmentManager):
    def install_python_packages(packages: list[str]) -> dict:
        """Install PyPI packages into this workspace's isolated Python environment.

        Use canonical PyPI package requirements only (names with optional version
        constraints/extras). Direct URLs, VCS repositories, paths, and flags are denied.
        The environment is mounted read-only into later analysis calls.
        """

        return manager.install("python", packages, "pypi")

    def install_r_packages(packages: list[str], repository: str = "cran") -> dict:
        """Install packages from CRAN or Bioconductor for this workspace.

        Use this when a required R package is missing or the image version is too
        old. The current repository release is installed into a version-recorded,
        read-only workspace library used by later analysis calls. Do not silently
        replace the requested scientific method with a weaker package or Python.

        Args:
            packages: Canonical R package names.
            repository: Exactly "cran" or "bioconductor".
        """

        return manager.install("r", packages, repository)

    return [install_python_packages, install_r_packages]
