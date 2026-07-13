"""Typed client and ADK tools for isolated per-workspace package environments."""

from __future__ import annotations

import json
import re
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


@dataclass
class EnvironmentManager:
    workspace: Path
    settings: EnvironmentSettings
    evidence_path: Path
    _records: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.settings.worker_url or not self.settings.worker_token:
            raise RuntimeError("package worker URL and token are both required")
        self.workspace_id = workspace_id_from_path(self.workspace)

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
            packages = validate_r_packages(packages, self.settings.max_packages_per_call)
            if repository not in {"cran", "bioconductor"}:
                raise ValueError("R repository must be cran or bioconductor")
        else:
            raise ValueError("unsupported package language")
        response = httpx.post(
            f"{self.settings.worker_url}/install",
            headers={"Authorization": f"Bearer {self.settings.worker_token}"},
            json={
                "workspace_id": self.workspace_id,
                "language": language,
                "repository": repository,
                "packages": packages,
                "timeout_seconds": self.settings.install_timeout_seconds,
            },
            timeout=self.settings.install_timeout_seconds + 30,
        )
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

    def install_r_packages(
        packages: list[str], repository: str = "cran"
    ) -> dict:
        """Install packages from CRAN or Bioconductor for this workspace.

        Args:
            packages: Canonical R package names.
            repository: Exactly "cran" or "bioconductor".
        """

        return manager.install("r", packages, repository)

    return [install_python_packages, install_r_packages]
