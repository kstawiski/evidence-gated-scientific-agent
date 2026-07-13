import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest

from scientific_agent.config import EnvironmentSettings
from scientific_agent.environment import (
    EnvironmentManager,
    validate_python_packages,
    validate_r_packages,
)
from scientific_agent.environment_worker import EnvironmentWorkerState, InstallRequest


def test_registry_requirements_allow_versions_but_deny_urls_flags_and_bad_r_names():
    assert validate_python_packages(["polars>=1.0", "scanpy[leiden]"], 4) == [
        "polars>=1.0",
        "scanpy[leiden]",
    ]
    for value in ("--index-url=https://example.test", "pkg @ https://example.test/pkg.whl"):
        with pytest.raises(ValueError):
            validate_python_packages([value], 4)
    assert validate_r_packages(["DESeq2", "data.table"], 4) == [
        "DESeq2",
        "data.table",
    ]
    with pytest.raises(ValueError):
        validate_r_packages(["github/user/repo"], 4)


def test_python_inventory_must_satisfy_requested_version():
    inventory = [{"name": "polars", "version": "1.0.0"}]
    assert EnvironmentWorkerState._missing_requested(
        "python", ["polars>=1"], inventory
    ) == []
    assert EnvironmentWorkerState._missing_requested(
        "python", ["polars>=2"], inventory
    ) == ["polars>=2"]


def test_worker_rejects_escaping_symlinks_and_special_files(tmp_path):
    packages = tmp_path / "packages"
    packages.mkdir()
    (packages / "safe.txt").write_text("safe", encoding="utf-8")
    (packages / "escape").symlink_to("/etc/passwd")
    os.mkfifo(packages / "pipe")

    assert sorted(EnvironmentWorkerState._unsafe_entries(packages)) == [
        "escape",
        "pipe",
    ]


def test_worker_confines_uuid_workspace_and_transactionally_commits(tmp_path, monkeypatch):
    state = EnvironmentWorkerState(tmp_path / "environments", "x" * 32)
    workspace_id = str(uuid.uuid4())

    def command(request, packages, staging, temporary):
        del request, packages, temporary
        return ["/bin/sh", "-c", f"touch {staging / 'installed'}"], {
            "PATH": "/usr/bin:/bin"
        }

    monkeypatch.setattr(state, "_command", command)
    monkeypatch.setattr(
        state,
        "_inventory",
        lambda language, path: [{"name": "polars", "version": "1.0.0"}],
    )
    result = state.install(
        InstallRequest(
            workspace_id=workspace_id,
            language="python",
            repository="pypi",
            packages=["polars"],
            timeout_seconds=30,
        )
    )
    assert result["status"] == "succeeded"
    generation = tmp_path / "environments" / workspace_id / "python"
    assert generation.is_symlink()
    assert (generation / "packages" / "installed").is_file()
    assert (generation / "lock.json").is_file()
    lock = __import__("json").loads((generation / "lock.json").read_text())
    assert lock["package_tree_sha256"]
    assert lock["package_tree_bytes"] == 0

    with pytest.raises(ValueError, match="invalid workspace ID"):
        state.workspace_root("../escape")


def test_manager_records_exact_worker_inventory(tmp_path, monkeypatch):
    workspace_id = str(uuid.uuid4())
    workspace = tmp_path / "data" / "workspaces" / workspace_id / "files"
    workspace.mkdir(parents=True)
    evidence = tmp_path / "run" / "package_installations.jsonl"

    class Response:
        is_error = False

        @staticmethod
        def json():
            return {
                "status": "succeeded",
                "installed": [{"name": "polars", "version": "1.0.0"}],
            }

    monkeypatch.setattr("scientific_agent.environment.httpx.post", lambda *a, **k: Response())
    settings = replace(
        EnvironmentSettings(), worker_url="http://packages:8091", worker_token="x" * 32
    )
    result = EnvironmentManager(workspace, settings, evidence).install(
        "python", ["polars"], "pypi"
    )
    assert result["installed"] == [{"name": "polars", "version": "1.0.0"}]
    assert '"version":"1.0.0"' in evidence.read_text(encoding="utf-8")
