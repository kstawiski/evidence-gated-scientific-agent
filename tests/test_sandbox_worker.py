import uuid
from pathlib import Path

import pytest

from scientific_agent.config import SandboxSettings
from scientific_agent.sandbox_worker import ExecuteRequest, WorkerState


def _state(tmp_path: Path) -> tuple[WorkerState, Path, Path]:
    workspace_id = str(uuid.uuid4())
    workspace = tmp_path / "workspaces" / workspace_id / "files"
    root = tmp_path / "workspaces" / workspace_id / "runs" / "run-1" / "computations"
    workspace.mkdir(parents=True)
    root.mkdir(parents=True)
    state = WorkerState(tmp_path, "x" * 32, SandboxSettings())
    return state, workspace, root


def test_worker_accepts_only_matching_confined_workspace_and_run_paths(tmp_path):
    state, workspace, root = _state(tmp_path)
    request = ExecuteRequest(
        workspace=str(workspace),
        computation_root=str(root),
        language="python",
        code="print(1)",
        timeout_seconds=10,
    )
    assert state.confined_paths(request) == (workspace.resolve(), root.resolve())

    request.computation_root = str(tmp_path / "outside")
    with pytest.raises(ValueError):
        state.confined_paths(request)


def test_worker_rejects_a_run_path_belonging_to_another_workspace(tmp_path):
    state, workspace, _ = _state(tmp_path)
    other = tmp_path / "workspaces" / str(uuid.uuid4()) / "runs" / "run-2"
    other.mkdir(parents=True)
    request = ExecuteRequest(
        workspace=str(workspace),
        computation_root=str(other),
        language="r",
        code="print(1)",
        timeout_seconds=10,
    )
    with pytest.raises(ValueError, match="invalid workspace"):
        state.confined_paths(request)


def test_handoff_tolerates_root_squashed_nfs_chown(tmp_path, monkeypatch):
    state, _, root = _state(tmp_path)
    artifact = root / "artifact.txt"
    artifact.write_text("result", encoding="utf-8")

    def denied(*args, **kwargs):
        del args, kwargs
        raise PermissionError("root squash")

    monkeypatch.setattr("scientific_agent.sandbox_worker.os.chown", denied)
    state._handoff_ownership(root)
