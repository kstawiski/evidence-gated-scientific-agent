import io
from concurrent.futures import ThreadPoolExecutor

import pytest

from scientific_agent.web.store import WorkspaceStore


def _store(tmp_path):
    return WorkspaceStore(tmp_path / "state.sqlite3", tmp_path / "workspaces")


def test_workspace_files_are_confined_and_have_an_upload_limit(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Cohort analysis")
    saved = store.save_file(
        workspace["id"], "values.csv", io.BytesIO(b"group,value\nA,1\n"), 1024
    )
    assert saved == {"name": "values.csv", "bytes": 16}
    assert store.file_path(workspace["id"], "values.csv").read_bytes().startswith(
        b"group,value"
    )

    for unsafe in ("../secret", "subdir/file.csv", "subdir\\file.csv"):
        with pytest.raises(ValueError, match="path"):
            store.save_file(workspace["id"], unsafe, io.BytesIO(b"x"), 1024)
    with pytest.raises(ValueError, match="upload limit"):
        store.save_file(workspace["id"], "large.bin", io.BytesIO(b"1234"), 3)
    assert not (tmp_path / "workspaces" / workspace["id"] / "files" / "large.bin").exists()


def test_only_one_active_run_can_be_created_per_workspace(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Race test")

    def create():
        try:
            return store.create_run(workspace["id"], "Perform the analysis", True, ())
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: create(), range(2)))
    assert sum(result is not None for result in results) == 1


def test_artifact_paths_cannot_escape_a_run(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Artifact test")
    run = store.create_run(workspace["id"], "Perform the analysis", False, ())
    _, runs = store.paths(workspace["id"])
    provenance = runs / "result"
    provenance.mkdir()
    (provenance / "report.md").write_text("report", encoding="utf-8")
    store.update_run(run["id"], status="supported", provenance_dir=str(provenance))

    assert store.run_artifact(run["id"], "report.md").read_text() == "report"
    with pytest.raises(KeyError):
        store.run_artifact(run["id"], "../../state.sqlite3")
