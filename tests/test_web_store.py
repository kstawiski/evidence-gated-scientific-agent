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
    assert (
        store.file_path(workspace["id"], "values.csv")
        .read_bytes()
        .startswith(b"group,value")
    )

    for unsafe in ("../secret", "subdir/file.csv", "subdir\\file.csv"):
        with pytest.raises(ValueError, match="path"):
            store.save_file(workspace["id"], unsafe, io.BytesIO(b"x"), 1024)
    with pytest.raises(ValueError, match="upload limit"):
        store.save_file(workspace["id"], "large.bin", io.BytesIO(b"1234"), 3)
    assert not (
        tmp_path / "workspaces" / workspace["id"] / "files" / "large.bin"
    ).exists()


@pytest.mark.asyncio
async def test_raw_stream_upload_is_atomic_and_size_bounded(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Large stream")

    async def chunks():
        yield b"a" * 7
        yield b"b" * 5

    saved = await store.save_streamed_file(workspace["id"], "large.bin", chunks(), 12)
    assert saved == {"name": "large.bin", "bytes": 12}
    assert store.file_path(workspace["id"], "large.bin").read_bytes() == (
        b"a" * 7 + b"b" * 5
    )

    async def oversized():
        yield b"1234"

    with pytest.raises(ValueError, match="upload limit"):
        await store.save_streamed_file(workspace["id"], "oversized.bin", oversized(), 3)
    assert not any(
        item.name.startswith(".upload-")
        for item in store.paths(workspace["id"])[0].iterdir()
    )


@pytest.mark.asyncio
async def test_run_started_during_stream_upload_wins_without_partial_input(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Upload and run race")

    async def chunks():
        yield b"first"
        store.create_run(workspace["id"], "Lock the immutable inputs", True, ())
        yield b"second"

    with pytest.raises(RuntimeError, match="run is active"):
        await store.save_streamed_file(workspace["id"], "racing.bin", chunks(), 64)

    assert not (store.paths(workspace["id"])[0] / "racing.bin").exists()
    assert not any(
        item.name.startswith(".upload-")
        for item in store.paths(workspace["id"])[0].iterdir()
    )
    assert len(store.list_runs(workspace["id"])) == 1


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


def test_requested_output_artifacts_round_trip_with_run(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Presentation analysis")

    run = store.create_run(
        workspace["id"],
        "Analyze and present the results",
        True,
        (),
        requested_outputs=("pptx_presentation", "data_bundle"),
    )

    assert run["requested_outputs"] == ["pptx_presentation", "data_bundle"]


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


def test_active_run_blocks_upload_and_events_are_monotonic(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Locked inputs")
    run = store.create_run(workspace["id"], "Perform the analysis", True, ())

    with pytest.raises(RuntimeError, match="upload files"):
        store.save_file(workspace["id"], "late.csv", io.BytesIO(b"x\n1\n"), 1024)

    assert store.start_run(run["id"])
    store.update_run(
        run["id"],
        phase="research",
        message="Qwen is computing",
        event_type="phase_changed",
        actor="Qwen",
    )
    events = store.list_events(run["id"])
    assert [event["id"] for event in events] == sorted(event["id"] for event in events)
    assert [event["actor"] for event in events][-1] == "Qwen"


def test_cancellation_is_idempotent_and_wins_over_completion(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Cancel race")
    run = store.create_run(workspace["id"], "Perform the analysis", True, ())
    assert store.start_run(run["id"])

    first = store.request_cancel(run["id"])
    second = store.request_cancel(run["id"])
    assert first["status"] == second["status"] == "cancel_requested"
    assert not store.finish_run(
        run["id"],
        status="supported",
        phase="complete",
        message="Must not commit",
        finished_at="2026-07-14T00:00:00Z",
    )
    assert store.mark_cancelled(run["id"])
    assert store.get_run(run["id"])["status"] == "cancelled"
    assert [event["event_type"] for event in store.list_events(run["id"])].count(
        "cancel_requested"
    ) == 1


def test_revision_run_records_parent_lineage_and_inherits_workspace(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Revision lineage")
    parent = store.create_run(workspace["id"], "Original analysis", True, ("context7",))
    store.finish_run(
        parent["id"],
        status="supported",
        phase="complete",
        message="Ready",
        finished_at="2026-07-14T00:00:00Z",
    )

    child = store.create_run(
        workspace["id"],
        "Clarify methods",
        parent["enable_code"],
        tuple(parent["mcp_servers"]),
        parent_run_id=parent["id"],
        run_kind="revision",
    )

    assert child["parent_run_id"] == parent["id"]
    assert child["run_kind"] == "revision"
    assert child["mcp_servers"] == ["context7"]


def test_report_discussion_is_ordered_and_allows_only_one_generating_reply(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Discussion")
    run = store.create_run(workspace["id"], "Completed analysis", False, ())
    store.finish_run(
        run["id"],
        status="supported",
        phase="complete",
        message="Ready",
        finished_at="2026-07-15T00:00:00Z",
    )

    response_id = store.start_discussion(run["id"], "Explain the result", "s8-gemma")
    with pytest.raises(RuntimeError, match="already answering"):
        store.start_discussion(run["id"], "A concurrent question", "s8-gemma")
    response = store.finish_discussion(
        response_id,
        content="The result is bounded by C1.",
        evidence_refs=["C1"],
        unresolved_uncertainties=["External validity is unknown."],
        suggested_revision_prompt="Clarify external validity and preserve C1.",
    )

    assert response["status"] == "complete"
    assert response["evidence_refs"] == ["C1"]
    messages = store.list_discussion(run["id"])
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[1]["suggested_revision_prompt"].startswith("Clarify")


def test_workspace_deletion_runs_cleanup_and_preserves_retained_workspace(tmp_path):
    store = _store(tmp_path)
    target = store.create_workspace("Delete target")
    retained = store.create_workspace("Retain target")
    target_root = store._workspace_root(target["id"])
    retained_root = store._workspace_root(retained["id"])
    target_run = store.create_run(target["id"], "Completed target analysis", False, ())
    retained_run = store.create_run(
        retained["id"], "Completed retained analysis", False, ()
    )
    for run_id in (target_run["id"], retained_run["id"]):
        store.finish_run(
            run_id,
            status="supported",
            phase="complete",
            message="Ready",
            finished_at="2026-07-14T00:00:00Z",
        )
    (target_root / "files" / "target.txt").write_text("delete", encoding="utf-8")
    (retained_root / "files" / "retained.txt").write_text("keep", encoding="utf-8")
    callbacks = []

    store.delete_workspace(
        target["id"], before_delete=lambda: callbacks.append(target["id"])
    )

    assert callbacks == [target["id"]]
    with pytest.raises(KeyError, match="workspace not found"):
        store.get_workspace(target["id"])
    with pytest.raises(KeyError, match="run not found"):
        store.get_run(target_run["id"])
    assert not target_root.exists()
    assert store.get_workspace(retained["id"])["name"] == "Retain target"
    assert store.get_run(retained_run["id"])["status"] == "supported"
    assert (retained_root / "files" / "retained.txt").read_text(
        encoding="utf-8"
    ) == "keep"


def test_workspace_deletion_rolls_back_when_environment_cleanup_fails(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Cleanup failure")
    run = store.create_run(workspace["id"], "Completed analysis", False, ())
    store.finish_run(
        run["id"],
        status="supported",
        phase="complete",
        message="Ready",
        finished_at="2026-07-14T00:00:00Z",
    )
    root = store._workspace_root(workspace["id"])
    (root / "files" / "evidence.txt").write_text("keep", encoding="utf-8")

    def fail_cleanup():
        raise RuntimeError("package cleanup unavailable")

    with pytest.raises(RuntimeError, match="package cleanup unavailable"):
        store.delete_workspace(workspace["id"], before_delete=fail_cleanup)

    assert store.get_workspace(workspace["id"])["name"] == "Cleanup failure"
    assert store.get_run(run["id"])["status"] == "supported"
    assert (root / "files" / "evidence.txt").read_text(encoding="utf-8") == "keep"


def test_active_run_blocks_workspace_cleanup_before_worker_call(tmp_path):
    store = _store(tmp_path)
    workspace = store.create_workspace("Active deletion guard")
    store.create_run(workspace["id"], "Analysis still queued", False, ())
    called = False

    def cleanup():
        nonlocal called
        called = True

    with pytest.raises(RuntimeError, match="active run"):
        store.delete_workspace(workspace["id"], before_delete=cleanup)

    assert called is False
    assert store.get_workspace(workspace["id"])["name"] == "Active deletion guard"
