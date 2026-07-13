import json

from scientific_agent.provenance import (
    EventLedger,
    build_environment_snapshot,
    build_input_manifest,
    build_manifest,
)


def test_event_ledger_is_append_only_jsonl_and_manifest_hashes_it(tmp_path):
    ledger = EventLedger(tmp_path / "tool_call_log.jsonl")
    ledger.append("one", {"value": 1})
    ledger.append("two", {"value": 2})
    records = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [item["event_type"] for item in records] == ["one", "two"]
    manifest = build_manifest(tmp_path)
    assert manifest["files"][0]["sha256"]


def test_input_manifest_hashes_only_flat_regular_files(tmp_path):
    (tmp_path / "values.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    manifest = build_input_manifest(tmp_path)
    assert [item["path"] for item in manifest["files"]] == ["values.csv"]
    assert len(manifest["files"][0]["sha256"]) == 64


def test_environment_snapshot_has_reproducibility_identity():
    snapshot = build_environment_snapshot(application_version="test")
    assert snapshot["application"] == {"name": "Evidence Bench", "version": "test"}
    assert snapshot["python"]["version"]
    assert snapshot["platform"]["system"]
