import json

from scientific_agent.provenance import EventLedger, build_manifest


def test_event_ledger_is_append_only_jsonl_and_manifest_hashes_it(tmp_path):
    ledger = EventLedger(tmp_path / "tool_call_log.jsonl")
    ledger.append("one", {"value": 1})
    ledger.append("two", {"value": 2})
    records = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert [item["event_type"] for item in records] == ["one", "two"]
    manifest = build_manifest(tmp_path)
    assert manifest["files"][0]["sha256"]
