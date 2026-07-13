from __future__ import annotations

import json

import pytest

from scientific_agent.cli import _read_objective
from scientific_agent.provenance import EventLedger


def test_private_prompt_file_and_stdin(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("bounded scientific task\n", encoding="utf-8")
    prompt.chmod(0o600)
    assert _read_objective(None, prompt) == "bounded scientific task"

    monkeypatch.setattr("sys.stdin.read", lambda: "stdin task\n")
    assert _read_objective("-", None) == "stdin task"


def test_prompt_file_rejects_permissive_mode(tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("secret task", encoding="utf-8")
    prompt.chmod(0o644)
    with pytest.raises(PermissionError):
        _read_objective(None, prompt)


def test_event_ledger_need_not_store_task_text(tmp_path):
    path = tmp_path / "events.jsonl"
    EventLedger(path).append(
        "run_started",
        {"run_id": "r1", "objective_sha256": "a" * 64, "objective_bytes": 12},
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert "objective" not in record
