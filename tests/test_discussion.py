import json

import pytest

from scientific_agent.config import Settings
from scientific_agent.discussion import _discussion_context, discuss_report
from scientific_agent.schemas import ReportDiscussionResponse


def test_discussion_context_uses_audited_report_surfaces_not_full_run_trace(tmp_path):
    report = {"title": "Audited report", "claims": [{"claim_id": "C1"}]}
    (tmp_path / "scientific_report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    (tmp_path / "deterministic_validation.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (tmp_path / "run_result.json").write_text(
        json.dumps({"private_workflow_trace": "x" * 100_000}), encoding="utf-8"
    )

    context = _discussion_context(tmp_path)

    assert context["scientific_report.json"] == report
    assert context["deterministic_validation.json"] == {"passed": True}
    assert "run_result.json" not in context


def test_discussion_context_requires_a_completed_report(tmp_path):
    (tmp_path / "gemma_review.json").write_text(
        json.dumps({"verdict": "pass"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="the run has no completed scientific report"):
        _discussion_context(tmp_path)


@pytest.mark.asyncio
async def test_discussion_honors_configured_gemma_temperature(tmp_path, monkeypatch):
    (tmp_path / "scientific_report.json").write_text(
        json.dumps({"title": "Audited report"}), encoding="utf-8"
    )
    observed = {}

    async def fake_request(*_args, **kwargs):
        observed.update(kwargs)
        return ReportDiscussionResponse(answer="Evidence-bounded explanation.")

    monkeypatch.setattr("scientific_agent.discussion.request_structured", fake_request)
    settings = Settings()

    await discuss_report(settings, tmp_path, [], "Explain this result.")

    assert observed["temperature"] == settings.gemma.temperature
