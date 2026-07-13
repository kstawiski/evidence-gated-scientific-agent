import pytest

from scientific_agent.schemas import SourceRecord, VerificationReport


def test_fail_requires_blocking_finding():
    with pytest.raises(ValueError):
        VerificationReport(verdict="fail")


def test_inconclusive_may_have_no_blocking_finding():
    report = VerificationReport(verdict="inconclusive")
    assert report.verdict == "inconclusive"


def test_source_record_requires_exactly_one_evidence_location():
    common = {
        "source_id": "s1",
        "title": "Evidence",
        "source_type": "dataset",
        "retrieved_at": "2026-07-13T00:00:00Z",
        "supporting_passage": "Evidence summary.",
    }
    with pytest.raises(ValueError):
        SourceRecord(**common)
    with pytest.raises(ValueError):
        SourceRecord(
            **common,
            url="https://example.com",
            artifact_path="/tmp/result.csv",
        )
