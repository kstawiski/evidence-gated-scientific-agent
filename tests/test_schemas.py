import pytest

from scientific_agent.schemas import (
    PLAN_AUDIT_CRITERIA,
    PlanAuditChecklist,
    PlanAuditFinding,
    PlanAuditReview,
    ScientificReport,
    SourceRecord,
    VerificationReport,
)


def test_fail_requires_blocking_finding():
    with pytest.raises(ValueError):
        VerificationReport(verdict="fail")


def test_inconclusive_may_have_no_blocking_finding():
    report = VerificationReport(verdict="inconclusive")
    assert report.verdict == "inconclusive"


def test_plan_audit_requires_each_unique_fixed_criterion():
    reviews = [
        PlanAuditReview(criterion=criterion, status="pass")
        for criterion in PLAN_AUDIT_CRITERIA
    ]
    assert len(PlanAuditChecklist(reviews=reviews).reviews) == 5

    duplicated = list(reviews)
    duplicated[-1] = reviews[0]
    with pytest.raises(ValueError, match="unique"):
        PlanAuditChecklist(reviews=duplicated)


def test_plan_audit_status_requires_matching_finding():
    with pytest.raises(ValueError, match="require a finding"):
        PlanAuditReview(criterion=PLAN_AUDIT_CRITERIA[0], status="fail")

    finding = PlanAuditFinding(
        location="plan.steps[0]",
        plan_evidence_quote="validator: source check",
        problem="The check is not independent.",
        why_it_matters="A shared failure mode could pass undetected.",
        falsification_test_or_correction="Add an independent recomputation.",
    )
    normalized = PlanAuditReview(
        criterion=PLAN_AUDIT_CRITERIA[0],
        status="pass",
        finding=finding,
    )
    assert normalized.status == "inconclusive"
    assert normalized.finding == finding


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


def test_literature_markdown_duplicate_artifact_path_is_normalized():
    markdown = "/run/references/example.md"
    source = SourceRecord(
        source_id="pubmed-1",
        title="Acquired article",
        url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        artifact_path=markdown,
        pmid="12345",
        local_markdown_path=markdown,
        full_text_status="abstract_only",
        source_type="review",
        retrieved_at="2026-07-15T00:00:00Z",
        supporting_passage="A bounded article passage.",
    )

    assert source.artifact_path is None
    assert source.local_markdown_path == markdown


@pytest.mark.parametrize(
    "field,value",
    [
        ("pmid", "12345"),
        ("pmcid", "PMC12345"),
        ("citekey", "example-2026"),
        ("rights_status", "metadata_abstract_only_no_reuse_rights"),
        ("terms_warning", "Publisher terms apply."),
        ("local_pdf_path", "/run/references/example.pdf"),
        ("local_markdown_path", "/run/references/example.md"),
        ("full_text_status", "unavailable"),
    ],
)
def test_generic_web_source_cannot_claim_typed_literature_acquisition(field, value):
    payload = {
        "source_id": "web-1",
        "title": "Browser result",
        "url": "https://example.com/page",
        "doi": "10.1000/example",
        "source_type": "web_page",
        "retrieved_at": "2026-07-15T00:00:00Z",
        "supporting_passage": "A browser-observed passage.",
        field: value,
    }
    with pytest.raises(ValueError, match="literature-acquisition fields null"):
        SourceRecord(**payload)


def test_generic_web_source_may_preserve_observed_doi_and_license():
    source = SourceRecord(
        source_id="web-1",
        title="Browser result",
        url="https://example.com/page",
        doi="10.1000/example",
        license="CC-BY-4.0",
        source_type="web_page",
        retrieved_at="2026-07-15T00:00:00Z",
        supporting_passage="A browser-observed passage.",
    )

    assert source.doi == "10.1000/example"
    assert source.full_text_status is None


def test_scientific_report_requires_distinct_article_sections():
    common = {
        "title": "Structured report",
        "executive_summary": "The abstract summarizes the result.",
        "introduction": "The introduction states the objective.",
        "methods": ["A reproducible method"],
        "results": "The results state the observed evidence.",
        "discussion": "The discussion interprets the evidence.",
        "conclusions": "The conclusion remains bounded.",
        "claims": [],
        "sources": [],
    }
    assert ScientificReport(**common).results.startswith("The results")
    for field in ("introduction", "results", "discussion", "conclusions"):
        invalid = dict(common)
        invalid[field] = ""
        with pytest.raises(ValueError):
            ScientificReport(**invalid)
