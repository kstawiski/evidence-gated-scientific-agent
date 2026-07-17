import builtins
import json
import zipfile

from PIL import Image

from scientific_agent.linting import (
    _inferential_consistency_findings,
    lint_plan,
    reconciliation_verdict,
    validate_report,
)
from scientific_agent.provenance import sha256_file
from scientific_agent.schemas import (
    CheckSpec,
    ClaimRecord,
    ComputationEvidence,
    ComputationRecord,
    EvidenceStatus,
    InlineCitation,
    ArtifactRef,
    InputColumnProfile,
    InputFileProfile,
    InputProfile,
    PlanProposal,
    PlanStep,
    RetrievalEvidence,
    ReportDisplay,
    ScientificReport,
    SourceRecord,
    TaskSpec,
)


def task():
    return TaskSpec(
        task_id="t1",
        objective="Review an intervention",
        deliverables=["scientific report"],
        acceptance_tests=["claims cite sources"],
    )


def good_plan():
    return PlanProposal(
        plan_label="A",
        objective="Review an intervention",
        steps=[
            PlanStep(
                step_id="s1",
                objective="Retrieve evidence",
                outputs=["scientific report"],
                methods=["systematic source retrieval"],
                validators=[
                    CheckSpec(
                        check_id="c1",
                        description="source record exists",
                        check_type="source",
                    )
                ],
                stop_conditions=["retrieval complete or explicitly inconclusive"],
            )
        ],
        expected_artifacts=["scientific report"],
    )


def article_report(**overrides):
    values = {
        "title": "Test scientific report",
        "executive_summary": "A concise test summary.",
        "introduction": "This test defines a scientific objective.",
        "methods": ["A reproducible test method"],
        "results": "The test result is reported with its evidence status.",
        "discussion": "The interpretation remains bounded by the test evidence.",
        "conclusions": "The test conclusion follows from the recorded result.",
        "claims": [],
        "sources": [],
    }
    values.update(overrides)
    return ScientificReport(**values)


def reconciliation_document(
    python_path,
    r_path,
    *,
    python_value: float,
    r_value: float,
    tolerance: float = 1e-6,
):
    difference = abs(python_value - r_value)
    passed = difference <= tolerance
    return {
        "all_pass": passed,
        "comparisons": [
            {
                "metric": "primary_point_estimate",
                "python": {
                    "language": "python",
                    "artifact_sha256": sha256_file(python_path),
                    "json_path": "primary.point_estimate",
                    "value": python_value,
                },
                "r": {
                    "language": "r",
                    "artifact_sha256": sha256_file(r_path),
                    "json_path": "primary.point_estimate",
                    "value": r_value,
                },
                "absolute_difference": difference,
                "tolerance": tolerance,
                "passed": passed,
            }
        ],
    }


def test_plan_linter_accepts_complete_read_only_plan():
    assert lint_plan(task(), good_plan()).passed


def test_locked_reader_displays_cannot_disappear_from_report():
    validation = validate_report(
        article_report(), required_display_kinds=("figure", "table")
    )

    missing = [
        finding.message
        for finding in validation.findings
        if finding.code == "required_report_display_missing"
    ]
    assert len(missing) == 2
    assert any("reader-facing figure" in message for message in missing)
    assert any("reader-facing table" in message for message in missing)


def test_controller_method_lock_satisfies_confirmatory_lint_without_magic_words():
    confirmatory = task().model_copy(update={"scientific_risk": "confirmatory"})

    raw = lint_plan(confirmatory, good_plan())
    bound = lint_plan(confirmatory, good_plan(), controller_method_lock=True)

    assert "missing_method_lock" in {item.code for item in raw.findings}
    assert bound.passed


def test_plan_linter_rejects_duplicate_ids_and_irreversible_action():
    plan = good_plan()
    duplicate = plan.steps[0].model_copy(update={"security_risk": "irreversible"})
    plan.steps.append(duplicate)
    report = lint_plan(task(), plan)
    assert not report.passed
    assert {item.code for item in report.findings} >= {
        "duplicate_step_id",
        "irreversible_action",
    }


def test_plan_linter_rejects_order_based_semantic_arm_assignment():
    plan = good_plan()
    plan.steps[0].methods = [
        "Assign the lexicographically smallest group label as control and the "
        "other as treatment."
    ]

    report = lint_plan(task(), plan)

    assert "arbitrary_semantic_arm_mapping" in {
        finding.code for finding in report.findings
    }
    assert not report.passed


def test_plan_linter_rejects_observed_baseline_arm_assignment():
    plan = good_plan()
    plan.steps[0].methods = [
        "If labels are unclear, assign the group with the higher mean baseline "
        "to control and the other group to treatment."
    ]

    report = lint_plan(task(), plan)

    finding = next(
        item
        for item in report.findings
        if item.code == "arbitrary_semantic_arm_mapping"
    )
    assert finding.blocking
    assert "observed baselines, outcomes, covariates" in finding.message
    assert not report.passed


def test_plan_linter_rejects_role_mapping_keys_absent_from_input_profile():
    profiled_task = task().model_copy(
        update={
            "input_profile": InputProfile(
                total_files=1,
                profiled_files=1,
                files=[
                    InputFileProfile(
                        path="/workspace/cohort.csv",
                        sha256="a" * 64,
                        bytes=100,
                        detected_format="delimited_text",
                        media_type="text/csv",
                        inspection_status="complete",
                        columns=[
                            InputColumnProfile(
                                name="group",
                                inferred_types=["string"],
                                non_missing_count=40,
                                missing_count=0,
                                missing_fraction=0,
                                distinct_non_missing=2,
                                candidate_role_labels=["control", "treatment"],
                                candidate_role_labels_complete=True,
                            )
                        ],
                    )
                ],
            )
        }
    )
    plan = good_plan()
    plan.steps[0].methods = [
        "Use {'Group_A': 'control', 'Group_B': 'treatment'} as the role mapping."
    ]

    rejected = lint_plan(profiled_task, plan)
    plan.steps[0].methods = [
        "Use {'control': 'control', 'treatment': 'treatment'} as the role mapping."
    ]
    accepted = lint_plan(profiled_task, plan)

    assert "role_mapping_not_grounded_in_input_profile" in {
        finding.code for finding in rejected.findings
    }
    assert "role_mapping_not_grounded_in_input_profile" not in {
        finding.code for finding in accepted.findings
    }


def test_plan_linter_rejects_wrong_hedges_j_parentheses():
    hedges_task = task().model_copy(
        update={"objective": "Calculate Hedges g with J = 1 - 3/(4*N - 9)."}
    )
    plan = good_plan()
    plan.steps[0].validators[
        0
    ].description = "Check Hedges g uses J = 1 - 3/(4*(N - 9))."

    report = lint_plan(hedges_task, plan)

    assert "invalid_hedges_j_parentheses" in {
        finding.code for finding in report.findings
    }


def test_plan_linter_rejects_unsupported_design_classification():
    plan = good_plan()
    plan.assumptions = [
        "No causal inference is claimed; the analysis is strictly observational."
    ]

    rejected = lint_plan(task(), plan)
    explicitly_observational = task().model_copy(
        update={"objective": "Analyze this observational cohort study."}
    )
    accepted = lint_plan(explicitly_observational, plan)
    plan.assumptions = [
        "Allocation and sampling design are unspecified; do not infer observational status."
    ]
    unspecified = lint_plan(task(), plan)

    assert "unsupported_plan_design_classification" in {
        finding.code for finding in rejected.findings
    }
    assert "unsupported_plan_design_classification" not in {
        finding.code for finding in accepted.findings
    }
    assert "unsupported_plan_design_classification" not in {
        finding.code for finding in unspecified.findings
    }


def test_plan_linter_rejects_unrequested_normality_based_primary_stop():
    confirmatory = task().model_copy(
        update={
            "objective": (
                "Run a prespecified Welch t-test and inspect the data for outliers."
            ),
            "scientific_risk": "confirmatory",
        }
    )
    plan = good_plan()
    plan.steps[0].stop_conditions = [
        "Halt execution if the Shapiro-Wilk p-value is below 0.05 or any value "
        "exceeds three standard deviations from the mean."
    ]

    report = lint_plan(confirmatory, plan, controller_method_lock=True)

    finding = next(
        item
        for item in report.findings
        if item.code == "data_dependent_primary_analysis_stop"
    )
    assert finding.blocking
    assert "predefine a sensitivity analysis" in finding.message
    assert not report.passed


def test_plan_linter_rejects_invented_input_filename_and_accepts_manifest_name():
    planned = good_plan()
    planned.required_data = ["invented_dataset.csv", "uploaded dataset"]
    planned.steps[0].inputs = ["invented_dataset.csv"]
    manifest_task = task().model_copy(
        update={
            "available_inputs": [
                ArtifactRef(
                    path="/workspace/known_effect.csv",
                    sha256="a" * 64,
                    description="immutable uploaded workspace input",
                )
            ]
        }
    )

    rejected = lint_plan(manifest_task, planned)

    assert [item.code for item in rejected.findings].count(
        "unknown_plan_input_artifact"
    ) == 2
    planned.required_data = ["known_effect.csv", "uploaded dataset"]
    planned.steps[0].inputs = ["known_effect.csv"]
    accepted = lint_plan(manifest_task, planned)
    assert "unknown_plan_input_artifact" not in {
        item.code for item in accepted.findings
    }


def test_plan_linter_accepts_task_named_file_that_the_plan_acquires():
    acquisition_task = task().model_copy(
        update={
            "objective": (
                "Use the managed browser to import the downloaded paper-42158852.pdf."
            )
        }
    )
    planned = good_plan()
    planned.required_data = ["paper-42158852.pdf"]
    planned.steps[0].inputs = ["paper-42158852.pdf"]
    planned.steps[0].methods = ["Enumerate browser downloads and import the PDF"]

    report = lint_plan(acquisition_task, planned)

    assert "unknown_plan_input_artifact" not in {item.code for item in report.findings}


def test_report_validator_requires_known_evidence():
    report = article_report(
        title="Test report",
        executive_summary="Test summary",
        methods=["retrieval"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Claim",
                claim_type="literature_supported",
                evidence_refs=["missing"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[],
        narrative="x",
    )
    validation = validate_report(report)
    assert not validation.passed
    assert "unknown_evidence_ref" in {item.code for item in validation.findings}


def test_report_validator_accepts_linked_claim():
    report = article_report(
        title="Test report",
        executive_summary="Test summary",
        methods=["retrieval"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Claim",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Source",
                url="https://example.com/source",
                source_type="web_page",
                retrieved_at="2026-07-13T00:00:00Z",
                supporting_passage="The source supports the claim.",
            )
        ],
        narrative="x",
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=["https://example.com/source"],
        retrieval_dates=["2026-07-13"],
    )
    assert validate_report(report, evidence).passed


def test_report_validator_requires_exact_inline_literature_citations():
    source = SourceRecord(
        source_id="s1",
        title="Acquired source",
        url="https://example.com/source",
        source_type="web_page",
        retrieved_at="2026-07-13T00:00:00Z",
        supporting_passage="The study reported the bounded finding.",
    )
    claim = ClaimRecord(
        claim_id="c1",
        text="The study reported the bounded finding.",
        claim_type="literature_supported",
        evidence_refs=["s1"],
        status=EvidenceStatus.SUPPORTED,
    )
    report = article_report(
        introduction="The study reported the bounded finding.",
        claims=[claim],
        sources=[source],
        inline_citations=[
            InlineCitation(
                citation_id="bounded-finding",
                section="introduction",
                anchor_text="The study reported the bounded finding.",
                source_ids=["s1"],
                claim_ids=["c1"],
            )
        ],
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=["https://example.com/source"],
        retrieval_dates=["2026-07-13"],
    )

    assert validate_report(report, evidence, require_inline_citations=True).passed

    missing = validate_report(
        report.model_copy(update={"inline_citations": []}),
        evidence,
        require_inline_citations=True,
    )
    assert "literature_claim_missing_inline_citation" in {
        item.code for item in missing.findings
    }

    wrong_anchor = report.inline_citations[0].model_copy(
        update={"anchor_text": "A sentence absent from this section."}
    )
    invalid = validate_report(
        report.model_copy(update={"inline_citations": [wrong_anchor]}),
        evidence,
        require_inline_citations=True,
    )
    assert "inline_citation_anchor_not_unique" in {
        item.code for item in invalid.findings
    }


def test_report_validator_rejects_overlapping_inline_citation_anchors():
    report = article_report(
        introduction="A bounded literature finding supports the method.",
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="A bounded literature finding supports the method.",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Source",
                url="https://example.com/source",
                source_type="web_page",
                retrieved_at="2026-07-13T00:00:00Z",
                supporting_passage="A bounded literature finding supports the method.",
            )
        ],
        inline_citations=[
            InlineCitation(
                citation_id="whole-finding",
                section="introduction",
                anchor_text="A bounded literature finding supports the method.",
                source_ids=["s1"],
                claim_ids=["c1"],
            ),
            InlineCitation(
                citation_id="nested-finding",
                section="introduction",
                anchor_text="literature finding",
                source_ids=["s1"],
                claim_ids=["c1"],
            ),
        ],
    )

    validation = validate_report(report, require_inline_citations=True)

    assert "overlapping_inline_citation_anchors" in {
        item.code for item in validation.findings
    }


def test_report_validator_rejects_inline_citation_on_unrelated_sentence():
    source = SourceRecord(
        source_id="s1",
        title="Mammalian taxonomy",
        url="https://example.com/taxonomy",
        source_type="documentation",
        retrieved_at="2026-07-16T00:00:00Z",
        supporting_passage="Cats are mammals.",
    )
    report = article_report(
        introduction="The intervention improved overall survival.",
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Cats are mammals.",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[source],
        inline_citations=[
            InlineCitation(
                citation_id="misplaced-citation",
                section="introduction",
                anchor_text="The intervention improved overall survival.",
                source_ids=["s1"],
                claim_ids=["c1"],
            )
        ],
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=["https://example.com/taxonomy"],
        retrieval_dates=["2026-07-16"],
    )

    validation = validate_report(report, evidence, require_inline_citations=True)

    assert "inline_citation_claim_anchor_mismatch" in {
        item.code for item in validation.findings
    }


def test_inline_citation_correspondence_accepts_paraphrase_and_numeric_formatting():
    source = SourceRecord(
        source_id="s1",
        title="Bounded evidence",
        url="https://example.com/evidence",
        source_type="documentation",
        retrieved_at="2026-07-16T00:00:00Z",
        supporting_passage="The source contains the bounded findings.",
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=["https://example.com/evidence"],
        retrieval_dates=["2026-07-16"],
    )
    cases = (
        (
            "Mortality was lower in the treated cohort after adjustment.",
            "Adjusted mortality decreased in the treatment cohort.",
        ),
        (
            "Ten-year event incidence was 10.0%.",
            "At a decade, outcomes occurred in 10% of participants.",
        ),
    )

    for index, (anchor, claim_text) in enumerate(cases):
        claim = ClaimRecord(
            claim_id=f"c{index}",
            text=claim_text,
            claim_type="literature_supported",
            evidence_refs=["s1"],
            status=EvidenceStatus.SUPPORTED,
        )
        report = article_report(
            introduction=anchor,
            claims=[claim],
            sources=[source],
            inline_citations=[
                InlineCitation(
                    citation_id=f"valid-citation-{index}",
                    section="introduction",
                    anchor_text=anchor,
                    source_ids=["s1"],
                    claim_ids=[claim.claim_id],
                )
            ],
        )

        validation = validate_report(report, evidence, require_inline_citations=True)

        assert "inline_citation_claim_anchor_mismatch" not in {
            item.code for item in validation.findings
        }


def test_requested_pptx_must_exist_and_be_structurally_valid(tmp_path):
    report = article_report()
    missing = validate_report(report, required_output_extensions=(".pptx",))
    assert "requested_output_artifact_missing" in {
        item.code for item in missing.findings
    }

    presentation = tmp_path / "presentation.pptx"
    presentation.write_bytes(b"not-an-office-document")
    artifact = ArtifactRef(
        path=str(presentation),
        sha256=sha256_file(presentation),
        description="sandbox-generated analysis artifact",
    )
    malformed = validate_report(
        report,
        computation=ComputationEvidence(successful_calls=1, artifacts=[artifact]),
        required_output_extensions=(".pptx",),
    )
    assert "requested_pptx_invalid" in {item.code for item in malformed.findings}

    with zipfile.ZipFile(presentation, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types/>")
        archive.writestr("ppt/presentation.xml", "<p:presentation/>")
    artifact = artifact.model_copy(update={"sha256": sha256_file(presentation)})
    preview = tmp_path / "output" / "visual-review" / "presentation-slide-1.png"
    preview.parent.mkdir(parents=True)
    Image.new("RGB", (640, 360), color="white").save(preview)
    preview_artifact = ArtifactRef(
        path=str(preview),
        sha256=sha256_file(preview),
        description="sandbox-generated analysis artifact",
    )
    valid = validate_report(
        report,
        computation=ComputationEvidence(
            successful_calls=1, artifacts=[artifact, preview_artifact]
        ),
        required_output_extensions=(".pptx",),
    )
    assert valid.passed, valid.findings


def test_report_validator_rejects_tautological_scientific_equation():
    report = article_report(
        methods=["Assume σ₁² = σ₁² before selecting the method."],
    )

    validation = validate_report(report)

    assert not validation.passed
    assert "tautological_equation" in {finding.code for finding in validation.findings}


def test_report_validator_does_not_confuse_distinct_equation_operands():
    report = article_report(
        methods=["The equal-variance boundary is σ₁² = σ₂²."],
    )

    assert validate_report(report).passed


def test_report_validator_requires_claim_for_method_recommendation():
    report = article_report(
        conclusions=(
            "Welch's test should be prioritized over Student's test for this use."
        ),
    )

    validation = validate_report(report)

    assert not validation.passed
    assert "methodological_recommendation_missing_claim" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_unscoped_web_only_method_recommendation():
    source = SourceRecord(
        source_id="s1",
        title="Secondary method summary",
        url="https://example.com/method-summary",
        source_type="web_page",
        retrieved_at="2026-07-15T00:00:00Z",
        supporting_passage="The page recommends one test over another.",
    )
    report = article_report(
        conclusions="Welch's test should be used as the default method.",
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Welch's test should be used as the default method.",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[source],
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=[str(source.url)],
        retrieval_dates=["2026-07-15"],
    )

    validation = validate_report(report, evidence)

    codes = {finding.code for finding in validation.findings}
    assert "methodological_recommendation_unscoped" in codes
    assert "methodological_recommendation_not_locally_grounded" in codes


def test_method_recommendation_scope_rejects_universal_and_accepts_bounded_conditions():
    local_source = SourceRecord(
        source_id="s1",
        title="Local method study",
        source_type="web_page",
        retrieved_at="2026-07-15T00:00:00Z",
        supporting_passage="The simulation evaluates bounded conditions.",
        artifact_path="/run/references/method.md",
    )

    def recommendation(text):
        return article_report(
            conclusions=text,
            claims=[
                ClaimRecord(
                    claim_id="c1",
                    text=text,
                    claim_type="literature_supported",
                    evidence_refs=["s1"],
                    status=EvidenceStatus.SUPPORTED,
                )
            ],
            sources=[local_source],
        )

    universal = recommendation(
        "Welch's test should be preferred for all distributions."
    )
    bounded = recommendation(
        "Within the simulated conditions at sample sizes 20 to 200, Welch's "
        "test should be preferred."
    )
    local_analysis = recommendation(
        "Bonferroni correction should be used for the six pairwise comparisons "
        "in this analysis."
    )
    bounded_quantifier = recommendation(
        "For all evaluated conditions in this simulation, Welch's test should "
        "be preferred."
    )
    newly_covered_wordings = [
        recommendation("We recommend Welch's t-test as the default."),
        recommendation("Welch's t-test is preferable in practice."),
        recommendation("Student's t-test should be replaced by Welch's t-test."),
    ]

    assert "methodological_recommendation_unscoped" in {
        finding.code for finding in validate_report(universal).findings
    }
    for report in (bounded, local_analysis, bounded_quantifier):
        assert "methodological_recommendation_unscoped" not in {
            finding.code for finding in validate_report(report).findings
        }
    for report in newly_covered_wordings:
        assert "methodological_recommendation_unscoped" in {
            finding.code for finding in validate_report(report).findings
        }


def test_report_validator_requires_local_verification_for_test_equivalence():
    source = SourceRecord(
        source_id="s1",
        title="Secondary comparison summary",
        url="https://example.com/comparison",
        source_type="web_page",
        retrieved_at="2026-07-15T00:00:00Z",
        supporting_passage="The page compares two tests.",
    )
    report = article_report(
        results="The two tests produce statistically equivalent p-values.",
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="The two tests produce statistically equivalent p-values.",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[source],
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=[str(source.url)],
        retrieval_dates=["2026-07-15"],
    )

    validation = validate_report(report, evidence)

    assert "procedure_equivalence_not_verified" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_paper_citation_without_local_article_record():
    report = article_report(
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="A paper-backed claim",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Unacquired paper",
                url="https://example.com/paper",
                source_type="primary_study",
                retrieved_at="2026-07-13T00:00:00Z",
                supporting_passage="A browser result is not a stored article record.",
            )
        ],
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        urls=["https://example.com/paper"],
        retrieval_dates=["2026-07-13"],
    )

    validation = validate_report(report, evidence)

    assert not validation.passed
    assert "literature_source_not_locally_acquired" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_doi_article_disguised_as_web_page():
    report = article_report(
        sources=[
            SourceRecord(
                source_id="s1",
                title="A DOI-bearing research article",
                url="https://example.com/articles/research",
                doi="10.1234/example.1",
                source_type="web_page",
                retrieved_at="2026-07-15T00:00:00Z",
                supporting_passage="A methodological result is reported.",
            )
        ]
    )

    validation = validate_report(report)

    assert not validation.passed
    assert "doi_source_misclassified_as_web_page" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_source_shaped_json_without_retrieval():
    report = article_report(
        title="Test report",
        executive_summary="Test summary",
        methods=["retrieval"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Claim",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Source",
                url="https://example.com/invented",
                source_type="web_page",
                retrieved_at="2026-07-13T00:00:00Z",
                supporting_passage="Claim-shaped text is not retrieval evidence.",
            )
        ],
        narrative="x",
    )
    validation = validate_report(report, RetrievalEvidence())
    assert not validation.passed
    assert {finding.code for finding in validation.findings} >= {
        "supported_without_retrieval",
        "source_url_not_retrieved",
    }


def test_report_validator_records_absent_retrieval_without_crashing():
    report = article_report(
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Claim",
                claim_type="literature_supported",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Source",
                url="https://example.com/source",
                source_type="web_page",
                retrieved_at="2026-07-13T00:00:00Z",
                supporting_passage="A source passage.",
            )
        ],
    )

    validation = validate_report(report)

    assert not validation.passed
    assert "supported_without_retrieval" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_accepts_url_and_date_seen_in_tool_output():
    report = article_report(
        title="Test report",
        executive_summary="Test summary",
        methods=["retrieval"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Claim",
                claim_type="observed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Source",
                url="https://example.com/source",
                source_type="documentation",
                retrieved_at="2026-07-13T15:00:00Z",
                supporting_passage="The source supports the claim.",
            )
        ],
        narrative="x",
    )
    evidence = RetrievalEvidence(
        successful_calls=1,
        tools=["query-docs"],
        urls=["https://example.com/source"],
        retrieval_dates=["2026-07-13"],
    )
    assert validate_report(report, evidence).passed


def test_report_validator_rejects_false_provenance_deferral():
    report = article_report(
        title="Test report",
        executive_summary="Test summary",
        methods=["retrieval"],
        claims=[],
        sources=[],
        limitations=["SHA-256 hashes for the provenance manifest are deferred."],
        narrative="x",
    )
    validation = validate_report(report)
    assert not validation.passed
    assert "false_provenance_deferral" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_accepts_generated_computation_artifact(tmp_path):
    output = tmp_path / "summary.csv"
    output.write_text("group,mean\nA,2.0\n", encoding="utf-8")
    report = article_report(
        title="Computed result",
        executive_summary="The sandbox computed a group mean.",
        methods=["Python aggregation"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="Group A has mean 2.0.",
                claim_type="computed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Generated summary table",
                artifact_path=str(output),
                source_type="dataset",
                retrieved_at="2026-07-13T15:00:00Z",
                supporting_passage="The generated row reports mean 2.0.",
            )
        ],
        narrative="The value was computed from the input table.",
    )
    artifact = ArtifactRef(path=str(output), sha256="abc", description="output")
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="abc",
                started_at="2026-07-13T14:59:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    assert validate_report(report, computation=computation).passed


def test_report_validator_rejects_unrecorded_computation_artifact(tmp_path):
    report = article_report(
        title="Computed result",
        executive_summary="A value was claimed.",
        methods=["Python"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="The value is 2.0.",
                claim_type="computed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Invented output",
                artifact_path=str(tmp_path / "never-generated.csv"),
                source_type="dataset",
                retrieved_at="2026-07-13T15:00:00Z",
                supporting_passage="Unverified output.",
            )
        ],
        narrative="No matching successful execution exists.",
    )
    validation = validate_report(report, computation=ComputationEvidence())
    assert not validation.passed
    assert {finding.code for finding in validation.findings} >= {
        "supported_without_computation",
        "source_artifact_not_generated",
    }


def test_report_validator_rejects_analysis_artifact_overreach(tmp_path):
    output = tmp_path / "diagnostics.json"
    output.write_text('{"normality_p": 0.03}\n', encoding="utf-8")
    artifact = ArtifactRef(
        path=str(output),
        sha256="abc",
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="abc",
                started_at="2026-07-13T14:59:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    report = article_report(
        title="Overreach",
        executive_summary="Diagnostics were computed.",
        methods=["Welch test"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="The protocol was locked prior to outcome inspection.",
                claim_type="computed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            ),
            ClaimRecord(
                claim_id="c2",
                text="The Welch test is robust to this assumption violation.",
                claim_type="inference",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            ),
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Generated diagnostics",
                artifact_path=str(output),
                source_type="dataset",
                retrieved_at="2026-07-13T15:00:00Z",
                supporting_passage="The normality p-value is 0.03.",
            )
        ],
        narrative="The artifact does not establish timing or method robustness.",
    )

    validation = validate_report(report, computation=computation)

    assert {finding.code for finding in validation.findings} >= {
        "protocol_timing_not_computed",
        "methodological_generalization_without_source",
    }


def test_report_validator_accepts_controller_protocol_timing_evidence(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"locked_at":"2026-07-13T14:00:00Z","status":"supported"}\n',
        encoding="utf-8",
    )
    artifact = ArtifactRef(
        path=str(protocol),
        sha256="abc",
        description="controller protocol lock written before research execution",
    )
    report = article_report(
        title="Controller provenance",
        executive_summary="The controller recorded the method lock.",
        methods=["Protocol lock before research execution"],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="The controller locked the protocol before research execution.",
                claim_type="observed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Controller protocol",
                artifact_path=str(protocol),
                source_type="documentation",
                retrieved_at="2026-07-13T14:00:00Z",
                supporting_passage="The controller recorded locked_at before research.",
            )
        ],
        narrative="The protocol artifact is controller evidence, not model output.",
    )

    validation = validate_report(
        report,
        controller_artifacts=(artifact,),
        controller_dates=("2026-07-13",),
    )

    assert validation.passed


def test_report_validator_rejects_protocol_ai_and_robustness_overclaims(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        '{"locked_at":"2026-07-13T14:00:00Z","status":"supported"}\n',
        encoding="utf-8",
    )
    artifact = ArtifactRef(
        path=str(protocol),
        sha256="abc",
        description="controller protocol lock written before research execution",
    )
    report = article_report(
        title="Overstated workflow report",
        executive_summary="A descriptive comparison.",
        methods=[
            "The protocol was locked prior to data inspection.",
            "AI was used only for report drafting and artifact registration.",
        ],
        claims=[
            ClaimRecord(
                claim_id="c1",
                text="The protocol was locked prior to data inspection.",
                claim_type="observed",
                evidence_refs=["s1"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Controller protocol",
                artifact_path=str(protocol),
                source_type="documentation",
                retrieved_at="2026-07-13T14:00:00Z",
                supporting_passage="The controller locked the protocol before research.",
            )
        ],
        narrative="The input inventory preceded planning.",
    )
    report.discussion = (
        "The balanced design mitigates Type I error inflation after non-normality."
    )
    report.conclusions = "The statistical result is robust and verified."

    validation = validate_report(
        report,
        controller_artifacts=(artifact,),
        controller_dates=("2026-07-13",),
    )

    assert {finding.code for finding in validation.findings} >= {
        "protocol_timing_overstates_input_blinding",
        "ai_role_understated",
        "balanced_design_assumption_reassurance",
        "unqualified_result_robustness",
    }


def test_report_validator_rejects_untested_normality_reassurance_and_robust_contrast():
    report = article_report(
        limitations=[
            "Normality assumptions were not formally tested, though the balanced "
            "sample size and symmetric distributions mitigate concern."
        ],
        conclusions=(
            "These findings establish a robust quantitative contrast within the "
            "supplied dataset."
        ),
    )

    validation = validate_report(report)

    assert {finding.code for finding in validation.findings} >= {
        "balanced_design_assumption_reassurance",
        "unqualified_result_robustness",
    }


def test_report_validator_rejects_unqualified_robust_association():
    report = article_report(
        conclusions=(
            "The prespecified analysis identifies a robust association between "
            "group assignment and change scores."
        )
    )

    validation = validate_report(report)

    assert "unqualified_result_robustness" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_sandbox_plan_as_protocol_timing_evidence(tmp_path):
    plan = tmp_path / "locked_analysis_plan.json"
    plan.write_text('{"status":"locked"}\n', encoding="utf-8")
    artifact = ArtifactRef(
        path=str(plan),
        sha256="abc",
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="abc",
                started_at="2026-07-13T15:00:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    report = article_report(
        title="Invalid timing provenance",
        executive_summary="The analysis completed.",
        methods=["The protocol was locked before outcome inspection."],
        claims=[],
        sources=[
            SourceRecord(
                source_id="s1",
                title="Locked analysis plan",
                artifact_path=str(plan),
                source_type="documentation",
                retrieved_at="2026-07-13T15:00:00Z",
                supporting_passage="The sandbox wrote a plan during analysis.",
            )
        ],
        narrative="The sandbox plan cannot prove controller timing.",
    )

    validation = validate_report(report, computation=computation)

    assert "protocol_timing_without_controller_artifact" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_unknown_design_and_domain_overreach():
    report = article_report(
        title="Unsupported framing",
        executive_summary="A descriptive comparison.",
        methods=["Welch test"],
        claims=[],
        sources=[],
        narrative="The outcome domain and units are unspecified.",
    )
    report.introduction = (
        "Changes following an intervention were compared. The study design is "
        "unspecified."
    )
    report.discussion = (
        "Without domain and scale context, clinical importance cannot be assessed."
    )

    validation = validate_report(report)

    assert {finding.code for finding in validation.findings} >= {
        "unspecified_design_intervention_framing",
        "unknown_domain_clinical_framing",
    }


def test_report_validator_requires_each_locked_computation_language(tmp_path):
    output = tmp_path / "summary.csv"
    output.write_text("estimate\n5\n", encoding="utf-8")
    artifact = ArtifactRef(
        path=str(output),
        sha256="abc",
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="abc",
                started_at="2026-07-13T14:59:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    report = article_report(
        title="Cross-check",
        executive_summary="Only Python succeeded.",
        methods=["Python"],
        claims=[],
        sources=[],
        narrative="R is still required.",
    )
    validation = validate_report(
        report,
        computation=computation,
        required_languages=("python", "r"),
    )
    assert not validation.passed
    assert "required_computation_language_missing" in {
        finding.code for finding in validation.findings
    }


def test_report_validator_rejects_nonfinite_generated_json(tmp_path):
    output = tmp_path / "result.json"
    output.write_text('{"estimate": NaN}\n', encoding="utf-8")
    artifact = ArtifactRef(
        path=str(output),
        sha256="abc",
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="abc",
                started_at="2026-07-13T14:59:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    report = article_report(
        title="Strict JSON",
        executive_summary="The artifact is machine readable.",
        methods=["Analysis method"],
        claims=[],
        sources=[],
        narrative="Strict serialization is required.",
    )

    invalid = validate_report(report, computation=computation)
    assert "invalid_generated_json" in {finding.code for finding in invalid.findings}

    output.write_text('{"estimate": null}\n', encoding="utf-8")
    assert validate_report(report, computation=computation).passed


def test_report_validator_requires_passing_reconciliation_artifact(tmp_path):
    report = article_report(
        title="Cross-check",
        executive_summary="Implementations compared.",
        methods=["Analysis method"],
        claims=[],
        sources=[],
        narrative="Comparison",
    )
    python_path = tmp_path / "python.json"
    r_path = tmp_path / "r.json"
    python_path.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    r_path.write_text('{"primary":{"point_estimate":5.1}}\n', encoding="utf-8")
    artifact_path = tmp_path / "reconciliation-original.json"
    artifact_path.write_text(
        json.dumps(
            reconciliation_document(python_path, r_path, python_value=5.0, r_value=5.1)
        ),
        encoding="utf-8",
    )
    python_artifact = ArtifactRef(
        path=str(python_path),
        sha256=sha256_file(python_path),
        description="sandbox-generated analysis artifact",
    )
    r_artifact = ArtifactRef(
        path=str(r_path),
        sha256=sha256_file(r_path),
        description="sandbox-generated analysis artifact",
    )
    artifact = ArtifactRef(
        path=str(artifact_path),
        sha256=sha256_file(artifact_path),
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=2,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-13T14:59:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[python_artifact, artifact],
            ),
            ComputationRecord(
                execution_id="exec-002",
                language="r",
                code_sha256="b" * 64,
                started_at="2026-07-13T15:00:00Z",
                duration_seconds=0.1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout-r.txt"),
                stderr_path=str(tmp_path / "stderr-r.txt"),
                artifacts=[r_artifact],
            ),
        ],
        artifacts=[python_artifact, r_artifact, artifact],
    )

    failed = validate_report(
        report,
        computation=computation,
        require_reconciliation=True,
    )
    assert "cross_language_reconciliation_failed" in {
        finding.code for finding in failed.findings
    }

    corrected_r_path = tmp_path / "r-corrected.json"
    corrected_r_path.write_text(
        '{"primary":{"point_estimate":5.0}}\n', encoding="utf-8"
    )
    corrected_path = tmp_path / "reconciliation-corrected.json"
    corrected_path.write_text(
        json.dumps(
            reconciliation_document(
                python_path, corrected_r_path, python_value=5.0, r_value=5.0
            )
        ),
        encoding="utf-8",
    )
    corrected_r = ArtifactRef(
        path=str(corrected_r_path),
        sha256=sha256_file(corrected_r_path),
        description="sandbox-generated analysis artifact",
    )
    corrected = ArtifactRef(
        path=str(corrected_path),
        sha256=sha256_file(corrected_path),
        description="sandbox-generated analysis artifact",
    )
    corrected_record = ComputationRecord(
        execution_id="exec-003",
        language="r",
        code_sha256="c" * 64,
        started_at="2026-07-13T15:01:00Z",
        duration_seconds=0.1,
        exit_code=0,
        status="succeeded",
        stdout_path=str(tmp_path / "stdout-2.txt"),
        stderr_path=str(tmp_path / "stderr-2.txt"),
        artifacts=[corrected_r, corrected],
    )
    corrected_computation = computation.model_copy(
        update={
            "records": [*computation.records, corrected_record],
            "artifacts": [*computation.artifacts, corrected_r, corrected],
        }
    )
    superseded = validate_report(
        report,
        computation=corrected_computation,
        require_reconciliation=True,
    )
    assert superseded.passed
    assert any(
        finding.code == "superseded_reconciliation_failure" and not finding.blocking
        for finding in superseded.findings
    )


def test_reconciliation_rejects_model_authored_forged_pass(tmp_path):
    python_path = tmp_path / "python.json"
    r_path = tmp_path / "r.json"
    python_path.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    r_path.write_text('{"primary":{"point_estimate":5.1}}\n', encoding="utf-8")
    payload = reconciliation_document(
        python_path, r_path, python_value=5.0, r_value=5.1
    )
    payload["all_pass"] = True
    payload["comparisons"][0]["passed"] = True
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(json.dumps(payload), encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (python_path, r_path, reconciliation_path)
    ]
    records = [
        ComputationRecord(
            execution_id=f"exec-{index}",
            language=language,
            code_sha256=str(index) * 64,
            started_at="2026-07-13T15:00:00Z",
            duration_seconds=0.1,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / f"stdout-{index}.txt"),
            stderr_path=str(tmp_path / f"stderr-{index}.txt"),
            artifacts=record_artifacts,
        )
        for index, (language, record_artifacts) in enumerate(
            (("python", [artifacts[0], artifacts[2]]), ("r", [artifacts[1]])),
            start=1,
        )
    ]
    validation = validate_report(
        article_report(),
        computation=ComputationEvidence(records=records, artifacts=artifacts),
        require_reconciliation=True,
    )
    assert "reconciliation_artifact_invalid" in {
        finding.code for finding in validation.findings
    }


def test_reconciliation_rejects_conflicting_shared_boolean_diagnostic(tmp_path):
    python_path = tmp_path / "python.json"
    r_path = tmp_path / "r.json"
    python_path.write_text(
        '{"primary":{"point_estimate":5.0},"diagnostics":{"missing_values":false}}\n',
        encoding="utf-8",
    )
    r_path.write_text(
        '{"primary":{"point_estimate":5.0},"diagnostics":{"missing_values":true}}\n',
        encoding="utf-8",
    )
    payload = reconciliation_document(
        python_path, r_path, python_value=5.0, r_value=5.0
    )
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(json.dumps(payload), encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (python_path, r_path, reconciliation_path)
    ]
    records = [
        ComputationRecord(
            execution_id=f"exec-{index}",
            language=language,
            code_sha256=str(index) * 64,
            started_at="2026-07-13T15:00:00Z",
            duration_seconds=0.1,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / f"stdout-{index}.txt"),
            stderr_path=str(tmp_path / f"stderr-{index}.txt"),
            artifacts=record_artifacts,
        )
        for index, (language, record_artifacts) in enumerate(
            (("python", [artifacts[0], artifacts[2]]), ("r", [artifacts[1]])),
            start=1,
        )
    ]

    validation = validate_report(
        article_report(),
        computation=ComputationEvidence(records=records, artifacts=artifacts),
        require_reconciliation=True,
    )

    assert "reconciliation_artifact_invalid" in {
        finding.code for finding in validation.findings
    }


def test_reconciliation_accepts_multiple_bound_artifacts_per_language(tmp_path):
    source_paths = {}
    for language in ("python", "r"):
        for index, (json_path, value) in enumerate(
            (("primary.estimate", 5.0), ("secondary.estimate", 2.0)), start=1
        ):
            section, field = json_path.split(".")
            path = tmp_path / f"{language}-{index}.json"
            path.write_text(
                json.dumps(
                    {section: {field: value}, "diagnostics": {"missing": False}}
                ),
                encoding="utf-8",
            )
            source_paths[(language, index)] = path

    comparisons = []
    for index, metric in enumerate(("primary", "secondary"), start=1):
        comparisons.append(
            {
                "metric": metric,
                "python": {
                    "language": "python",
                    "artifact_sha256": sha256_file(source_paths[("python", index)]),
                    "json_path": f"{metric}.estimate",
                    "value": 5.0 if index == 1 else 2.0,
                },
                "r": {
                    "language": "r",
                    "artifact_sha256": sha256_file(source_paths[("r", index)]),
                    "json_path": f"{metric}.estimate",
                    "value": 5.0 if index == 1 else 2.0,
                },
                "absolute_difference": 0.0,
                "tolerance": 1e-6,
                "passed": True,
            }
        )
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(
        json.dumps({"all_pass": True, "comparisons": comparisons}), encoding="utf-8"
    )
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (*source_paths.values(), reconciliation_path)
    ]
    records = [
        ComputationRecord(
            execution_id=f"exec-{index}",
            language=language,
            code_sha256=str(index) * 64,
            started_at="2026-07-13T15:00:00Z",
            duration_seconds=0.1,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / f"stdout-{index}.txt"),
            stderr_path=str(tmp_path / f"stderr-{index}.txt"),
            artifacts=[artifacts[index - 1]],
        )
        for index, language in enumerate(("python", "python", "r", "r"), start=1)
    ]
    records[0].artifacts.append(artifacts[-1])
    computation = ComputationEvidence(records=records, artifacts=artifacts)

    assert reconciliation_verdict(reconciliation_path, computation) is True


def test_reconciliation_allows_method_dependent_boolean_diagnostics(tmp_path):
    python_path = tmp_path / "python.json"
    r_path = tmp_path / "r.json"
    python_path.write_text(
        '{"primary":{"point_estimate":5.0},"diagnostics":{"normality_passed":false}}\n',
        encoding="utf-8",
    )
    r_path.write_text(
        '{"primary":{"point_estimate":5.0},"diagnostics":{"normality_passed":true}}\n',
        encoding="utf-8",
    )
    payload = reconciliation_document(
        python_path, r_path, python_value=5.0, r_value=5.0
    )
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(json.dumps(payload), encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (python_path, r_path, reconciliation_path)
    ]
    records = [
        ComputationRecord(
            execution_id=f"exec-{index}",
            language=language,
            code_sha256=str(index) * 64,
            started_at="2026-07-13T15:00:00Z",
            duration_seconds=0.1,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / f"stdout-{index}.txt"),
            stderr_path=str(tmp_path / f"stderr-{index}.txt"),
            artifacts=record_artifacts,
        )
        for index, (language, record_artifacts) in enumerate(
            (("python", [artifacts[0], artifacts[2]]), ("r", [artifacts[1]])),
            start=1,
        )
    ]

    assert (
        reconciliation_verdict(
            reconciliation_path,
            ComputationEvidence(records=records, artifacts=artifacts),
        )
        is True
    )


def test_cross_language_claim_must_cite_reconciliation_artifact(tmp_path):
    python_path = tmp_path / "python_estimate.json"
    python_path.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    r_path = tmp_path / "r_verification.json"
    r_path.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(
        json.dumps(
            reconciliation_document(python_path, r_path, python_value=5.0, r_value=5.0)
        ),
        encoding="utf-8",
    )
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (reconciliation_path, python_path, r_path)
    ]
    computation = ComputationEvidence(
        successful_calls=2,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="c" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=artifacts[:2],
            ),
            ComputationRecord(
                execution_id="exec-002",
                language="r",
                code_sha256="d" * 64,
                started_at="2026-07-15T00:01:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout-r.txt"),
                stderr_path=str(tmp_path / "stderr-r.txt"),
                artifacts=[artifacts[2]],
            ),
        ],
        artifacts=artifacts,
    )
    claim = ClaimRecord(
        claim_id="c-r",
        text=(
            "Independent R computation reproduces the Python estimate exactly "
            "with absolute difference 0.0."
        ),
        claim_type="computed",
        evidence_refs=["s-r"],
        status=EvidenceStatus.SUPPORTED,
    )
    report = article_report(
        claims=[claim],
        sources=[
            SourceRecord(
                source_id="s-r",
                title="R verification",
                artifact_path=str(r_path),
                source_type="other",
                retrieved_at="2026-07-15T00:00:00Z",
                supporting_passage="R point estimate 5.0.",
            )
        ],
    )

    bad = validate_report(report, computation=computation, require_reconciliation=True)

    assert "cross_language_claim_missing_reconciliation_source" in {
        finding.code for finding in bad.findings
    }
    corrected = report.model_copy(
        update={
            "sources": [
                report.sources[0].model_copy(
                    update={"artifact_path": str(reconciliation_path)}
                )
            ]
        }
    )
    good = validate_report(
        corrected, computation=computation, require_reconciliation=True
    )
    assert good.passed
    assert "cross_language_claim_missing_reconciliation_source" not in {
        finding.code for finding in good.findings
    }


def test_computed_diagnostic_must_exist_in_cited_artifact(tmp_path):
    result_path = tmp_path / "analysis.json"
    result_path.write_text(
        '{"primary":{"point_estimate":5.0,"p_value":2.3e-15}}\n',
        encoding="utf-8",
    )
    artifact = ArtifactRef(
        path=str(result_path),
        sha256=sha256_file(result_path),
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )
    report = article_report(
        claims=[
            ClaimRecord(
                claim_id="diagnostics",
                text="Shapiro-Wilk p = 0.117 and Levene's p = 1.00.",
                claim_type="computed",
                evidence_refs=["analysis"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="analysis",
                title="Analysis results",
                artifact_path=str(result_path),
                source_type="other",
                retrieved_at="2026-07-15T00:00:00Z",
                supporting_passage="Machine-readable analysis result.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "computed_diagnostic_not_in_artifact" in {
        finding.code for finding in validation.findings
    }


def test_t_df_and_p_value_must_be_arithmetically_consistent(tmp_path):
    result_path = tmp_path / "inferential.json"
    result_path.write_text(
        json.dumps(
            {
                "primary": {
                    "t_statistic": 12.809215162980795,
                    "degrees_freedom": 76,
                    "p_value": 2.3070327275079796e-15,
                }
            }
        ),
        encoding="utf-8",
    )
    artifact = ArtifactRef(
        path=str(result_path),
        sha256=sha256_file(result_path),
        description="sandbox-generated analysis artifact",
    )
    record = ComputationRecord(
        execution_id="exec-001",
        language="python",
        code_sha256="a" * 64,
        started_at="2026-07-15T00:00:00Z",
        duration_seconds=1,
        exit_code=0,
        status="succeeded",
        stdout_path=str(tmp_path / "stdout.txt"),
        stderr_path=str(tmp_path / "stderr.txt"),
        artifacts=[artifact],
    )

    invalid = validate_report(
        article_report(),
        computation=ComputationEvidence(records=[record], artifacts=[artifact]),
    )
    assert "inferential_statistic_inconsistent" in {
        finding.code for finding in invalid.findings
    }

    result_path.write_text(
        json.dumps(
            {
                "primary": {
                    "t_statistic": "12.809215162980795",
                    "degrees_freedom": "38",
                    "p_value": "2.3070327275079796e-15",
                }
            }
        ),
        encoding="utf-8",
    )
    string_bypass = validate_report(
        article_report(),
        computation=ComputationEvidence(records=[record], artifacts=[artifact]),
    )
    assert "inferential_statistic_inconsistent" in {
        finding.code for finding in string_bypass.findings
    }

    result_path.write_text(
        json.dumps(
            {
                "primary": {
                    "t_statistic": 12.809215162980795,
                    "degrees_freedom": 38,
                    "p_value": 2.3070327275079796e-15,
                }
            }
        ),
        encoding="utf-8",
    )
    valid = validate_report(
        article_report(),
        computation=ComputationEvidence(records=[record], artifacts=[artifact]),
    )
    assert "inferential_statistic_inconsistent" not in {
        finding.code for finding in valid.findings
    }

    mismatched_prose = validate_report(
        article_report(
            results=(
                "The Welch test gave t=10.90 (df ≈ 564.06), p=2.97e-13, "
                "with a mean difference of 5.0."
            )
        ),
        computation=ComputationEvidence(records=[record], artifacts=[artifact]),
    )
    assert "reported_degrees_of_freedom_not_in_machine_results" in {
        finding.code for finding in mismatched_prose.findings
    }

    matching_prose = validate_report(
        article_report(results="The Welch test gave t=10.90 (df = 38)."),
        computation=ComputationEvidence(records=[record], artifacts=[artifact]),
    )
    assert "reported_degrees_of_freedom_not_in_machine_results" not in {
        finding.code for finding in matching_prose.findings
    }


def test_inferential_check_rejects_impossible_two_sided_p_value_alias(tmp_path):
    findings = _inferential_consistency_findings(
        tmp_path / "python_results.json",
        {
            "welch_t_test": {
                "t_statistic": 10.897247358851683,
                "degrees_of_freedom": 564.0624999999999,
                "p_value_two_sided": 2.971749478841818e-13,
            }
        },
    )

    assert "inferential_statistic_inconsistent" in {
        finding.code for finding in findings
    }


def test_report_df_check_preserves_same_logical_output_from_each_language(tmp_path):
    artifacts = []
    records = []
    for language, degrees_freedom in (("python", 10.0), ("r", 20.0)):
        path = tmp_path / language / "exec-001" / "output" / "data" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps({"degrees_of_freedom": degrees_freedom}), encoding="utf-8"
        )
        artifact = ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        artifacts.append(artifact)
        records.append(
            ComputationRecord(
                execution_id=f"exec-{language}",
                language=language,
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / language / "stdout.txt"),
                stderr_path=str(tmp_path / language / "stderr.txt"),
                artifacts=[artifact],
            )
        )

    validation = validate_report(
        article_report(results="Python reported df = 10."),
        computation=ComputationEvidence(records=records, artifacts=artifacts),
    )

    assert "reported_degrees_of_freedom_not_in_machine_results" not in {
        finding.code for finding in validation.findings
    }


def test_corrected_logical_json_supersedes_bad_inferential_tuple(tmp_path):
    artifacts = []
    records = []
    for attempt, degrees_freedom in (("attempt-0", 564.0625), ("attempt-1", 38.0)):
        path = tmp_path / attempt / "exec-001" / "output" / "data" / "results.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "t_statistic": 10.897247358851683,
                    "degrees_of_freedom": degrees_freedom,
                    "p_value_two_sided": 2.971749478841818e-13,
                }
            ),
            encoding="utf-8",
        )
        artifact = ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        artifacts.append(artifact)
        records.append(
            ComputationRecord(
                execution_id=f"exec-{attempt}",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / attempt / "stdout.txt"),
                stderr_path=str(tmp_path / attempt / "stderr.txt"),
                artifacts=[artifact],
            )
        )

    validation = validate_report(
        article_report(),
        computation=ComputationEvidence(records=records, artifacts=artifacts),
    )

    assert validation.passed
    superseded = [
        finding
        for finding in validation.findings
        if finding.code == "superseded_inferential_statistic_inconsistent"
    ]
    assert len(superseded) == 1
    assert superseded[0].blocking is False


def test_inferential_check_supports_repo_aliases_and_signed_one_sided_tests(tmp_path):
    path = tmp_path / "inferential.json"
    invalid_alias_shape = _inferential_consistency_findings(
        path,
        {
            "welch_t_statistic": 12.809215162980795,
            "degrees_of_freedom": 76,
            "p_value": 2.3070327275079796e-15,
        },
    )
    assert {finding.code for finding in invalid_alias_shape} == {
        "inferential_statistic_inconsistent"
    }

    directional = _inferential_consistency_findings(
        path,
        {
            "welch_t_statistic": -2.0,
            "welch_df": 20,
            "welch_pvalue": 0.970367,
            "alternative": "greater",
        },
    )
    assert directional == []

    two_sided_r_spelling = _inferential_consistency_findings(
        path,
        {
            "t_statistic": 2.0,
            "df": 20,
            "p_value": 0.059265535,
            "alternative": "two.sided",
        },
    )
    assert two_sided_r_spelling == []


def test_inferential_check_rejects_conflicting_aliases_and_invalid_ranges(tmp_path):
    path = tmp_path / "inferential.json"
    conflicting = _inferential_consistency_findings(
        path,
        {
            "t_statistic": 2.0,
            "welch_t_statistic": 3.0,
            "degrees_freedom": 20,
            "p_value": 0.058,
        },
    )
    assert len(conflicting) == 1
    assert "Conflicting aliases" in conflicting[0].message

    invalid_range = _inferential_consistency_findings(
        path,
        {"t_statistic": 2.0, "df": 0, "p_value": 1.2},
    )
    assert len(invalid_range) == 1
    assert "valid ranges" in invalid_range[0].message

    unknown_alternative = _inferential_consistency_findings(
        path,
        {
            "t_statistic": 2.0,
            "df": 20,
            "p_value": 0.058,
            "alternative": "sometimes",
        },
    )
    assert len(unknown_alternative) == 1
    assert "alternative hypothesis" in unknown_alternative[0].message


def test_inferential_check_rejects_p_value_error_that_crosses_alpha(tmp_path):
    findings = _inferential_consistency_findings(
        tmp_path / "inferential.json",
        {
            "t_statistic": 2.06638,
            "df": 25,
            "p_value": 0.0502367,
            "alternative": "two-sided",
        },
    )

    assert {finding.code for finding in findings} == {
        "inferential_statistic_inconsistent"
    }


def test_inferential_check_fails_closed_without_scipy(tmp_path, monkeypatch):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "scipy.stats":
            raise ImportError("test-only missing SciPy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    findings = _inferential_consistency_findings(
        tmp_path / "inferential.json",
        {"t_statistic": 2.0, "df": 20, "p_value": 0.058},
    )

    assert {finding.code for finding in findings} == {
        "inferential_validator_unavailable"
    }


def test_guideline_claim_requires_guideline_source_type():
    report = article_report(
        claims=[
            ClaimRecord(
                claim_id="guideline",
                text="Reporting guidelines require confidence intervals.",
                claim_type="literature_supported",
                evidence_refs=["review"],
                status=EvidenceStatus.SUPPORTED,
            )
        ],
        sources=[
            SourceRecord(
                source_id="review",
                title="Unrelated systematic review",
                url="https://example.org/review",
                source_type="review",
                retrieved_at="2026-07-15T00:00:00Z",
                supporting_passage="A review of an intervention.",
            )
        ],
    )

    validation = validate_report(report)

    assert "literature_source_type_mismatch" in {
        finding.code for finding in validation.findings
    }


def test_sensitivity_and_diagnostic_nonrejection_are_not_proof():
    report = article_report(
        results=(
            "The adjusted estimate was similar, confirming robustness to baseline "
            "imbalance. Shapiro-Wilk and Levene tests were nonsignificant, so the "
            "normality and homoscedasticity assumptions were met."
        ),
        discussion=(
            "The covariates did not materially confound the estimate, validating "
            "the analytical pipeline and confirming algorithmic equivalence."
        ),
    )

    validation = validate_report(report)
    codes = {finding.code for finding in validation.findings}

    assert "sensitivity_analysis_overclaim" in codes
    assert "diagnostic_nonrejection_overclaim" in codes


def test_welch_does_not_excuse_detected_nonnormality():
    report = article_report(
        discussion=(
            "Normality diagnostics indicated mild departures from Gaussian "
            "assumptions, which the Welch procedure accommodates."
        ),
        limitations=[
            "The change scores were non-normal, though Welch's test remains applicable."
        ],
    )

    validation = validate_report(report)

    assert "welch_normality_overclaim" in {
        finding.code for finding in validation.findings
    }


def test_welch_variance_scope_and_nonnormality_limitation_are_valid():
    report = article_report(
        methods=["Welch's test was used to avoid assuming equal variances."],
        limitations=[
            "The change-score distributions departed from normality; no separate "
            "robustness analysis was performed."
        ],
    )

    validation = validate_report(report)

    assert "welch_normality_overclaim" not in {
        finding.code for finding in validation.findings
    }


def _display_computation(tmp_path, artifact_path):
    artifact = ArtifactRef(
        path=str(artifact_path),
        sha256=sha256_file(artifact_path),
        description="sandbox-generated analysis artifact",
    )
    return ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="e" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[artifact],
            )
        ],
        artifacts=[artifact],
    )


def test_group_table_cannot_place_overall_estimates_under_control(tmp_path):
    table = tmp_path / "output" / "tables" / "results.csv"
    table.parent.mkdir(parents=True)
    table.write_text(
        "Metric,Control,Treatment\n"
        "Mean,0.05,5.05\n"
        "Primary estimand (difference),5.00,\n"
        "Two-sided p-value,< 0.001,\n",
        encoding="utf-8",
    )
    report = article_report(
        results="Table 1 reports the group results.",
        displays=[
            ReportDisplay(
                display_id="group-table",
                kind="table",
                title="Group results",
                caption="Control, treatment, and overall estimates.",
                artifact_path=str(table),
            )
        ],
    )

    validation = validate_report(
        report, computation=_display_computation(tmp_path, table)
    )

    assert "table_ambiguous_overall_estimate_column" in {
        finding.code for finding in validation.findings
    }


def test_group_table_accepts_neutral_overall_estimate_column(tmp_path):
    table = tmp_path / "output" / "tables" / "results.csv"
    table.parent.mkdir(parents=True)
    table.write_text(
        "Metric,Control group,Treatment group,Estimate\n"
        "Mean,0.05,5.05,\n"
        "Primary estimand (difference),,,5.00\n"
        "Two-sided p-value,,,< 0.001\n",
        encoding="utf-8",
    )
    report = article_report(
        results="Table 1 reports the group results.",
        displays=[
            ReportDisplay(
                display_id="group-table",
                kind="table",
                title="Group results",
                caption="Group summaries and neutral overall estimates.",
                artifact_path=str(table),
            )
        ],
    )

    validation = validate_report(
        report, computation=_display_computation(tmp_path, table)
    )

    assert validation.passed
    assert "table_ambiguous_overall_estimate_column" not in {
        finding.code for finding in validation.findings
    }


def test_same_logical_display_key_supersedes_historical_version(tmp_path):
    old_root = (
        tmp_path / "computations" / "attempt-1" / "exec-001" / "output" / "tables"
    )
    new_root = (
        tmp_path / "computations" / "attempt-3" / "exec-001" / "output" / "tables"
    )
    old_root.mkdir(parents=True)
    new_root.mkdir(parents=True)
    raw_json = old_root / "primary_analysis_results.json"
    raw_table = old_root / "results.csv"
    clean_table = new_root / "results.csv"
    raw_json.write_text('{"estimate": 5.000000000000001}\n', encoding="utf-8")
    raw_table.write_text("Metric,Value\nEstimate,5.000000000000001\n", encoding="utf-8")
    clean_table.write_text(
        "Metric,Control group,Treatment group,Estimate\n"
        "Mean change,0.05,5.05,\n"
        "Primary difference,,,5.00\n",
        encoding="utf-8",
    )
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (raw_json, raw_table, clean_table)
    ]
    computation = ComputationEvidence(
        successful_calls=2,
        records=[
            ComputationRecord(
                execution_id=f"exec-{index:03d}",
                language="python",
                code_sha256=str(index) * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / f"stdout-{index}.txt"),
                stderr_path=str(tmp_path / f"stderr-{index}.txt"),
                artifacts=(artifacts[:2] if index == 1 else artifacts[2:]),
            )
            for index in (1, 3)
        ],
        artifacts=artifacts,
    )
    report = article_report(
        results="Table 1 reports the corrected presentation generation.",
        displays=[
            ReportDisplay(
                display_id="corrected-table",
                kind="table",
                title="Corrected results",
                caption="Rounded group summaries and the primary difference.",
                artifact_path=str(clean_table),
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert validation.passed
    assert "unregistered_report_artifact" not in {
        finding.code for finding in validation.findings
    }
    assert any(
        finding.code == "non_display_artifact_in_reader_facing_folder"
        and not finding.blocking
        for finding in validation.findings
    )


def test_corrected_figure_supersedes_historical_render_warning(tmp_path):
    old_figure = (
        tmp_path
        / "computations"
        / "attempt-1"
        / "exec-001"
        / "output"
        / "figures"
        / "effect.png"
    )
    new_figure = (
        tmp_path
        / "computations"
        / "attempt-2"
        / "exec-001"
        / "output"
        / "figures"
        / "effect.png"
    )
    old_figure.parent.mkdir(parents=True)
    new_figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(old_figure, dpi=(300, 300))
    Image.new("RGB", (800, 600), color="white").save(new_figure, dpi=(300, 300))
    old_stderr = tmp_path / "old-stderr.txt"
    new_stderr = tmp_path / "new-stderr.txt"
    old_stderr.write_text(
        "UserWarning: This figure includes Axes that are not compatible with "
        "tight_layout, so results might be incorrect.\n",
        encoding="utf-8",
    )
    new_stderr.write_text("", encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (old_figure, new_figure)
    ]
    computation = ComputationEvidence(
        successful_calls=2,
        records=[
            ComputationRecord(
                execution_id="exec-old",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "old-stdout.txt"),
                stderr_path=str(old_stderr),
                artifacts=[artifacts[0]],
            ),
            ComputationRecord(
                execution_id="exec-new",
                language="python",
                code_sha256="b" * 64,
                started_at="2026-07-15T00:01:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "new-stdout.txt"),
                stderr_path=str(new_stderr),
                artifacts=[artifacts[1]],
            ),
        ],
        artifacts=artifacts,
    )
    report = article_report(
        results="Figure 1 shows the corrected effect display.",
        displays=[
            ReportDisplay(
                display_id="corrected-figure",
                kind="figure",
                title="Corrected effect",
                caption="Corrected effect display.",
                artifact_path=str(new_figure),
                alt_text="Corrected effect display.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_render_warning" not in {
        finding.code for finding in validation.findings
    }


def test_latest_presentation_attempt_still_requires_all_its_outputs_registered(
    tmp_path,
):
    root = tmp_path / "computations" / "attempt-3" / "exec-001" / "output" / "tables"
    root.mkdir(parents=True)
    displayed = root / "results.csv"
    omitted = root / "omitted.csv"
    displayed.write_text("Metric,Estimate\nDifference,5.00\n", encoding="utf-8")
    omitted.write_text("Metric,Estimate\nSensitivity,5.04\n", encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (displayed, omitted)
    ]
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=artifacts,
            )
        ],
        artifacts=artifacts,
    )
    report = article_report(
        results="Table 1 reports the primary estimate.",
        displays=[
            ReportDisplay(
                display_id="results",
                kind="table",
                title="Results",
                caption="Primary treatment difference.",
                artifact_path=str(displayed),
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "unregistered_report_artifact" in {
        finding.code for finding in validation.findings
    }


def test_new_current_run_output_does_not_hide_distinct_parent_output(tmp_path):
    parent = (
        tmp_path
        / "runs"
        / "parent"
        / "computations"
        / "attempt-9"
        / "exec-001"
        / "output"
        / "tables"
        / "parent.csv"
    )
    current = (
        tmp_path
        / "runs"
        / "current"
        / "computations"
        / "attempt-1"
        / "exec-001"
        / "output"
        / "tables"
        / "current.csv"
    )
    for path, value in ((parent, "4.90"), (current, "5.00")):
        path.parent.mkdir(parents=True)
        path.write_text(f"Metric,Estimate\nDifference,{value}\n", encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (parent, current)
    ]
    computation = ComputationEvidence(
        successful_calls=2,
        records=[
            ComputationRecord(
                execution_id=f"exec-{index:03d}",
                language="python",
                code_sha256=str(index) * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / f"stdout-{index}.txt"),
                stderr_path=str(tmp_path / f"stderr-{index}.txt"),
                artifacts=[artifact],
            )
            for index, artifact in enumerate(artifacts, start=1)
        ],
        artifacts=artifacts,
    )
    report = article_report(
        results="Table 1 reports the current-run estimate.",
        displays=[
            ReportDisplay(
                display_id="current-results",
                kind="table",
                title="Current results",
                caption="Current-run primary treatment difference.",
                artifact_path=str(current),
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert any(
        finding.code == "unregistered_report_artifact"
        and finding.location == str(parent)
        for finding in validation.findings
    )


def test_json_in_latest_tables_folder_is_nonblocking_path_hygiene(tmp_path):
    machine_json = (
        tmp_path
        / "computations"
        / "attempt-4"
        / "exec-001"
        / "output"
        / "tables"
        / "primary_results.json"
    )
    machine_json.parent.mkdir(parents=True)
    machine_json.write_text('{"estimate": 5.0}\n', encoding="utf-8")

    validation = validate_report(
        article_report(), computation=_display_computation(tmp_path, machine_json)
    )

    assert validation.passed
    assert any(
        finding.code == "non_display_artifact_in_reader_facing_folder"
        and finding.location == str(machine_json)
        and not finding.blocking
        for finding in validation.findings
    )
    assert "unregistered_report_artifact" not in {
        finding.code for finding in validation.findings
    }


def test_unversioned_reader_output_remains_mandatory_with_versioned_repairs(tmp_path):
    versioned = (
        tmp_path
        / "computations"
        / "attempt-3"
        / "exec-001"
        / "output"
        / "tables"
        / "results.csv"
    )
    legacy = tmp_path / "output" / "tables" / "legacy.csv"
    for path, label in ((versioned, "Difference"), (legacy, "Legacy")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Metric,Estimate\n{label},5.00\n", encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=sha256_file(path),
            description="sandbox-generated analysis artifact",
        )
        for path in (legacy, versioned)
    ]
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=artifacts,
            )
        ],
        artifacts=artifacts,
    )
    report = article_report(
        results="Table 1 reports the versioned result.",
        displays=[
            ReportDisplay(
                display_id="versioned-results",
                kind="table",
                title="Versioned results",
                caption="Primary treatment difference.",
                artifact_path=str(versioned),
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert any(
        finding.code == "unregistered_report_artifact"
        and finding.location == str(legacy)
        for finding in validation.findings
    )


def test_figure_caption_cannot_claim_absent_r_squared_annotation(tmp_path, monkeypatch):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {
            "available": True,
            "text": "Treatment effect 5.00 95% CI 4.21 to 5.79",
            "words": [{"text": "Treatment"}],
        },
    )
    report = article_report(
        results="Figure 1 reports the estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Treatment effect",
                caption="Panel annotations include adjusted R-squared.",
                artifact_path=str(figure),
                alt_text="A treatment effect with confidence interval.",
            )
        ],
    )

    validation = validate_report(
        report, computation=_display_computation(tmp_path, figure)
    )

    assert "figure_caption_claims_missing_annotation" in {
        finding.code for finding in validation.findings
    }


def test_figure_rejects_jitter_on_quantitative_scatter_axis(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.scatter([1] * len(values), values + jitter)\n", encoding="utf-8"
    )
    computation = _display_computation(tmp_path, figure)
    source_artifact = ArtifactRef(
        path=str(source),
        sha256=sha256_file(source),
        description="python analysis source",
    )
    computation.records[0].artifacts.append(source_artifact)
    computation.artifacts.append(source_artifact)
    report = article_report(
        results="Figure 1 shows the source observations.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Raw outcome values by group.",
                artifact_path=str(figure),
                alt_text="Raw observations separated by group.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_numeric_axis_jitter" in {
        finding.code for finding in validation.findings
    }


def test_figure_allows_jitter_on_categorical_scatter_axis(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text("ax.scatter(group_position + jitter, values)\n", encoding="utf-8")
    computation = _display_computation(tmp_path, figure)
    source_artifact = ArtifactRef(
        path=str(source),
        sha256=sha256_file(source),
        description="python analysis source",
    )
    computation.records[0].artifacts.append(source_artifact)
    computation.artifacts.append(source_artifact)
    report = article_report(
        results="Figure 1 shows the source observations.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Raw outcome values by group.",
                artifact_path=str(figure),
                alt_text="Raw observations separated by group.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_numeric_axis_jitter" not in {
        finding.code for finding in validation.findings
    }


def test_figure_rejects_effect_estimate_transposed_off_labeled_x_axis(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.set_xlabel('Mean Difference with 95% CI')\n"
        "ax.plot([0], [mean_diff])\n"
        "ax.errorbar([0], [mean_diff], xerr=[[0.9], [0.9]])\n",
        encoding="utf-8",
    )
    computation = _display_computation(tmp_path, figure)
    source_artifact = ArtifactRef(
        path=str(source),
        sha256=sha256_file(source),
        description="python analysis source",
    )
    computation.records[0].artifacts.append(source_artifact)
    report = article_report(
        results="Figure 1 shows the effect estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Mean difference with a 95% confidence interval.",
                artifact_path=str(figure),
                alt_text="Effect estimate and confidence interval.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_effect_axis_transposed" in {
        finding.code for finding in validation.findings
    }


def test_figure_allows_effect_estimate_on_labeled_x_axis(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.set_xlabel('Mean Difference with 95% CI')\n"
        "ax.plot([mean_diff], [0])\n"
        "ax.errorbar([mean_diff], [0], xerr=[[0.9], [0.9]])\n",
        encoding="utf-8",
    )
    computation = _display_computation(tmp_path, figure)
    source_artifact = ArtifactRef(
        path=str(source),
        sha256=sha256_file(source),
        description="python analysis source",
    )
    computation.records[0].artifacts.append(source_artifact)
    report = article_report(
        results="Figure 1 shows the effect estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Mean difference with a 95% confidence interval.",
                artifact_path=str(figure),
                alt_text="Effect estimate and confidence interval.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_effect_axis_transposed" not in {
        finding.code for finding in validation.findings
    }


def test_figure_rejects_live_transposed_interval_with_short_variable_name(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.set_xlabel('Treatment - Control\\nDifference in Mean Change')\n"
        "ax.errorbar([0], [md], xerr=[[md - ci_lo], [ci_hi - md]])\n",
        encoding="utf-8",
    )
    computation = _display_computation(tmp_path, figure)
    computation.records[0].artifacts.append(
        ArtifactRef(
            path=str(source),
            sha256=sha256_file(source),
            description="python analysis source",
        )
    )
    report = article_report(
        results="Figure 1 shows the effect estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Difference in mean change with a 95% confidence interval.",
                artifact_path=str(figure),
                alt_text="Effect estimate and confidence interval.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_effect_axis_transposed" in {
        finding.code for finding in validation.findings
    }


def test_figure_rejects_y_interval_on_effect_x_axis(tmp_path):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.errorbar([0.0], [md], yerr=np.array([[el], [eh]]))\n"
        "ax.set_xlabel('Mean Difference (Treatment - Control)')\n",
        encoding="utf-8",
    )
    computation = _display_computation(tmp_path, figure)
    computation.records[0].artifacts.append(
        ArtifactRef(
            path=str(source),
            sha256=sha256_file(source),
            description="python analysis source",
        )
    )
    report = article_report(
        results="Figure 1 shows the effect estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect",
                caption="Difference in mean change with a 95% confidence interval.",
                artifact_path=str(figure),
                alt_text="Effect estimate and confidence interval.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_effect_axis_transposed" in {
        finding.code for finding in validation.findings
    }


def test_figure_rejects_duplicate_category_tick_positions(tmp_path):
    figure = tmp_path / "output" / "figures" / "groups.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    source = tmp_path / "analysis.py"
    source.write_text(
        "ax.scatter(control_x, control_y)\n"
        "ax.scatter(treatment_x, treatment_y)\n"
        "ax.set_xticks([0, 0])\n"
        "ax.set_xticklabels(['Control', 'Treatment'])\n",
        encoding="utf-8",
    )
    computation = _display_computation(tmp_path, figure)
    computation.records[0].artifacts.append(
        ArtifactRef(
            path=str(source),
            sha256=sha256_file(source),
            description="python analysis source",
        )
    )
    report = article_report(
        results="Figure 1 shows both groups.",
        displays=[
            ReportDisplay(
                display_id="group-figure",
                kind="figure",
                title="Groups",
                caption="Control and treatment observations by group.",
                artifact_path=str(figure),
                alt_text="Two groups on separate categorical positions.",
            )
        ],
    )

    validation = validate_report(report, computation=computation)

    assert "figure_duplicate_category_positions" in {
        finding.code for finding in validation.findings
    }


def test_unavailable_figure_ocr_does_not_invalidate_truthful_caption(
    tmp_path, monkeypatch
):
    figure = tmp_path / "output" / "figures" / "effect.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {"available": False, "reason": "ocr_worker_failed"},
    )
    report = article_report(
        results="Figure 1 reports the estimate.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Treatment effect",
                caption="Panel annotations include adjusted R-squared.",
                artifact_path=str(figure),
                alt_text="A treatment effect with confidence interval.",
            )
        ],
    )

    validation = validate_report(
        report, computation=_display_computation(tmp_path, figure)
    )

    assert validation.passed
    assert "figure_caption_claims_missing_annotation" not in {
        finding.code for finding in validation.findings
    }


def test_observed_baseline_result_cannot_justify_prespecification():
    report = article_report(
        discussion=(
            "Baseline values differed significantly (p = 0.006),\njustifying the "
            "prespecified ANCOVA sensitivity analysis."
        )
    )

    validation = validate_report(report)

    assert "posthoc_result_cannot_justify_prespecification" in {
        finding.code for finding in validation.findings
    }


def test_extracted_source_image_cannot_be_registered_as_final_display(
    tmp_path, monkeypatch
):
    figure = tmp_path / "output" / "extracted" / "source.png"
    figure.parent.mkdir(parents=True)
    Image.new("RGB", (800, 600), color="white").save(figure, dpi=(300, 300))
    monkeypatch.setattr(
        "scientific_agent.linting.extract_figure_ocr",
        lambda _path: {"available": True, "text": "Source image", "words": []},
    )
    report = article_report(
        results="Figure 1 shows the extracted source image.",
        displays=[
            ReportDisplay(
                display_id="source-image",
                kind="figure",
                title="Source image",
                caption="An archive extraction copy.",
                artifact_path=str(figure),
                alt_text="An extracted scientific source image.",
            )
        ],
    )

    validation = validate_report(
        report, computation=_display_computation(tmp_path, figure)
    )

    assert "display_not_reader_facing_output" in {
        finding.code for finding in validation.findings
    }
