from PIL import Image

from scientific_agent.linting import lint_plan, validate_report
from scientific_agent.provenance import sha256_file
from scientific_agent.schemas import (
    CheckSpec,
    ClaimRecord,
    ComputationEvidence,
    ComputationRecord,
    EvidenceStatus,
    ArtifactRef,
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


def test_plan_linter_accepts_complete_read_only_plan():
    assert lint_plan(task(), good_plan()).passed


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
    artifact_path = tmp_path / "reconciliation.json"
    artifact_path.write_text('{"all_pass": false}\n', encoding="utf-8")
    artifact = ArtifactRef(
        path=str(artifact_path),
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

    failed = validate_report(
        report,
        computation=computation,
        require_reconciliation=True,
    )
    assert "cross_language_reconciliation_failed" in {
        finding.code for finding in failed.findings
    }

    artifact_path.write_text('{"all_pass": true}\n', encoding="utf-8")
    passed = validate_report(
        report,
        computation=computation,
        require_reconciliation=True,
    )
    assert passed.passed

    artifact_path.write_text('{"all_pass": false}\n', encoding="utf-8")
    corrected_path = tmp_path / "reconciliation-corrected.json"
    corrected_path.write_text('{"all_pass": true}\n', encoding="utf-8")
    corrected = ArtifactRef(
        path=str(corrected_path),
        sha256="def",
        description="sandbox-generated analysis artifact",
    )
    corrected_record = ComputationRecord(
        execution_id="exec-002",
        language="r",
        code_sha256="def",
        started_at="2026-07-13T15:00:00Z",
        duration_seconds=0.1,
        exit_code=0,
        status="succeeded",
        stdout_path=str(tmp_path / "stdout-2.txt"),
        stderr_path=str(tmp_path / "stderr-2.txt"),
        artifacts=[corrected],
    )
    corrected_computation = computation.model_copy(
        update={
            "records": [*computation.records, corrected_record],
            "artifacts": [*computation.artifacts, corrected],
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


def test_cross_language_claim_must_cite_reconciliation_artifact(tmp_path):
    reconciliation_path = tmp_path / "cross_language_reconciliation.json"
    reconciliation_path.write_text(
        '{"all_pass": true, "absolute_difference": 0.0}\n', encoding="utf-8"
    )
    r_path = tmp_path / "r_verification.json"
    r_path.write_text('{"r_point_estimate": 5.0}\n', encoding="utf-8")
    artifacts = [
        ArtifactRef(
            path=str(path),
            sha256=token * 64,
            description="sandbox-generated analysis artifact",
        )
        for path, token in ((reconciliation_path, "a"), (r_path, "b"))
    ]
    computation = ComputationEvidence(
        successful_calls=1,
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
                artifacts=artifacts,
            )
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
