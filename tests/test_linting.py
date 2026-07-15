from scientific_agent.linting import lint_plan, validate_report
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
