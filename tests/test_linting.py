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
    report = ScientificReport(
        title="x",
        executive_summary="x",
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
    report = ScientificReport(
        title="x",
        executive_summary="x",
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


def test_report_validator_rejects_source_shaped_json_without_retrieval():
    report = ScientificReport(
        title="x",
        executive_summary="x",
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


def test_report_validator_accepts_url_and_date_seen_in_tool_output():
    report = ScientificReport(
        title="x",
        executive_summary="x",
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
    report = ScientificReport(
        title="x",
        executive_summary="x",
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
    report = ScientificReport(
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
    report = ScientificReport(
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
    report = ScientificReport(
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
    report = ScientificReport(
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
    report = ScientificReport(
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
    report = ScientificReport(
        title="Strict JSON",
        executive_summary="The artifact is machine readable.",
        methods=[],
        claims=[],
        sources=[],
        narrative="Strict serialization is required.",
    )

    invalid = validate_report(report, computation=computation)
    assert "invalid_generated_json" in {
        finding.code for finding in invalid.findings
    }

    output.write_text('{"estimate": null}\n', encoding="utf-8")
    assert validate_report(report, computation=computation).passed


def test_report_validator_requires_passing_reconciliation_artifact(tmp_path):
    report = ScientificReport(
        title="Cross-check",
        executive_summary="Implementations compared.",
        methods=[],
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
