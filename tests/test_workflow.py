import pytest

from scientific_agent.config import Settings
from scientific_agent.orchestrator import (
    _audit_report_resilient,
    _compact_computation_summary,
    _merge_computation_evidence,
    _merge_retrieval_evidence,
    _prepare_task_spec,
    _write_attempt_bundle,
)
from scientific_agent.provenance import EventLedger
from scientific_agent.schemas import (
    ArtifactRef,
    CheckSpec,
    ComputationEvidence,
    ComputationRecord,
    DeterministicValidation,
    MasterPlan,
    PlanProposal,
    PlanStep,
    RetrievalEvidence,
    ScientificReport,
    TaskSpec,
    VerificationReport,
)
from scientific_agent.workflow import (
    bind_controller_task,
    build_simple_planning,
    build_planning_workflow,
    merge_and_lint,
    normalize_task,
    package_planning,
)


def _task():
    return TaskSpec(
        task_id="t",
        objective="Produce a report",
        deliverables=["scientific report"],
        acceptance_tests=["validated"],
    )


def _plan(label):
    return PlanProposal(
        plan_label=label,
        objective="Produce a report",
        steps=[
            PlanStep(
                step_id=f"{label}-1",
                objective="work",
                outputs=["scientific report"],
                methods=["retrieval"],
                validators=[CheckSpec(check_id="c", description="check", check_type="source")],
                stop_conditions=["done"],
            )
        ],
        expected_artifacts=["scientific report"],
    )


def test_join_bundle_recovers_blinded_plans():
    bundle = merge_and_lint(
        {"keep_task": _task(), "planner_a": _plan("A"), "planner_b": _plan("B")}
    )
    assert bundle.plan_a.plan_label == "A"
    assert bundle.plan_b.plan_label == "B"


def test_adk_graph_builds_and_validates():
    workflow = build_planning_workflow(Settings())
    assert workflow.name == "evidence_gated_planning"
    assert {node.name for node in workflow.graph.nodes} >= {
        "planner_a",
        "planner_b",
        "join_independent_plans",
        "plan_synthesizer",
        "plan_auditor",
    }


def test_task_normalizer_locks_explicit_computation_languages():
    task = normalize_task("Cross-check this analysis independently in Python and R.")
    assert task.required_computation_languages == ["python", "r"]


def test_run_authorization_clears_languages_for_documentation_only_task():
    task = _prepare_task_spec(
        "Compare the official Python and R API documentation.", enable_code=False
    )

    assert task.required_computation_languages == []
    assert any("no code-execution authorization" in item for item in task.constraints)


def test_controller_task_cannot_be_shortened_by_plan_synthesis():
    full = normalize_task(
        "Analyze the dataset in Python and R, then save a reconciliation artifact."
    )
    shortened = full.model_copy(
        update={"objective": "Analyze the dataset.", "required_computation_languages": []}
    )
    master = MasterPlan(
        task=shortened,
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )

    bound = bind_controller_task(master, full)

    assert bound.task.objective == full.objective
    assert bound.task.required_computation_languages == ["python", "r"]


def test_evidence_pending_plan_can_reach_retrieval():
    master = MasterPlan(
        task=_task(),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )
    result = package_planning(
        {
            "keep_master": master,
            "plan_auditor": VerificationReport(verdict="inconclusive"),
        }
    )
    assert result.status == "supported"


def test_repair_evidence_is_cumulative_and_attempts_are_persisted(tmp_path):
    retrieval = _merge_retrieval_evidence(
        RetrievalEvidence(successful_calls=1, tools=["brave"], urls=["https://a"]),
        RetrievalEvidence(successful_calls=2, tools=["context7"], urls=["https://b"]),
    )
    computation = _merge_computation_evidence(
        ComputationEvidence(successful_calls=2),
        ComputationEvidence(successful_calls=1),
    )
    assert retrieval.successful_calls == 3
    assert retrieval.tools == ["brave", "context7"]
    assert computation.successful_calls == 3

    report = ScientificReport(
        title="Rejected draft",
        executive_summary="Draft",
        methods=[],
        claims=[],
        sources=[],
        narrative="Draft narrative",
    )
    _write_attempt_bundle(
        tmp_path,
        0,
        report,
        DeterministicValidation(passed=False),
        VerificationReport(verdict="inconclusive"),
        retrieval,
        computation,
    )
    attempt = tmp_path / "attempts" / "attempt-0"
    assert (attempt / "scientific_report.json").is_file()
    assert (attempt / "deterministic_validation.json").is_file()
    assert (attempt / "gemma_review.json").is_file()


def test_compact_computation_summary_keeps_evidence_and_drops_log_artifacts():
    generated = ArtifactRef(
        path="/run/output/result.json",
        sha256="a" * 64,
        description="sandbox-generated analysis artifact",
    )
    source = ArtifactRef(
        path="/run/analysis.py",
        sha256="b" * 64,
        description="python analysis source",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="c" * 64,
                started_at="2026-07-13T20:00:00Z",
                duration_seconds=1.2,
                exit_code=0,
                status="succeeded",
                stdout_path="/run/stdout.txt",
                stderr_path="/run/stderr.txt",
                artifacts=[source, generated],
            ),
            ComputationRecord(
                execution_id="exec-002",
                language="r",
                code_sha256="d" * 64,
                started_at="2026-07-13T20:01:00Z",
                duration_seconds=0.4,
                exit_code=1,
                status="failed",
                stdout_path="/run/stdout-r.txt",
                stderr_path="/run/stderr-r.txt",
                artifacts=[source],
            ),
        ],
        artifacts=[generated],
    )

    summary = _compact_computation_summary(computation)

    assert summary["failed_or_denied_calls"] == 1
    assert summary["records"][0]["analysis_artifacts"] == [
        generated.model_dump(mode="json")
    ]
    assert summary["records"][1]["analysis_artifacts"] == []
    assert summary["artifacts"] == [generated.model_dump(mode="json")]
    assert "stdout_path" not in summary["records"][0]


@pytest.mark.anyio
async def test_unavailable_critic_is_inconclusive_and_recorded(tmp_path, monkeypatch):
    async def unavailable(*args, **kwargs):
        del args, kwargs
        raise TimeoutError("critic timeout")

    monkeypatch.setattr("scientific_agent.orchestrator._audit_report", unavailable)
    ledger = EventLedger(tmp_path / "events.jsonl")
    review = await _audit_report_resilient(
        Settings(),
        None,
        None,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        ledger,
    )

    assert review.verdict == "inconclusive"
    assert "TimeoutError" in review.unsupported_claims[0]
    assert "independent_critic_unavailable" in (tmp_path / "events.jsonl").read_text()


def test_simple_planning_uses_one_qwen_request(monkeypatch):
    calls = []

    async def fake_request(*_args, **kwargs):
        calls.append(kwargs["system_prompt"])
        return _plan("MASTER")

    monkeypatch.setattr("scientific_agent.workflow.request_structured", fake_request)
    import asyncio

    controller_report = "Evidence-backed scientific report with claim and source ledgers"
    task = _task().model_copy(update={"deliverables": [controller_report]})
    result = asyncio.run(build_simple_planning(Settings(), task))
    assert result.status == "supported"
    assert len(calls) == 1
    assert result.audit.verdict == "pass"
    assert controller_report in result.master_plan.plan.expected_artifacts
