from scientific_agent.config import Settings
from scientific_agent.schemas import (
    CheckSpec,
    MasterPlan,
    PlanProposal,
    PlanStep,
    TaskSpec,
    VerificationReport,
)
from scientific_agent.workflow import (
    build_planning_workflow,
    merge_and_lint,
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
