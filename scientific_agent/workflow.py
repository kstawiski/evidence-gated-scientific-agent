"""ADK 2 graph for independent planning, synthesis, and plan audit."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from google.adk import Workflow
from google.adk.workflow import JoinNode
from pydantic import BaseModel

from .config import Settings
from .linting import lint_plan
from .prompts import PLAN_AUDITOR, PLANNER_A, PLANNER_B, SIMPLE_PLANNER, SYNTHESIZER
from .schemas import (
    MasterPlan,
    PlanBundle,
    PlanProposal,
    PlanningResult,
    TaskSpec,
    VerificationReport,
)
from .structured_client import request_structured


def normalize_task(node_input: str) -> TaskSpec:
    objective = node_input.strip()
    if objective.startswith("{"):
        try:
            return TaskSpec.model_validate_json(objective)
        except Exception:
            pass
    task_id = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:16]
    return TaskSpec(
        task_id=task_id,
        objective=objective,
        deliverables=["Evidence-backed scientific report with claim and source ledgers"],
        constraints=[
            "Read-only MVP: no shell, package installation, or model-controlled writes",
            "Model agreement is not proof; deterministic checks and retrieved evidence outrank both models",
            "Unknown requirements must remain explicit",
        ],
        unknowns=["Domain-specific acceptance criteria not stated by the user"],
        scientific_domain="general",
        task_type="mixed",
        security_risk="low",
        scientific_risk="exploratory",
        acceptance_tests=[
            "Every plan step declares outputs, validators, and stop conditions",
            "Every supported substantive claim links to a retrieved source record",
            "Gemma returns no unresolved blocking finding",
            "The provenance manifest hashes every run artifact",
        ],
    )


def keep_task(node_input: TaskSpec) -> TaskSpec:
    return node_input


def keep_master(node_input: MasterPlan) -> MasterPlan:
    return node_input


def _collect(value: Any, model_type: type[BaseModel]) -> list[BaseModel]:
    found: list[BaseModel] = []
    if isinstance(value, model_type):
        return [value]
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, dict):
        try:
            found.append(model_type.model_validate(value))
            return found
        except Exception:
            for nested in value.values():
                found.extend(_collect(nested, model_type))
    elif isinstance(value, list | tuple):
        for nested in value:
            found.extend(_collect(nested, model_type))
    return found


def merge_and_lint(node_input: dict) -> PlanBundle:
    tasks = _collect(node_input.get("keep_task", node_input), TaskSpec)
    branch_a = _collect(node_input.get("planner_a", {}), PlanProposal)
    branch_b = _collect(node_input.get("planner_b", {}), PlanProposal)
    if len(tasks) != 1:
        raise ValueError(f"planning join must contain one TaskSpec, found {len(tasks)}")
    if not branch_a or not branch_b:
        # Compatibility fallback for synthetic/offline join dictionaries.
        plans = _collect(node_input, PlanProposal)
        by_label = {plan.plan_label: plan for plan in plans}
        branch_a = [by_label["A"]] if "A" in by_label else []
        branch_b = [by_label["B"]] if "B" in by_label else []
    if len(branch_a) != 1 or len(branch_b) != 1:
        raise ValueError("planning join must contain one output from each blinded branch")
    task = TaskSpec.model_validate(tasks[0])
    # Branch identity is controller state. Do not trust a model to self-label it.
    plan_a = PlanProposal.model_validate(branch_a[0]).model_copy(update={"plan_label": "A"})
    plan_b = PlanProposal.model_validate(branch_b[0]).model_copy(update={"plan_label": "B"})
    return PlanBundle(
        task=task,
        plan_a=plan_a,
        plan_b=plan_b,
        lint_a=lint_plan(task, plan_a),
        lint_b=lint_plan(task, plan_b),
    )


def package_planning(node_input: dict) -> PlanningResult:
    masters = _collect(node_input, MasterPlan)
    audits = _collect(node_input, VerificationReport)
    if len(masters) != 1 or len(audits) != 1:
        raise ValueError("audit join must contain one MasterPlan and one VerificationReport")
    master = MasterPlan.model_validate(masters[0])
    audit = VerificationReport.model_validate(audits[0])
    master_lint = lint_plan(master.task, master.plan)
    if not master_lint.passed or audit.verdict == "fail":
        status = "requires_revision"
    elif audit.verdict == "inconclusive" and audit.blocking_findings:
        status = "inconclusive"
    else:
        status = "supported"
    return PlanningResult(
        master_plan=master,
        audit=audit,
        plan_lints=[master_lint],
        status=status,
    )


async def build_simple_planning(settings: Settings, task: TaskSpec) -> PlanningResult:
    """Create one lean Qwen plan; final-result Gemma audit remains independent."""

    proposal = await request_structured(
        settings.qwen,
        system_prompt=SIMPLE_PLANNER,
        payload=task,
        output_type=PlanProposal,
        temperature=0.2,
        max_tokens=1800,
        timeout=90,
        enable_thinking=False,
    )
    proposal = proposal.model_copy(update={"plan_label": "MASTER"})
    lint = lint_plan(task, proposal)
    audit = VerificationReport(
        verdict="pass" if lint.passed else "inconclusive",
        evidence_refs=["deterministic simple-plan lint; final Gemma result audit required"],
    )
    return PlanningResult(
        master_plan=MasterPlan(
            task=task,
            plan=proposal,
            resolutions=[],
            method_lock_required=task.scientific_risk in {"confirmatory", "decision_critical"},
            protocol_fields=[],
        ),
        audit=audit,
        plan_lints=[lint],
        status="supported" if lint.passed else "inconclusive",
    )


def build_planning_workflow(settings: Settings) -> Workflow:
    async def planner_a(node_input: TaskSpec) -> PlanProposal:
        return await request_structured(
            settings.qwen,
            system_prompt=PLANNER_A,
            payload=node_input,
            output_type=PlanProposal,
            temperature=0.8,
            max_tokens=3200,
            timeout=120,
            enable_thinking=True,
        )

    async def planner_b(node_input: TaskSpec) -> PlanProposal:
        return await request_structured(
            settings.gemma,
            system_prompt=PLANNER_B,
            payload=node_input,
            output_type=PlanProposal,
            temperature=0.3,
            max_tokens=1800,
            timeout=150,
            enable_thinking=False,
        )

    plan_join = JoinNode(name="join_independent_plans")

    async def plan_synthesizer(node_input: PlanBundle) -> MasterPlan:
        return await request_structured(
            settings.qwen,
            system_prompt=SYNTHESIZER,
            payload=node_input,
            output_type=MasterPlan,
            temperature=0.5,
            max_tokens=4000,
            timeout=150,
            enable_thinking=True,
        )

    async def plan_auditor(node_input: MasterPlan) -> VerificationReport:
        return await request_structured(
            settings.gemma,
            system_prompt=PLAN_AUDITOR,
            payload=node_input,
            output_type=VerificationReport,
            temperature=0.2,
            max_tokens=1400,
            timeout=150,
            enable_thinking=False,
        )

    audit_join = JoinNode(name="join_master_and_audit")
    return Workflow(
        name="evidence_gated_planning",
        input_schema=str,
        output_schema=PlanningResult,
        max_concurrency=2,
        edges=[
            (
                "START",
                normalize_task,
                (keep_task, planner_a, planner_b),
                plan_join,
                merge_and_lint,
                plan_synthesizer,
                (keep_master, plan_auditor),
                audit_join,
                package_planning,
            )
        ],
    )
