"""ADK 2 graph for independent planning, synthesis, and plan audit."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable

from google.adk import Workflow
from google.adk.workflow import JoinNode
from pydantic import BaseModel

from .config import Settings
from .linting import lint_plan
from .prompts import PLAN_AUDITOR, PLANNER_A, PLANNER_B, SIMPLE_PLANNER, SYNTHESIZER
from .schemas import (
    CheckSpec,
    Finding,
    MasterPlan,
    PlanAuditChecklist,
    PlanBundle,
    PlanLintReport,
    PlanProposal,
    PlanningResult,
    TaskSpec,
    VerificationReport,
)
from .structured_client import request_structured

PLAN_CRITIC_UNAVAILABLE = "plan-critic-unavailable"
DEFAULT_METHOD_LOCK_FIELDS = [
    "primary estimand and endpoints",
    "eligibility and exclusions",
    "preprocessing and missing-data handling",
    "statistical models and covariates",
    "effect measures and uncertainty",
    "multiplicity control",
    "sensitivity analyses",
    "software versions and random seeds",
    "protocol-deviation amendment rule",
]


def _scientific_risk(objective: str) -> str:
    """Conservatively infer explicit protocol stakes from a plain-text task."""

    normalized = " ".join(objective.casefold().split())
    normalized = re.sub(
        r"\b(can|could|do|does|did|is|are|was|were|must|should|would|will)"
        r"n['’]t\b",
        r"\1 not",
        normalized,
    )

    def has_unnegated(pattern: str) -> bool:
        for match in re.finditer(pattern, normalized):
            prefix = re.split(
                r"[.;:!?]|\b(?:and|but|however|instead|yet|then)\b",
                normalized[: match.start()],
            )[-1][-100:]
            if prefix.endswith("non-"):
                continue
            if re.search(r"\bnot\s+only\s*$", prefix):
                return True
            direct_negation = re.search(
                r"\b(?:not|never|without|cannot|can not|will not)\b"
                r"(?:\s+(?:a|an|the|to|be|been|currently|directly|explicitly|"
                r"necessarily|expected|considered|deemed|treated|intended)){0,5}"
                r"\s*$",
                prefix,
            )
            auxiliary_negation = re.search(
                r"\b(?:do|does|did|should|would|could|must|can|will)\s+not\b"
                r"[^,.;:!?]{0,40}$",
                prefix,
            )
            coordinated_negation = re.search(
                r"\bneither\b[^,.;:!?]{0,60}(?:\bnor\b[^,.;:!?]{0,30})?$",
                prefix,
            )
            if direct_negation or auxiliary_negation or coordinated_negation:
                continue
            return True
        return False

    if has_unnegated(r"\bdecision[ -]?critical\b") or has_unnegated(
        r"\b(?:inform|support|guide|drive|make|use|used)\b[^,.;:!?]{0,60}\b"
        r"(?:patient care|(?:regulatory|clinical|treatment|patient[ -]?care|"
        r"bedside clinical)\s+(?:decisions?|decision[ -]?making))\b",
    ):
        return "decision_critical"

    current_confirmatory_task = has_unnegated(
        r"\b(?:perform|conduct|run|execute|undertake|analy[sz]e|evaluate)\b"
        r"(?:(?!\b(?:not|never|non[ -]?|future|planned|subsequent|whether)\b)"
        r"[^,.;:!?]){0,45}\bconfirmatory\b|"
        r"\b(?:is|as)\s+(?:a\s+)?confirmatory\b|"
        r"^(?:a\s+)?confirmatory (?:analysis|test|evaluation|inference)\b"
    )
    if current_confirmatory_task:
        return "confirmatory"

    analysis_requested = bool(
        re.search(
            r"\b(?:analy[sz]e|estimate|evaluate|test|model|compute|conduct|run|"
            r"execute|perform|undertake)\b",
            normalized,
        )
    )
    operational_lock = has_unnegated(
        r"\b(?:prespecified|pre[ -]?specified|preregistered|pre[ -]?registered)\b"
        r"[^,.;:!?]{0,35}\b(?:endpoint|estimand|analysis|protocol|hypothesis|"
        r"model|outcome|study|design)\b|"
        r"\b(?:endpoint|estimand|analysis|protocol|hypothesis|model|outcome|"
        r"study|design)\b.{0,50}\b(?:was |is )?"
        r"(?:prespecified|pre[ -]?specified|preregistered|pre[ -]?registered)\b|"
        r"\b(?:endpoint|estimand|analysis|protocol|hypothesis|model|outcome)\b"
        r"[^,.;:!?]{0,35}\b(?:specified|defined) a priori\b|"
        r"\bas (?:prespecified|pre[ -]?specified|preregistered|"
        r"pre[ -]?registered)\b|\baccording to (?:the )?preregistration\b|"
        r"\blocked\b[^,.;:!?]{0,25}\b(?:analysis plan|protocol|plan)\b|"
        r"\b(?:statistical )?analysis plan\b[^,.;:!?]{0,40}\bfinalized\b"
        r"[^,.;:!?]{0,40}\bbefore\b[^,.;:!?]{0,30}\boutcomes?\b|"
        r"\bmethod lock\b|\block\b[^,.;:!?]{0,35}\b(?:method|analysis|model|"
        r"test|endpoint|estimand|welch)\b"
    )
    if analysis_requested and operational_lock:
        return "confirmatory"
    return "exploratory"


def normalize_task(node_input: str) -> TaskSpec:
    objective = node_input.strip()
    if objective.startswith("{"):
        try:
            return TaskSpec.model_validate_json(objective)
        except Exception:
            pass
    task_id = hashlib.sha256(objective.encode("utf-8")).hexdigest()[:16]
    required_languages = []
    if re.search(r"\bpython\b", objective, flags=re.IGNORECASE):
        required_languages.append("python")
    if re.search(r"(?<![A-Za-z0-9])R(?![A-Za-z0-9])", objective):
        required_languages.append("r")
    return TaskSpec(
        task_id=task_id,
        objective=objective,
        deliverables=[
            "Evidence-backed scientific report with claim and source ledgers"
        ],
        constraints=[
            "Read-only MVP: no shell, package installation, or model-controlled writes",
            "Model agreement is not proof; deterministic checks and retrieved evidence outrank both models",
            "Unknown requirements must remain explicit",
        ],
        unknowns=["Domain-specific acceptance criteria not stated by the user"],
        scientific_domain="general",
        task_type="mixed",
        security_risk="low",
        scientific_risk=_scientific_risk(objective),
        required_computation_languages=required_languages,
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


def bind_controller_task(master: MasterPlan, task: TaskSpec) -> MasterPlan:
    """Prevent model synthesis from rewriting controller-normalized requirements."""

    method_lock_required = task.scientific_risk in {
        "confirmatory",
        "decision_critical",
    }
    protocol_fields = list(master.protocol_fields)
    if method_lock_required:
        protocol_fields = list(
            dict.fromkeys([*DEFAULT_METHOD_LOCK_FIELDS, *protocol_fields])
        )[:12]
    return master.model_copy(
        update={
            "task": task,
            "method_lock_required": method_lock_required,
            "protocol_fields": protocol_fields,
        }
    )


def lint_bound_master(master: MasterPlan) -> PlanLintReport:
    """Lint a master plan against its complete controller-owned protocol lock."""

    required = set(DEFAULT_METHOD_LOCK_FIELDS)
    controller_method_lock = bool(
        master.task.scientific_risk in {"confirmatory", "decision_critical"}
        and master.method_lock_required
        and required.issubset(master.protocol_fields)
    )
    return lint_plan(
        master.task,
        master.plan,
        controller_method_lock=controller_method_lock,
    )


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
        raise ValueError(
            "planning join must contain one output from each blinded branch"
        )
    task = TaskSpec.model_validate(tasks[0])
    # Branch identity is controller state. Do not trust a model to self-label it.
    plan_a = PlanProposal.model_validate(branch_a[0]).model_copy(
        update={"plan_label": "A"}
    )
    plan_b = PlanProposal.model_validate(branch_b[0]).model_copy(
        update={"plan_label": "B"}
    )
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
        raise ValueError(
            "audit join must contain one MasterPlan and one VerificationReport"
        )
    master = MasterPlan.model_validate(masters[0])
    audit = VerificationReport.model_validate(audits[0])
    master_lint = lint_bound_master(master)
    status = planning_status(master_lint, audit)
    return PlanningResult(
        master_plan=master,
        audit=audit,
        plan_lints=[master_lint],
        status=status,
    )


def build_plan_audit_packet(master: MasterPlan) -> dict:
    """Build a compact blinded packet with deterministic lint evidence."""

    task = master.task
    plan = master.plan
    lint = lint_bound_master(master)
    return {
        "task": {
            "objective": task.objective,
            "deliverables": task.deliverables,
            "available_inputs": [
                item.model_dump(mode="json") for item in task.available_inputs
            ],
            "input_profile": (
                task.input_profile.model_dump(mode="json")
                if task.input_profile is not None
                else None
            ),
            "constraints": task.constraints,
            "unknowns": task.unknowns,
            "scientific_domain": task.scientific_domain,
            "task_type": task.task_type,
            "scientific_risk": task.scientific_risk,
            "acceptance_tests": task.acceptance_tests,
        },
        "plan": {
            "objective": plan.objective,
            "assumptions": plan.assumptions,
            "required_data": plan.required_data,
            "alternatives_considered": plan.alternatives_considered,
            "foreseeable_failure_modes": plan.foreseeable_failure_modes,
            "steps": [step.model_dump(mode="json") for step in plan.steps],
            "expected_artifacts": plan.expected_artifacts,
            "unresolved_questions": plan.unresolved_questions,
            "estimated_resources": plan.estimated_resources,
        },
        "resolutions": [item.model_dump(mode="json") for item in master.resolutions],
        "method_lock_required": master.method_lock_required,
        "protocol_fields": master.protocol_fields,
        "deterministic_lint": lint.model_dump(mode="json"),
    }


def plan_audit_to_verification(
    checklist: PlanAuditChecklist,
    *,
    master: MasterPlan,
) -> VerificationReport:
    """Derive the overall verdict in deterministic controller code."""

    blocking: list[Finding] = []
    nonblocking = list(checklist.nonblocking_findings)
    proposed_tests: list[CheckSpec] = []
    statuses: list[str] = []
    for review in checklist.reviews:
        statuses.append(review.status)
        if review.status == "pass":
            continue
        finding = review.finding
        assert finding is not None
        blocking.append(
            Finding(
                finding_id=f"plan-audit-{review.criterion}",
                location=finding.location,
                problem=finding.problem,
                why_it_matters=finding.why_it_matters,
                evidence=finding.plan_evidence_quote,
                falsification_test_or_correction=(
                    finding.falsification_test_or_correction
                ),
            )
        )
        proposed_tests.append(
            CheckSpec(
                check_id=f"test-{review.criterion}",
                description=finding.falsification_test_or_correction,
                check_type="test",
                blocking=True,
            )
        )

    lint = lint_bound_master(master)
    for item in lint.findings:
        target = blocking if item.blocking else nonblocking
        if len(target) >= 8:
            continue
        target.append(
            Finding(
                finding_id=f"deterministic-{item.code}",
                location=item.location,
                problem=item.message,
                why_it_matters="The executable plan contract did not pass deterministic lint.",
                evidence=f"Deterministic lint code: {item.code}",
                falsification_test_or_correction=(
                    "Correct the declared plan field and rerun deterministic plan lint."
                ),
            )
        )

    if not lint.passed or "fail" in statuses:
        verdict = "fail"
    elif "inconclusive" in statuses:
        verdict = "inconclusive"
    elif nonblocking:
        verdict = "pass_with_nonblocking_comments"
    else:
        verdict = "pass"
    return VerificationReport(
        verdict=verdict,
        blocking_findings=blocking,
        nonblocking_findings=nonblocking,
        proposed_falsification_tests=proposed_tests,
        evidence_refs=[
            "bounded independent plan audit: "
            + ", ".join(
                f"{review.criterion}={review.status}" for review in checklist.reviews
            ),
            f"deterministic plan lint passed={lint.passed}",
        ],
    )


async def audit_master_plan(
    settings: Settings,
    master: MasterPlan,
    on_visible_text: Callable[[str, str], None] | None = None,
) -> VerificationReport:
    if on_visible_text is not None:
        try:
            on_visible_text(
                "Gemma",
                "\n[Controller: Gemma plan audit started; private reasoning is not "
                "displayed.]\n",
            )
        except Exception:
            pass
    try:
        checklist = await request_structured(
            settings.gemma,
            system_prompt=PLAN_AUDITOR,
            payload=build_plan_audit_packet(master),
            output_type=PlanAuditChecklist,
            temperature=settings.gemma.temperature,
            timeout=150,
            on_visible_text=(
                (lambda text: on_visible_text("Gemma", text))
                if on_visible_text is not None
                else None
            ),
        )
    except Exception as exc:
        return VerificationReport(
            verdict="inconclusive",
            blocking_findings=[
                Finding(
                    finding_id=PLAN_CRITIC_UNAVAILABLE,
                    location="plan audit",
                    problem="The independent plan critic did not return a valid audit.",
                    why_it_matters=(
                        "Research cannot begin without the required independent "
                        "methodological review."
                    ),
                    evidence=f"critic transition failed closed: {type(exc).__name__}",
                    falsification_test_or_correction=(
                        "Rerun the bounded Gemma plan audit when the critic can "
                        "produce a valid non-repetitive response."
                    ),
                )
            ],
            evidence_refs=[f"critic unavailable: {type(exc).__name__}"],
        )
    return plan_audit_to_verification(checklist, master=master)


def planning_status(master_lint, audit: VerificationReport) -> str:
    if any(
        finding.finding_id == PLAN_CRITIC_UNAVAILABLE
        for finding in audit.blocking_findings
    ):
        return "inconclusive"
    if not master_lint.passed or audit.verdict == "fail" or audit.blocking_findings:
        return "requires_revision"
    return "supported"


async def build_simple_planning(
    settings: Settings,
    task: TaskSpec,
    on_visible_text: Callable[[str, str], None] | None = None,
) -> PlanningResult:
    """Create one lean Qwen plan and subject it to a bounded Gemma audit."""

    proposal = await request_structured(
        settings.qwen,
        system_prompt=SIMPLE_PLANNER,
        payload=task,
        output_type=PlanProposal,
        temperature=0.2,
        timeout=90,
        on_visible_text=(
            (lambda text: on_visible_text("Qwen", text))
            if on_visible_text is not None
            else None
        ),
    )
    artifacts = list(proposal.expected_artifacts)
    controller_report = (
        "Evidence-backed scientific report with claim and source ledgers"
    )
    if controller_report in task.deliverables and controller_report not in artifacts:
        artifacts.append(controller_report)
    proposal = proposal.model_copy(
        update={"plan_label": "MASTER", "expected_artifacts": artifacts}
    )
    master = MasterPlan(
        task=task,
        plan=proposal,
        resolutions=[],
        method_lock_required=task.scientific_risk
        in {"confirmatory", "decision_critical"},
        protocol_fields=(
            list(DEFAULT_METHOD_LOCK_FIELDS)
            if task.scientific_risk in {"confirmatory", "decision_critical"}
            else []
        ),
    )
    lint = lint_bound_master(master)
    audit = await audit_master_plan(settings, master, on_visible_text)
    status = planning_status(lint, audit)
    return PlanningResult(
        master_plan=master,
        audit=audit,
        plan_lints=[lint],
        status=status,
    )


def build_planning_workflow(
    settings: Settings,
    on_visible_text: Callable[[str, str], None] | None = None,
) -> Workflow:
    async def planner_a(node_input: TaskSpec) -> PlanProposal:
        return await request_structured(
            settings.qwen,
            system_prompt=PLANNER_A,
            payload=node_input,
            output_type=PlanProposal,
            temperature=0.8,
            timeout=180,
            on_visible_text=(
                (lambda text: on_visible_text("Qwen", text))
                if on_visible_text is not None
                else None
            ),
        )

    async def planner_b(node_input: TaskSpec) -> PlanProposal:
        return await request_structured(
            settings.gemma,
            system_prompt=PLANNER_B,
            payload=node_input,
            output_type=PlanProposal,
            temperature=settings.gemma.temperature,
            timeout=150,
            on_visible_text=(
                (lambda text: on_visible_text("Gemma", text))
                if on_visible_text is not None
                else None
            ),
        )

    plan_join = JoinNode(name="join_independent_plans")

    async def plan_synthesizer(node_input: PlanBundle) -> MasterPlan:
        master = await request_structured(
            settings.qwen,
            system_prompt=SYNTHESIZER,
            payload=node_input,
            output_type=MasterPlan,
            temperature=0.5,
            timeout=150,
            on_visible_text=(
                (lambda text: on_visible_text("Qwen", text))
                if on_visible_text is not None
                else None
            ),
        )
        return bind_controller_task(master, node_input.task)

    async def plan_auditor(node_input: MasterPlan) -> VerificationReport:
        return await audit_master_plan(settings, node_input, on_visible_text)

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
