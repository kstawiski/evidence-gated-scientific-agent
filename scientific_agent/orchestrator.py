"""Evidence-gated run controller for scientific research and computation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from google.adk import Agent

from .config import Settings
from .execution import AnalysisExecutor, build_analysis_tools
from .linting import lint_plan, validate_report
from .mcp import build_mcp_toolsets, close_mcp_toolsets
from .models import qwen_model
from .policy import ToolPolicy, default_allowed_tools
from .prompts import (
    PLAN_AUDITOR,
    REPAIRER,
    REPORT_AUDITOR,
    REPORTER,
    RESEARCHER,
    SYNTHESIZER,
)
from .provenance import EventLedger, build_manifest, utc_now, write_json
from .runtime import run_text, run_typed
from .schemas import (
    MasterPlan,
    ComputationEvidence,
    PlanBundle,
    PlanningResult,
    RunResult,
    RetrievalEvidence,
    ScientificReport,
    VerificationReport,
)
from .workflow import build_planning_workflow, build_simple_planning, normalize_task
from .workspace_tools import build_workspace_tools
from .structured_client import request_structured


def _fallback_computation_packet(
    error: Exception,
    computation: ComputationEvidence,
) -> str:
    """Preserve successful evidence after a malformed trailing model tool call."""

    logs = []
    for record in computation.records:
        if record.status != "succeeded":
            continue
        logs.append(
            {
                "execution_id": record.execution_id,
                "language": record.language,
                "stdout": Path(record.stdout_path).read_text(
                    encoding="utf-8", errors="replace"
                )[: 32 * 1024],
                "stderr": Path(record.stderr_path).read_text(
                    encoding="utf-8", errors="replace"
                )[: 8 * 1024],
            }
        )
    output_previews = []
    remaining = 64 * 1024
    for artifact in computation.artifacts:
        path = Path(artifact.path)
        if path.suffix.lower() not in {".csv", ".json", ".md", ".tsv", ".txt"}:
            continue
        content = path.read_text(encoding="utf-8", errors="replace")[: min(8192, remaining)]
        output_previews.append({"path": artifact.path, "content": content})
        remaining -= len(content.encode("utf-8"))
        if remaining <= 0:
            break
    return json.dumps(
        {
            "research_agent_warning": (
                "The ADK research turn ended after successful computation because "
                "a later model tool call was malformed. Use only the controller-"
                "recorded execution evidence below."
            ),
            "error_type": type(error).__name__,
            "successful_execution_logs": logs,
            "generated_output_previews": output_previews,
        },
        sort_keys=True,
    )


def _report_markdown(report: ScientificReport) -> str:
    lines = [f"# {report.title}", "", report.executive_summary, "", "## Methods", ""]
    lines.extend(f"- {method}" for method in report.methods)
    lines.extend(["", "## Report", "", report.narrative, "", "## Claims", ""])
    for claim in report.claims:
        refs = ", ".join(claim.evidence_refs) or "none"
        lines.append(
            f"- **{claim.claim_id} [{claim.status.value}]** {claim.text} (evidence: {refs})"
        )
    lines.extend(["", "## Sources", ""])
    for source in report.sources:
        if source.url is not None:
            lines.append(f"- **{source.source_id}:** [{source.title}]({source.url})")
        else:
            lines.append(
                f"- **{source.source_id}:** {source.title} "
                f"(artifact: `{source.artifact_path}`)"
            )
    if report.unresolved_issues:
        lines.extend(["", "## Unresolved issues", ""])
        lines.extend(f"- {item}" for item in report.unresolved_issues)
    if report.limitations:
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in report.limitations)
    return "\n".join(lines) + "\n"


async def _audit_plan(settings: Settings, master: MasterPlan) -> VerificationReport:
    return await request_structured(
        settings.gemma,
        system_prompt=PLAN_AUDITOR,
        payload=master,
        output_type=VerificationReport,
        temperature=0.2,
        max_tokens=1400,
        timeout=150,
        enable_thinking=False,
    )


async def _repair_plan(settings: Settings, planning: PlanningResult) -> PlanningResult:
    bundle = {
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "audit": planning.audit.model_dump(mode="json"),
        "instruction": "Correct only concrete blocking findings and preserve uncertainty.",
    }
    master = await request_structured(
        settings.qwen,
        system_prompt=SYNTHESIZER,
        payload=bundle,
        output_type=MasterPlan,
        temperature=0.3,
        max_tokens=3000,
        timeout=150,
        enable_thinking=True,
    )
    audit = await _audit_plan(settings, master)
    lint = lint_plan(master.task, master.plan)
    if lint.passed and audit.verdict in {"pass", "pass_with_nonblocking_comments"}:
        status = "supported"
    elif audit.verdict == "inconclusive" and audit.blocking_findings:
        status = "inconclusive"
    else:
        status = "requires_revision"
    return PlanningResult(master_plan=master, audit=audit, plan_lints=[lint], status=status)


async def _produce_report(
    settings: Settings,
    planning: PlanningResult,
    ledger: EventLedger,
    mcp_names: tuple[str, ...],
    include_chrome: bool,
    prior_report: ScientificReport | None = None,
    validation=None,
    review: VerificationReport | None = None,
    evidence_dir: Path | None = None,
    enable_code: bool = False,
    computation_dir: Path | None = None,
    simple_mode: bool = False,
) -> tuple[ScientificReport, RetrievalEvidence, ComputationEvidence]:
    toolsets = build_mcp_toolsets(settings, mcp_names) if mcp_names else []
    workspace_tools = build_workspace_tools(settings.workspace)
    executor = None
    analysis_tools = []
    if enable_code:
        if computation_dir is None:
            raise ValueError("computation_dir is required when code execution is enabled")
        sandbox = (
            replace(settings.sandbox, max_calls_per_attempt=2, max_wall_seconds=120)
            if simple_mode
            else settings.sandbox
        )
        executor = AnalysisExecutor(
            settings.workspace,
            computation_dir,
            sandbox,
        )
        analysis_tools = build_analysis_tools(executor)
    policy = ToolPolicy(
        ledger=ledger,
        allowed_tools=default_allowed_tools(
            include_chrome=include_chrome,
            enable_code=enable_code,
        ),
        evidence_dir=evidence_dir,
    )
    repairing = prior_report is not None
    research_agent = Agent(
        name="qwen_research_repairer" if repairing else "qwen_researcher",
        model=qwen_model(
            settings,
            temperature=0.3 if repairing else 0.6,
            max_tokens=2600 if simple_mode else 4000,
            timeout=120 if simple_mode else 240,
        ),
        instruction=RESEARCHER,
        tools=[*workspace_tools, *analysis_tools, *toolsets],
        before_tool_callback=policy.before_tool,
        after_tool_callback=policy.after_tool,
        mode="chat",
        include_contents="none",
    )
    payload = {
        "task": planning.master_plan.task.model_dump(mode="json"),
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "retrieval_requirement": (
            "External retrieval tools are available. Use them when the task needs "
            "current facts, documentation, literature, or citations. Every source "
            "URL in the report must occur in a successful tool result."
            if mcp_names
            else "No external retrieval tool is configured."
        ),
        "runtime_provenance_contract": (
            "The deterministic controller writes all run artifacts and creates "
            "manifest.json with SHA-256 hashes after report review."
        ),
        "code_execution": (
            "AUTHORIZED: Python and R may run only through the offline sandbox "
            "tools. /workspace is read-only and /output is the only writable path."
            if enable_code
            else "DISABLED: no Python or R execution tool is available."
        ),
    }
    if repairing:
        payload.update(
            {
                "report": prior_report.model_dump(mode="json"),
                "deterministic_validation": validation.model_dump(mode="json"),
                "scientific_review": review.model_dump(mode="json") if review else None,
            }
        )
    research_error: Exception | None = None
    try:
        research_packet = await run_text(research_agent, payload)
    except Exception as exc:
        research_error = exc
        research_packet = ""
    finally:
        await close_mcp_toolsets(toolsets)
    retrieval = policy.retrieval_evidence()
    computation = executor.evidence() if executor is not None else ComputationEvidence()
    if research_error is not None:
        if computation.successful_calls == 0:
            raise research_error
        ledger.append(
            "research_error_recovered",
            {
                "error_type": type(research_error).__name__,
                "successful_computations": computation.successful_calls,
            },
        )
        research_packet = _fallback_computation_packet(research_error, computation)
    report_payload = {
        **payload,
        "research_packet": research_packet,
        "retrieval_evidence": retrieval.model_dump(mode="json"),
        "computation_evidence": computation.model_dump(mode="json"),
    }
    report = await request_structured(
        settings.qwen,
        system_prompt=REPAIRER if repairing else REPORTER,
        payload=report_payload,
        output_type=ScientificReport,
        temperature=0.2 if repairing else 0.4,
        max_tokens=3000 if simple_mode else 5000,
        timeout=90 if simple_mode else 180,
        enable_thinking=False,
    )
    return report, retrieval, computation


async def _audit_report(
    settings: Settings,
    planning: PlanningResult,
    report: ScientificReport,
    validation,
    retrieval: RetrievalEvidence,
    computation: ComputationEvidence,
    simple_mode: bool = False,
) -> VerificationReport:
    payload = {
        "task": planning.master_plan.task.model_dump(mode="json"),
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "deterministic_validation": validation.model_dump(mode="json"),
        "retrieval_evidence": retrieval.model_dump(mode="json"),
        "computation_evidence": computation.model_dump(mode="json"),
        "runtime_provenance_contract": (
            "After this audit, the deterministic controller writes run artifacts "
            "and generates manifest.json with SHA-256 hashes."
        ),
    }
    return await request_structured(
        settings.gemma,
        system_prompt=REPORT_AUDITOR,
        payload=payload,
        output_type=VerificationReport,
        temperature=0.3,
        max_tokens=1000 if simple_mode else 1600,
        timeout=90 if simple_mode else 150,
        enable_thinking=False,
    )


async def run_scientific_task(
    objective: str,
    settings: Settings,
    *,
    mcp_names: tuple[str, ...] | None = None,
    include_chrome: bool = False,
    enable_code: bool = False,
    simple_mode: bool = False,
    progress: Callable[[str, str], None] | None = None,
) -> RunResult:
    def report_progress(phase: str, message: str) -> None:
        if progress is None:
            return
        try:
            progress(phase, message)
        except Exception:
            # Progress reporting is observational and must never change a run result.
            pass

    run_id = f"{utc_now().replace(':', '').replace('+00:00', 'Z')}-{uuid.uuid4().hex[:8]}"
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    ledger = EventLedger(run_dir / "tool_call_log.jsonl")
    objective_bytes = objective.encode("utf-8")
    ledger.append(
        "run_started",
        {
            "run_id": run_id,
            "objective_sha256": hashlib.sha256(objective_bytes).hexdigest(),
            "objective_bytes": len(objective_bytes),
        },
    )
    selected_mcp = mcp_names if mcp_names is not None else settings.mcp_servers
    report_progress(
        "planning",
        "Qwen is preparing one lean plan"
        if simple_mode
        else "Qwen and Gemma are preparing independent plans",
    )
    write_json(
        run_dir / "run_configuration.json",
        {
            "qwen": {"model": settings.qwen.model, "base_url": settings.qwen.base_url},
            "gemma": {"model": settings.gemma.model, "base_url": settings.gemma.base_url},
            "mcp_servers": list(selected_mcp),
            "chrome_enabled": include_chrome,
            "code_execution_enabled": enable_code,
            "sandbox": (
                {
                    "enabled": True,
                    "network": "unshared",
                    "workspace": "read-only",
                    "python": str(settings.sandbox.python),
                    "rscript": str(settings.sandbox.rscript),
                    "limits": {
                        "wall_seconds": settings.sandbox.max_wall_seconds,
                        "memory_bytes": settings.sandbox.max_memory_bytes,
                        "processes": settings.sandbox.max_processes,
                        "file_bytes": settings.sandbox.max_file_bytes,
                        "total_output_bytes": settings.sandbox.max_output_bytes,
                        "code_bytes": settings.sandbox.max_code_bytes,
                        "calls_per_attempt": settings.sandbox.max_calls_per_attempt,
                    },
                }
                if enable_code
                else {"enabled": False}
            ),
            "max_repair_rounds": settings.max_repair_rounds,
            "execution_mode": "simple" if simple_mode else "full",
        },
    )

    planning_input = objective
    if enable_code:
        task = normalize_task(objective)
        constraints = [
            constraint
            for constraint in task.constraints
            if not constraint.startswith("Read-only MVP")
        ]
        constraints.append(
            "Python and R are authorized only through the offline bubblewrap sandbox; "
            "inputs are read-only and outputs are confined and resource-bounded"
        )
        acceptance_tests = [
            (
                "Every supported substantive claim links to an exact retrieved URL "
                "or successful generated computation artifact"
                if test
                == "Every supported substantive claim links to a retrieved source record"
                else test
            )
            for test in task.acceptance_tests
        ]
        task = task.model_copy(
            update={
                "constraints": constraints,
                "acceptance_tests": acceptance_tests,
                "security_risk": "medium",
            }
        )
        planning_input = task.model_dump_json()
    if simple_mode:
        planning = await build_simple_planning(settings, normalize_task(planning_input))
    else:
        workflow = build_planning_workflow(settings)
        planning = await run_typed(workflow, planning_input, PlanningResult)
    if planning.status == "requires_revision" and not simple_mode:
        report_progress("plan-review", "The plan audit found a concrete issue; repairing it")
        ledger.append("plan_repair_started", {"reason": planning.audit.verdict})
        planning = await _repair_plan(settings, planning)
    write_json(run_dir / "planning_result.json", planning)

    if planning.status != "supported":
        report_progress("stopped", "Planning did not produce an evidence-ready protocol")
        result = RunResult(
            run_id=run_id,
            status="inconclusive" if planning.status == "inconclusive" else "requires_more_evidence",
            planning=planning,
            provenance_dir=str(run_dir),
        )
        write_json(run_dir / "run_result.json", result)
        build_manifest(run_dir)
        return result

    report_progress("research", "Executing the locked method and collecting evidence")
    report, retrieval, computation = await _produce_report(
        settings,
        planning,
        ledger,
        selected_mcp,
        include_chrome,
        evidence_dir=run_dir / "evidence" / "attempt-0",
        enable_code=enable_code,
        computation_dir=run_dir / "computations" / "attempt-0",
        simple_mode=simple_mode,
    )
    report_progress("validation", "Running deterministic claim and artifact checks")
    validation = validate_report(report, retrieval, computation)
    report_progress("scientific-review", "Gemma is independently auditing the result")
    review = await _audit_report(
        settings, planning, report, validation, retrieval, computation, simple_mode
    )
    repair_rounds = 0
    while (
        (
            not validation.passed
            or (not simple_mode and review.verdict in {"fail", "inconclusive"})
        )
        and repair_rounds < settings.max_repair_rounds
    ):
        repair_rounds += 1
        report_progress(
            "repair",
            f"Addressing falsifiable findings (repair {repair_rounds} of {settings.max_repair_rounds})",
        )
        ledger.append(
            "report_repair_started",
            {
                "round": repair_rounds,
                "deterministic_passed": validation.passed,
                "gemma_verdict": review.verdict,
            },
        )
        report, repair_retrieval, repair_computation = await _produce_report(
            settings,
            planning,
            ledger,
            selected_mcp,
            include_chrome,
            prior_report=report,
            validation=validation,
            review=review,
            evidence_dir=run_dir / "evidence" / f"attempt-{repair_rounds}",
            enable_code=enable_code,
            computation_dir=(
                run_dir / "computations" / f"attempt-{repair_rounds}"
            ),
            simple_mode=simple_mode,
        )
        retrieval = RetrievalEvidence(
            successful_calls=retrieval.successful_calls
            + repair_retrieval.successful_calls,
            tools=sorted(set(retrieval.tools) | set(repair_retrieval.tools)),
            urls=sorted(set(retrieval.urls) | set(repair_retrieval.urls)),
            retrieval_dates=sorted(
                set(retrieval.retrieval_dates)
                | set(repair_retrieval.retrieval_dates)
            ),
            artifacts=[*retrieval.artifacts, *repair_retrieval.artifacts],
        )
        computation = ComputationEvidence(
            successful_calls=(
                computation.successful_calls + repair_computation.successful_calls
            ),
            records=[*computation.records, *repair_computation.records],
            artifacts=[*computation.artifacts, *repair_computation.artifacts],
        )
        validation = validate_report(report, retrieval, computation)
        report_progress("scientific-review", "Gemma is auditing the repaired result")
        review = await _audit_report(
            settings, planning, report, validation, retrieval, computation, simple_mode
        )

    if validation.passed and review.verdict == "pass":
        status = "supported"
    elif validation.passed and review.verdict == "pass_with_nonblocking_comments":
        status = "supported_with_comments"
    elif review.verdict == "inconclusive":
        status = "inconclusive"
    else:
        status = "requires_more_evidence"

    result = RunResult(
        run_id=run_id,
        status=status,
        planning=planning,
        report=report,
        deterministic_validation=validation,
        retrieval_evidence=retrieval,
        computation_evidence=computation,
        scientific_review=review,
        repair_rounds=repair_rounds,
        provenance_dir=str(run_dir),
    )
    report_progress("finalizing", "Writing the report, evidence ledger, and manifest")
    write_json(run_dir / "scientific_report.json", report)
    (run_dir / "report.md").write_text(_report_markdown(report), encoding="utf-8")
    write_json(run_dir / "deterministic_validation.json", validation)
    write_json(run_dir / "retrieval_evidence.json", retrieval)
    write_json(run_dir / "computation_evidence.json", computation)
    write_json(run_dir / "gemma_review.json", review)
    write_json(run_dir / "run_result.json", result)
    ledger.append("run_completed", {"status": status, "repair_rounds": repair_rounds})
    build_manifest(run_dir)
    report_progress("complete", "Validated result is ready")
    return result


def run(objective: str, settings: Settings, **kwargs) -> RunResult:
    return asyncio.run(run_scientific_task(objective, settings, **kwargs))
