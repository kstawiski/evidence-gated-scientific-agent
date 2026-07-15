"""Evidence-gated run controller for scientific research and computation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from google.adk import Agent

from . import __version__
from .config import Settings
from .execution import build_analysis_tools, create_analysis_executor
from .environment import EnvironmentManager, build_environment_tools
from .linting import lint_plan, validate_report
from .literature import (
    LiteratureAcquirer,
    RemotePdfTextExtractor,
    build_acquired_article_audit,
    build_literature_tools,
)
from .mcp import build_mcp_toolsets, close_mcp_toolsets
from .models import qwen_model
from .policy import ToolPolicy, default_allowed_tools
from .prompts import (
    DISPLAY_AUDITOR,
    PLAN_REPAIRER,
    REPAIRER,
    REPORT_AUDITOR,
    REPORTER,
    RESEARCHER,
    REVISION_REPORTER,
    SIMPLE_REPORTER,
)
from .provenance import (
    EventLedger,
    build_environment_snapshot,
    build_input_manifest,
    build_manifest,
    sha256_file,
    utc_now,
    write_json,
)
from .reporting import (
    materialize_references,
    materialize_displays,
    prepare_display_audit,
    render_report_markdown,
)
from .runtime import run_text, run_typed
from .schemas import (
    ArtifactRef,
    ComputationEvidence,
    DeterministicValidation,
    Finding,
    MasterPlan,
    PlanningResult,
    RunResult,
    RetrievalEvidence,
    ScientificReport,
    TaskSpec,
    VerificationReport,
)
from .workflow import (
    audit_master_plan,
    build_planning_workflow,
    build_simple_planning,
    normalize_task,
    planning_status,
)
from .workspace_tools import build_workspace_tools
from .structured_client import request_structured


ActivityCallback = Callable[[str, str, str, str, str | None], None]
PRESENTATION_ONLY_FINDINGS = {
    "computed_without_artifact",
    "display_mentions_out_of_order",
    "display_not_mentioned",
    "display_unknown_claim",
    "display_unknown_evidence_ref",
    "duplicate_claim_id",
    "duplicate_display_id",
    "duplicate_source_id",
    "figure_alt_text_missing",
    "model_supplied_display_number",
    "source_artifact_not_generated",
    "unknown_evidence_ref",
    "unregistered_report_artifact",
}


def _cancel_checkpoint(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise asyncio.CancelledError


class ResearchBudgetExceeded(RuntimeError):
    """A deterministic ADK research budget was exhausted."""


@dataclass
class ResearchBudgetController:
    """Controller-owned cumulative budget for one scientific run."""

    max_model_turns: int
    max_tool_calls: int
    max_repeated_tool_results: int
    model_turns: int = 0
    tool_calls: int = 0
    _last_tool_signature: str | None = None
    _last_result_sha256: str | None = None
    _identical_result_streak: int = 0

    @staticmethod
    def _tool_signature(tool_name: str, arguments: dict) -> str:
        normalized = {
            key: value for key, value in arguments.items() if key != "isolatedContext"
        }
        encoded = json.dumps(
            {"tool": tool_name, "arguments": normalized},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def record_model_turn(self, cancel_event: threading.Event | None = None) -> None:
        _cancel_checkpoint(cancel_event)
        if self.model_turns >= self.max_model_turns:
            raise ResearchBudgetExceeded(
                f"ADK research model-turn budget exceeded ({self.max_model_turns})"
            )
        self.model_turns += 1

    def record_tool_call(
        self,
        tool_name: str,
        arguments: dict,
        cancel_event: threading.Event | None = None,
    ) -> None:
        _cancel_checkpoint(cancel_event)
        signature = self._tool_signature(tool_name, arguments)
        if (
            signature == self._last_tool_signature
            and self._identical_result_streak >= self.max_repeated_tool_results
        ):
            raise ResearchBudgetExceeded(
                "ADK research no-progress budget exceeded: identical tool "
                f"call and result repeated {self._identical_result_streak} times"
            )
        if self.tool_calls >= self.max_tool_calls:
            raise ResearchBudgetExceeded(
                f"ADK research tool-call budget exceeded ({self.max_tool_calls})"
            )
        self.tool_calls += 1

    def record_tool_result(
        self,
        tool_name: str,
        arguments: dict,
        result,
        cancel_event: threading.Event | None = None,
    ) -> None:
        _cancel_checkpoint(cancel_event)
        signature = self._tool_signature(tool_name, arguments)
        result_sha256 = hashlib.sha256(
            json.dumps(result, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        if (
            signature == self._last_tool_signature
            and result_sha256 == self._last_result_sha256
        ):
            self._identical_result_streak += 1
        else:
            self._identical_result_streak = 0
        self._last_tool_signature = signature
        self._last_result_sha256 = result_sha256


def _is_noop_typography_finding(finding: Finding) -> bool:
    """Reject critic findings whose proposed typo correction changes nothing."""

    if not any(
        token in f"{finding.problem} {finding.why_it_matters}".casefold()
        for token in ("typo", "spelling", "typograph")
    ):
        return False
    quoted = re.findall(
        r"['\"]([^'\"]{1,120})['\"]",
        finding.falsification_test_or_correction,
    )
    if len(quoted) < 2:
        return False

    def normalize(value: str) -> str:
        return " ".join(value.casefold().split())

    return normalize(quoted[-2]) == normalize(quoted[-1])


def _without_noop_typography(review: VerificationReport) -> VerificationReport:
    blocking = [
        finding
        for finding in review.blocking_findings
        if not _is_noop_typography_finding(finding)
    ]
    if len(blocking) == len(review.blocking_findings):
        return review
    unsupported = [*review.unsupported_claims]
    unsupported.append(
        "The critic emitted a no-op typography correction; it was discarded "
        "deterministically and provides no evidence of display quality."
    )
    verdict = review.verdict
    if verdict == "fail" and not blocking:
        verdict = "inconclusive"
    return review.model_copy(
        update={
            "verdict": verdict,
            "blocking_findings": blocking,
            "unsupported_claims": list(dict.fromkeys(unsupported)),
        }
    )


def _merge_reviews(
    report_review: VerificationReport,
    display_review: VerificationReport,
) -> VerificationReport:
    """Combine independent article and display audits without model voting."""

    display_review = _without_noop_typography(display_review)

    rank = {
        "pass": 0,
        "pass_with_nonblocking_comments": 1,
        "inconclusive": 2,
        "fail": 3,
    }
    verdict = max(
        (report_review.verdict, display_review.verdict),
        key=lambda item: rank[item],
    )
    return VerificationReport(
        verdict=verdict,
        blocking_findings=[
            *report_review.blocking_findings,
            *display_review.blocking_findings,
        ],
        nonblocking_findings=[
            *report_review.nonblocking_findings,
            *display_review.nonblocking_findings,
        ],
        protocol_deviations=list(
            dict.fromkeys(
                [
                    *report_review.protocol_deviations,
                    *display_review.protocol_deviations,
                ]
            )
        ),
        unsupported_claims=list(
            dict.fromkeys(
                [
                    *report_review.unsupported_claims,
                    *display_review.unsupported_claims,
                ]
            )
        ),
        proposed_falsification_tests=[
            *report_review.proposed_falsification_tests,
            *display_review.proposed_falsification_tests,
        ],
        evidence_refs=list(
            dict.fromkeys([*report_review.evidence_refs, *display_review.evidence_refs])
        ),
    )


def _figures_missing_ocr(display_inputs: list[dict]) -> list[str]:
    """Return figure IDs that cannot support a text-only display audit."""

    missing = []
    for item in display_inputs:
        if item.get("kind") != "figure":
            continue
        ocr = item.get("ocr")
        if not isinstance(ocr, dict) or ocr.get("available") is not True:
            missing.append(str(item.get("display_id") or "unknown-figure"))
            continue
        if not str(ocr.get("text") or "").strip():
            missing.append(str(item.get("display_id") or "unknown-figure"))
    return missing


def _is_presentation_only_repair(
    validation: DeterministicValidation | None,
    review: VerificationReport | None,
) -> bool:
    return bool(
        validation is not None
        and validation.findings
        and all(
            finding.code in PRESENTATION_ONLY_FINDINGS
            for finding in validation.findings
        )
        and not (
            review and (review.blocking_findings or review.proposed_falsification_tests)
        )
    )


def _needs_repair(
    validation: DeterministicValidation,
    review: VerificationReport,
) -> bool:
    return bool(
        not validation.passed
        or review.verdict == "fail"
        or (review.verdict == "inconclusive" and review.blocking_findings)
    )


def _final_run_status(
    validation: DeterministicValidation,
    review: VerificationReport,
    *,
    quality_gate_exhausted: bool,
) -> str:
    if validation.passed and review.verdict == "pass":
        return "supported"
    if validation.passed and review.verdict == "pass_with_nonblocking_comments":
        return "supported_with_comments"
    if quality_gate_exhausted:
        return "requires_human_decision"
    if review.verdict == "inconclusive":
        return "inconclusive"
    return "requires_more_evidence"


def _remove_display_ids_from_claim_evidence(
    report: ScientificReport,
) -> ScientificReport:
    """Remove model-confused display IDs; only SourceRecord IDs are evidence refs."""

    display_ids = {display.display_id for display in report.displays}
    if not display_ids:
        return report
    claims = []
    changed = False
    for claim in report.claims:
        evidence_refs = [
            reference
            for reference in claim.evidence_refs
            if reference not in display_ids
        ]
        changed = changed or evidence_refs != claim.evidence_refs
        claims.append(claim.model_copy(update={"evidence_refs": evidence_refs}))
    return report.model_copy(update={"claims": claims}) if changed else report


def _ensure_declared_display_mentions(report: ScientificReport) -> ScientificReport:
    """Add neutral controller cross-references without changing scientific claims."""

    counters = {"figure": 0, "table": 0}
    additions: dict[str, list[str]] = {
        "methods": [],
        "results": [],
        "discussion": [],
    }
    methods_text = "\n".join(report.methods)
    section_text = {
        "methods": methods_text,
        "results": report.results,
        "discussion": report.discussion,
    }
    for display in report.displays:
        counters[display.kind] += 1
        number = counters[display.kind]
        label = "Figure" if display.kind == "figure" else "Table"
        if re.search(
            rf"\b{label}\s+{number}\b",
            section_text[display.placement],
            re.IGNORECASE,
        ):
            continue
        evidence_kind = "visual" if display.kind == "figure" else "tabular"
        additions[display.placement].append(
            f"{label} {number} presents the registered {evidence_kind} evidence "
            "for this section."
        )
    if not any(additions.values()):
        return report
    methods = [*report.methods, *additions["methods"]]
    results = " ".join([report.results, *additions["results"]]).strip()
    discussion = " ".join([report.discussion, *additions["discussion"]]).strip()
    return report.model_copy(
        update={
            "methods": methods,
            "results": results,
            "discussion": discussion,
        }
    )


def _load_report_compat(path: Path) -> ScientificReport:
    """Load pre-v0.4 reports without mutating their immutable parent files."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = str(
        payload.get("executive_summary") or "Prior report summary unavailable."
    )
    narrative = str(payload.get("narrative") or summary)
    payload.setdefault(
        "introduction",
        "This revision continues the objective and evidence scope of the parent run.",
    )
    payload.setdefault("results", narrative)
    payload.setdefault(
        "discussion",
        "The parent report is retained as immutable provenance; this revision "
        "reassesses its interpretation and stated limitations.",
    )
    payload.setdefault("conclusions", summary)
    payload.setdefault("displays", [])
    return ScientificReport.model_validate(payload)


def _load_ancestor_protocol_artifacts(
    parent_root: Path,
    runs_root: Path,
) -> tuple[tuple[ArtifactRef, ...], tuple[str, ...]]:
    """Load controller protocol locks across an immutable revision lineage."""

    runs_root = runs_root.resolve()
    current = parent_root.resolve()
    artifacts: list[ArtifactRef] = []
    dates: list[str] = []
    seen: set[Path] = set()
    while True:
        if current.parent != runs_root or not current.is_dir():
            raise ValueError("parent lineage provenance is outside this workspace")
        if current in seen:
            raise ValueError("parent lineage contains a cycle")
        seen.add(current)

        protocol_path = current / "protocol.json"
        if protocol_path.is_file():
            artifacts.append(
                ArtifactRef(
                    path=str(protocol_path.resolve()),
                    sha256=sha256_file(protocol_path),
                    description=(
                        "controller-verified ancestor protocol lock inherited by "
                        "the immutable revision"
                    ),
                )
            )
            payload = json.loads(protocol_path.read_text(encoding="utf-8"))
            locked_at = payload.get("locked_at")
            if isinstance(locked_at, str) and len(locked_at) >= 10:
                dates.append(locked_at[:10])

        lineage_path = current / "parent_lineage.json"
        if not lineage_path.is_file():
            break
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        parent_run = lineage.get("parent_run")
        if (
            not isinstance(parent_run, str)
            or not parent_run
            or Path(parent_run).name != parent_run
        ):
            raise ValueError("parent lineage contains an invalid run identifier")
        current = (runs_root / parent_run).resolve()

    return tuple(artifacts), tuple(dict.fromkeys(dates))


def _read_text_prefix(path: Path, max_bytes: int) -> tuple[str, int]:
    with path.open("rb") as handle:
        data = handle.read(max_bytes)
    return data.decode("utf-8", errors="replace"), len(data)


def _fallback_evidence_packet(
    error: Exception,
    computation: ComputationEvidence,
    retrieval: RetrievalEvidence,
) -> str:
    """Preserve bounded controller evidence after a failed trailing model turn."""

    logs = []
    log_bytes_remaining = 64 * 1024
    for record in computation.records:
        if record.status != "succeeded" or log_bytes_remaining <= 0:
            continue
        stdout, consumed = _read_text_prefix(
            Path(record.stdout_path), min(24 * 1024, log_bytes_remaining)
        )
        log_bytes_remaining -= consumed
        stderr, consumed = _read_text_prefix(
            Path(record.stderr_path), min(8 * 1024, max(0, log_bytes_remaining))
        )
        log_bytes_remaining -= consumed
        logs.append(
            {
                "execution_id": record.execution_id,
                "language": record.language,
                "stdout": stdout,
                "stderr": stderr,
            }
        )
    output_previews = []
    remaining = 64 * 1024
    for artifact in computation.artifacts:
        path = Path(artifact.path)
        if path.suffix.lower() not in {".csv", ".json", ".md", ".tsv", ".txt"}:
            continue
        content, consumed = _read_text_prefix(path, min(8192, remaining))
        output_previews.append({"path": artifact.path, "content": content})
        remaining -= consumed
        if remaining <= 0:
            break
    retrieval_previews = []
    remaining = 64 * 1024
    for value in retrieval.artifacts:
        path = Path(value)
        if (
            path.suffix.lower() not in {".json", ".md", ".txt"}
            or not path.is_file()
            or path.is_symlink()
        ):
            continue
        content, consumed = _read_text_prefix(path, min(8192, remaining))
        retrieval_previews.append({"path": str(path), "content": content})
        remaining -= consumed
        if remaining <= 0:
            break
    return json.dumps(
        {
            "research_agent_warning": (
                "The ADK research turn ended after controller-recorded evidence "
                "was collected. Use only the bounded retrieval and computation "
                "evidence below; do not infer missing observations."
            ),
            "error_type": type(error).__name__,
            "successful_execution_logs": logs,
            "generated_output_previews": output_previews,
            "retrieval_previews": retrieval_previews,
        },
        sort_keys=True,
    )


def _can_continue_after_research_error(
    *,
    repairing: bool,
    computation: ComputationEvidence,
    retrieval: RetrievalEvidence,
) -> bool:
    """Allow reporting when existing controller evidence can still be gated."""

    return bool(
        repairing or computation.successful_calls > 0 or retrieval.successful_calls > 0
    )


def _merge_retrieval_evidence(
    previous: RetrievalEvidence | None,
    current: RetrievalEvidence,
) -> RetrievalEvidence:
    if previous is None:
        return current
    return RetrievalEvidence(
        successful_calls=previous.successful_calls + current.successful_calls,
        tools=sorted(set(previous.tools) | set(current.tools)),
        urls=sorted(set(previous.urls) | set(current.urls)),
        retrieval_dates=sorted(
            set(previous.retrieval_dates) | set(current.retrieval_dates)
        ),
        artifacts=[*previous.artifacts, *current.artifacts],
    )


def _merge_computation_evidence(
    previous: ComputationEvidence | None,
    current: ComputationEvidence,
) -> ComputationEvidence:
    if previous is None:
        return current
    return ComputationEvidence(
        successful_calls=previous.successful_calls + current.successful_calls,
        records=[*previous.records, *current.records],
        artifacts=[*previous.artifacts, *current.artifacts],
    )


def _compact_computation_summary(computation: ComputationEvidence) -> dict:
    """Keep audit evidence complete without replaying non-evidence file records."""

    records = []
    for record in computation.records:
        records.append(
            {
                "execution_id": record.execution_id,
                "language": record.language,
                "code_sha256": record.code_sha256,
                "started_at": record.started_at,
                "duration_seconds": record.duration_seconds,
                "exit_code": record.exit_code,
                "status": record.status,
                "environment_locks": record.environment_locks,
                "analysis_artifacts": [
                    artifact.model_dump(mode="json")
                    for artifact in record.artifacts
                    if artifact.description == "sandbox-generated analysis artifact"
                ],
            }
        )
    return {
        "successful_calls": computation.successful_calls,
        "failed_or_denied_calls": sum(
            record.status != "succeeded" for record in computation.records
        ),
        "records": records,
        "artifacts": [
            artifact.model_dump(mode="json") for artifact in computation.artifacts
        ],
    }


def _write_attempt_bundle(
    run_dir: Path,
    attempt: int,
    report: ScientificReport,
    validation,
    review: VerificationReport,
    retrieval: RetrievalEvidence,
    computation: ComputationEvidence,
) -> None:
    root = run_dir / "attempts" / f"attempt-{attempt}"
    root.mkdir(parents=True, mode=0o700, exist_ok=True)
    write_json(root / "scientific_report.json", report)
    write_json(root / "deterministic_validation.json", validation)
    write_json(root / "gemma_review.json", review)
    write_json(root / "retrieval_evidence.json", retrieval)
    write_json(root / "computation_evidence.json", computation)


async def _audit_plan(
    settings: Settings,
    master: MasterPlan,
    on_visible_text: Callable[[str, str], None] | None = None,
) -> VerificationReport:
    return await audit_master_plan(settings, master, on_visible_text)


async def _repair_plan(
    settings: Settings,
    planning: PlanningResult,
    on_visible_text: Callable[[str, str], None] | None = None,
) -> PlanningResult:
    bundle = {
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "audit": planning.audit.model_dump(mode="json"),
        "instruction": "Correct only concrete blocking findings and preserve uncertainty.",
    }
    master = await request_structured(
        settings.qwen,
        system_prompt=PLAN_REPAIRER,
        payload=bundle,
        output_type=MasterPlan,
        temperature=0.3,
        timeout=150,
        on_visible_text=(
            (lambda text: on_visible_text("Qwen", text))
            if on_visible_text is not None
            else None
        ),
    )
    master = master.model_copy(update={"task": planning.master_plan.task})
    audit = await _audit_plan(settings, master, on_visible_text)
    lint = lint_plan(master.task, master.plan)
    status = planning_status(lint, audit)
    return PlanningResult(
        master_plan=master, audit=audit, plan_lints=[lint], status=status
    )


async def _produce_report(
    settings: Settings,
    planning: PlanningResult,
    ledger: EventLedger,
    research_budget: ResearchBudgetController,
    mcp_names: tuple[str, ...],
    include_chrome: bool,
    prior_report: ScientificReport | None = None,
    validation=None,
    review: VerificationReport | None = None,
    evidence_dir: Path | None = None,
    enable_code: bool = False,
    computation_dir: Path | None = None,
    existing_retrieval: RetrievalEvidence | None = None,
    existing_computation: ComputationEvidence | None = None,
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
    simple_mode: bool = False,
    revision_request: str | None = None,
    live_dir: Path | None = None,
    activity: ActivityCallback | None = None,
    phase_progress: Callable[[str, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[ScientificReport, RetrievalEvidence, ComputationEvidence]:
    _cancel_checkpoint(cancel_event)
    toolsets = build_mcp_toolsets(settings, mcp_names) if mcp_names else []
    workspace_tools = build_workspace_tools(settings.workspace)
    literature = LiteratureAcquirer(
        settings.workspace,
        settings.literature,
        pdf_text_extractor=RemotePdfTextExtractor(settings.sandbox),
    )
    literature_tools = build_literature_tools(literature)
    executor = None
    analysis_tools = []
    environment_tools = []
    packages_enabled = bool(
        enable_code
        and settings.environment.worker_url
        and settings.environment.worker_token
    )
    if enable_code:
        if computation_dir is None:
            raise ValueError(
                "computation_dir is required when code execution is enabled"
            )
        if prior_report is not None:
            sandbox = replace(
                settings.sandbox,
                max_calls_per_attempt=min(settings.sandbox.max_calls_per_attempt, 8),
                max_wall_seconds=min(settings.sandbox.max_wall_seconds, 120),
            )
        elif simple_mode:
            sandbox = replace(
                settings.sandbox,
                max_calls_per_attempt=min(settings.sandbox.max_calls_per_attempt, 4),
                max_wall_seconds=min(settings.sandbox.max_wall_seconds, 120),
            )
        else:
            sandbox = settings.sandbox
        executor = create_analysis_executor(
            settings.workspace,
            computation_dir,
            sandbox,
            cancel_event=cancel_event,
        )
        analysis_tools = build_analysis_tools(executor)
        if packages_enabled:
            environment_manager = EnvironmentManager(
                settings.workspace,
                settings.environment,
                computation_dir.parents[1] / "package_installations.jsonl",
                cancel_event=cancel_event,
            )
            environment_tools = build_environment_tools(environment_manager)
    policy = ToolPolicy(
        ledger=ledger,
        allowed_tools=default_allowed_tools(
            include_chrome=include_chrome,
            enable_code=enable_code,
            enable_packages=packages_enabled,
        ),
        evidence_dir=evidence_dir,
        retrieval_artifact_roots=(settings.workspace,),
        observer=(
            lambda event_type, tool_name, status: (
                activity(
                    event_type,
                    "Qwen",
                    "research",
                    f"{tool_name}: {status}",
                    None,
                )
                if activity is not None
                else None
            )
        ),
    )

    def before_research_tool(tool, args: dict, tool_context):
        name = getattr(tool, "name", type(tool).__name__)
        research_budget.record_tool_call(name, args, cancel_event)
        policy_response = policy.before_tool(tool, args, tool_context)
        if policy_response is not None:
            research_budget.record_tool_result(
                name, args, policy_response, cancel_event
            )
        return policy_response

    def after_research_tool(tool, args: dict, tool_context, tool_response):
        name = getattr(tool, "name", type(tool).__name__)
        policy_response = policy.after_tool(tool, args, tool_context, tool_response)
        research_budget.record_tool_result(
            name,
            args,
            policy_response if policy_response is not None else tool_response,
            cancel_event,
        )
        return policy_response

    repairing = prior_report is not None
    research_agent = Agent(
        name="qwen_research_repairer" if repairing else "qwen_researcher",
        model=qwen_model(
            settings,
            temperature=0.3 if repairing else 0.6,
            timeout=120 if simple_mode else 240,
        ),
        instruction=RESEARCHER,
        tools=[
            *workspace_tools,
            *literature_tools,
            *environment_tools,
            *analysis_tools,
            *toolsets,
        ],
        before_tool_callback=before_research_tool,
        after_tool_callback=after_research_tool,
        mode="chat",
        include_contents="none",
    )
    payload = {
        "task": planning.master_plan.task.model_dump(mode="json"),
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "retrieval_requirement": (
            "Typed PubMed search and article acquisition tools are available. For "
            "every biomedical or health-science analysis, search PubMed and acquire "
            "each "
            "cited PMID, and preserve its verified local Markdown/PDF paths and "
            "canonical identifiers. Additional external retrieval tools are also "
            "available. Every source URL in the report must occur in a successful "
            "tool result."
            if mcp_names
            else "Typed PubMed search and article acquisition tools are available. "
            "They are mandatory for biomedical or health-science analyses; no general web "
            "retrieval tool is configured."
        ),
        "runtime_provenance_contract": (
            "The deterministic controller writes all run artifacts and creates "
            "manifest.json with SHA-256 hashes after report review."
        ),
        "code_execution": (
            "AUTHORIZED: Python and R may run only through the offline sandbox "
            "tools. /workspace and /prior are read-only and /output is the only "
            "writable path. "
            + (
                "Missing packages may be installed only through the isolated "
                "PyPI/CRAN/Bioconductor package tools; successful installs become "
                "read-only analysis libraries."
                if packages_enabled
                else "No package-installation worker is configured."
            )
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
                "existing_retrieval_evidence": (
                    existing_retrieval.model_dump(mode="json")
                    if existing_retrieval is not None
                    else None
                ),
                "existing_computation_evidence": (
                    _compact_computation_summary(existing_computation)
                    if existing_computation is not None
                    else None
                ),
                "repair_evidence_instruction": (
                    "Reuse successful existing evidence when it resolves the findings. "
                    "Do not rerun a valid computation solely to rewrite prose or repair "
                    "claim-to-artifact links. A blocking defect in an actual figure or "
                    "reader-facing table is not a prose repair: use the analysis tools "
                    "to regenerate a corrected artifact under the same logical "
                    "/output/figures or /output/tables path, reading prior-attempt data "
                    "from /history. Verify the replacement in the successful tool "
                    "response. For display-only findings, reuse the existing numeric "
                    "values already present in the prior report, review, and research "
                    "packet and make one call that writes all corrected displays. "
                    "Read the immutable workspace input directly in that generation "
                    "call when raw points are needed; do not spend calls listing or "
                    "printing /history unless a required value is genuinely absent. "
                    "Do not repeat Python/R estimation, cross-language "
                    "reconciliation, or provenance generation. If that call fails, "
                    "make at most one direct retry from stderr. Execute only a "
                    "falsification test, display correction, or missing analysis "
                    "required by a concrete finding."
                ),
            }
        )
    if revision_request is not None:
        payload["user_revision_request"] = revision_request
        payload["revision_instruction"] = (
            "Create a new child report. The parent report and its provenance are "
            "immutable; preserve valid evidence and make only warranted changes."
        )
    presentation_only_repair = repairing and _is_presentation_only_repair(
        validation, review
    )
    visible_stream_path: Path | None = None
    visible_stream_state = {"bytes": 0, "announced": False}
    if live_dir is not None:
        live_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        visible_stream_path = live_dir / "qwen_visible_output.txt"
        visible_stream_path.write_text("", encoding="utf-8")
        visible_stream_path.chmod(0o600)

    def record_visible_text(text: str) -> None:
        if visible_stream_path is None or visible_stream_state["bytes"] >= 120_000:
            return
        encoded = text.encode("utf-8")
        remaining = 120_000 - visible_stream_state["bytes"]
        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
        with visible_stream_path.open("a", encoding="utf-8") as handle:
            handle.write(chunk)
            handle.write("\n")
        visible_stream_state["bytes"] += len(chunk.encode("utf-8")) + 1
        if activity is not None and not visible_stream_state["announced"]:
            visible_stream_state["announced"] = True
            activity(
                "model_output_stream",
                "Qwen",
                "research",
                "Visible non-thought model output is updating",
                str(visible_stream_path),
            )

    research_error: Exception | None = None
    try:
        if presentation_only_repair:
            research_packet = json.dumps(
                {
                    "repair_scope": "presentation_only",
                    "instruction": (
                        "Reuse the supplied successful computation and retrieval "
                        "evidence. Correct only report structure, registrations, "
                        "captions, references, or prose; do not create new evidence."
                    ),
                    "finding_codes": sorted(
                        {finding.code for finding in validation.findings}
                    ),
                },
                sort_keys=True,
            )
            await close_mcp_toolsets(toolsets)
        else:
            try:
                _cancel_checkpoint(cancel_event)
                research_packet = await run_text(
                    research_agent,
                    payload,
                    on_visible_text=record_visible_text,
                    on_model_turn=lambda: research_budget.record_model_turn(
                        cancel_event
                    ),
                    cancel_event=cancel_event,
                )
                _cancel_checkpoint(cancel_event)
            except Exception as exc:
                research_error = exc
                research_packet = ""
            finally:
                await close_mcp_toolsets(toolsets)
    finally:
        if executor is not None:
            executor.close()
        literature.close()
    retrieval = policy.retrieval_evidence()
    computation = executor.evidence() if executor is not None else ComputationEvidence()
    if research_error is not None:
        if not _can_continue_after_research_error(
            repairing=repairing,
            computation=computation,
            retrieval=retrieval,
        ):
            raise research_error
        ledger.append(
            "research_error_recovered",
            {
                "error_type": type(research_error).__name__,
                "successful_computations": computation.successful_calls,
                "successful_retrievals": retrieval.successful_calls,
                "repairing_existing_report": repairing,
            },
        )
        research_packet = _fallback_evidence_packet(
            research_error, computation, retrieval
        )
    if live_dir is not None:
        live_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        packet_path = live_dir / "qwen_research_packet.txt"
        packet_path.write_text(research_packet, encoding="utf-8")
        packet_path.chmod(0o600)
        if activity is not None:
            activity(
                "model_output",
                "Qwen",
                "research",
                "Visible research output is available",
                str(packet_path),
            )
    effective_retrieval = _merge_retrieval_evidence(existing_retrieval, retrieval)
    effective_computation = _merge_computation_evidence(
        existing_computation, computation
    )
    report_payload = {
        **payload,
        "research_packet": research_packet,
        "retrieval_evidence": effective_retrieval.model_dump(mode="json"),
        "computation_evidence": _compact_computation_summary(effective_computation),
        "controller_evidence": {
            "artifacts": [
                artifact.model_dump(mode="json") for artifact in controller_artifacts
            ],
            "recorded_dates": list(controller_dates),
        },
    }
    report_phase = "repair" if repairing and revision_request is None else "reporting"
    report_stream_path: Path | None = None
    report_stream_state = {"bytes": 0, "announced": False}
    if live_dir is not None:
        report_stream_path = live_dir / "qwen_report_visible_output.txt"
        report_stream_path.write_text("", encoding="utf-8")
        report_stream_path.chmod(0o600)

    def record_report_visible_text(text: str) -> None:
        if report_stream_path is None or report_stream_state["bytes"] >= 120_000:
            return
        encoded = text.encode("utf-8")
        remaining = 120_000 - report_stream_state["bytes"]
        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
        with report_stream_path.open("a", encoding="utf-8") as handle:
            handle.write(chunk)
        report_stream_state["bytes"] += len(chunk.encode("utf-8"))
        if activity is not None and not report_stream_state["announced"]:
            report_stream_state["announced"] = True
            activity(
                "model_output_stream",
                "Qwen",
                report_phase,
                "Visible non-thought article output is updating",
                str(report_stream_path),
            )

    if phase_progress is not None:
        phase_progress(
            report_phase,
            (
                "Qwen is revising the article against concrete findings"
                if report_phase == "repair"
                else "Qwen is drafting the evidence-linked scientific article"
            ),
        )
    _cancel_checkpoint(cancel_event)
    report = await request_structured(
        settings.qwen,
        system_prompt=(
            REVISION_REPORTER
            if revision_request is not None
            else REPAIRER
            if repairing
            else SIMPLE_REPORTER
            if simple_mode
            else REPORTER
        ),
        payload=report_payload,
        output_type=ScientificReport,
        temperature=0.2 if repairing else 0.4,
        # Full article-shaped reports can approach the schema token ceiling on
        # local inference. Keep the wait bounded, but do not discard a healthy
        # visible stream merely because a repair needs more than three minutes.
        timeout=90 if simple_mode else 360,
        on_visible_text=record_report_visible_text,
        cancel_event=cancel_event,
    )
    normalized_report = _ensure_declared_display_mentions(
        _remove_display_ids_from_claim_evidence(report)
    )
    if normalized_report is not report:
        ledger.append(
            "report_cross_references_normalized",
            {
                "rules": [
                    "ClaimRecord.evidence_refs accept SourceRecord IDs only",
                    "Registered displays are referenced in their declared section",
                ],
            },
        )
        report = normalized_report
    _cancel_checkpoint(cancel_event)
    if live_dir is not None:
        draft_path = live_dir / "qwen_report_draft.json"
        write_json(draft_path, report)
        if activity is not None:
            activity(
                "model_output",
                "Qwen",
                "reporting",
                "Structured article draft is available",
                str(draft_path),
            )
    return report, retrieval, computation


async def _audit_report(
    settings: Settings,
    planning: PlanningResult,
    report: ScientificReport,
    validation,
    retrieval: RetrievalEvidence,
    computation: ComputationEvidence,
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
    simple_mode: bool = False,
    cancel_event: threading.Event | None = None,
    on_visible_text: Callable[[str], None] | None = None,
    audit_outputs: dict[str, VerificationReport] | None = None,
) -> VerificationReport:
    _cancel_checkpoint(cancel_event)
    _, display_inputs = prepare_display_audit(report, computation)
    acquired_article_evidence = build_acquired_article_audit(report, retrieval)
    payload = {
        "task": planning.master_plan.task.model_dump(mode="json"),
        "master_plan": planning.master_plan.model_dump(mode="json"),
        "report": report.model_dump(mode="json"),
        "deterministic_validation": validation.model_dump(mode="json"),
        "retrieval_evidence": retrieval.model_dump(mode="json"),
        "acquired_article_evidence": acquired_article_evidence,
        "computation_evidence": _compact_computation_summary(computation),
        "controller_evidence": {
            "artifacts": [
                artifact.model_dump(mode="json") for artifact in controller_artifacts
            ],
            "recorded_dates": list(controller_dates),
        },
        "runtime_provenance_contract": (
            "After this audit, the deterministic controller writes run artifacts "
            "and generates manifest.json with SHA-256 hashes."
        ),
        "display_inputs": display_inputs,
        "visual_input_order": [
            item["display_id"] for item in display_inputs if item["kind"] == "figure"
        ],
    }
    report_result = await request_structured(
        settings.gemma,
        system_prompt=REPORT_AUDITOR,
        payload=payload,
        output_type=VerificationReport,
        temperature=settings.gemma.temperature,
        timeout=90 if simple_mode else 240,
        on_visible_text=on_visible_text,
        cancel_event=cancel_event,
    )
    if audit_outputs is not None:
        audit_outputs["gemma_report"] = report_result
    _cancel_checkpoint(cancel_event)
    if not display_inputs:
        return report_result

    display_payload = {
        "task_objective": planning.master_plan.task.objective,
        "deterministic_validation": validation.model_dump(mode="json"),
        "displays": [display.model_dump(mode="json") for display in report.displays],
        "article_context": {
            "results": report.results,
            "discussion": report.discussion,
            "conclusions": report.conclusions,
        },
        "display_inputs": display_inputs,
        "visual_input_order": [
            item["display_id"] for item in display_inputs if item["kind"] == "figure"
        ],
    }
    missing_ocr = _figures_missing_ocr(display_inputs)
    if missing_ocr:
        display_result = VerificationReport(
            verdict="inconclusive",
            unsupported_claims=[
                "Pixel-level review is unavailable and controller OCR is missing "
                "or empty for figure display(s): "
                f"{', '.join(missing_ocr)}. No display approval is inferred."
            ],
        )
    else:
        if on_visible_text is not None:
            on_visible_text("\n\n--- independent OCR/geometry display audit ---\n\n")
        try:
            display_result = await request_structured(
                settings.gemma,
                system_prompt=(
                    DISPLAY_AUDITOR
                    + """

Invocation mode (overrides any instruction above to inspect supplied rasters or
visible pixels): no raster bytes are supplied. The producer identity is withheld.
Judge figures only from controller-extracted `ocr` text and geometry metadata in
display_inputs, and judge tables only from their bounded previews. Never claim to
have viewed an image or infer a visual feature absent from those records. Return
inconclusive if the supplied OCR/geometry is insufficient for a requested check."""
                ),
                payload=display_payload,
                output_type=VerificationReport,
                temperature=settings.gemma.temperature,
                timeout=120 if simple_mode else 240,
                on_visible_text=on_visible_text,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            display_result = VerificationReport(
                verdict="inconclusive",
                unsupported_claims=[
                    "Independent display review unavailable "
                    f"({type(exc).__name__}); article review is preserved but no "
                    "display approval is inferred."
                ],
            )
    if audit_outputs is not None:
        audit_outputs["gemma_display"] = display_result
    _cancel_checkpoint(cancel_event)
    return _merge_reviews(report_result, display_result)


async def _audit_report_resilient(
    settings: Settings,
    planning: PlanningResult,
    report: ScientificReport,
    validation,
    retrieval: RetrievalEvidence,
    computation: ComputationEvidence,
    ledger: EventLedger,
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
    simple_mode: bool = False,
    live_dir: Path | None = None,
    activity: ActivityCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> VerificationReport:
    visible_stream_path: Path | None = None
    visible_stream_state = {"bytes": 0, "announced": False}
    if live_dir is not None:
        live_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        visible_stream_path = live_dir / "gemma_visible_output.txt"
        visible_stream_path.write_text("", encoding="utf-8")
        visible_stream_path.chmod(0o600)

    def record_visible_text(text: str) -> None:
        current_bytes = int(visible_stream_state["bytes"])
        if visible_stream_path is None or current_bytes >= 120_000:
            return
        encoded = text.encode("utf-8")
        remaining = 120_000 - current_bytes
        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
        with visible_stream_path.open("a", encoding="utf-8") as handle:
            handle.write(chunk)
        visible_stream_state["bytes"] = current_bytes + len(chunk.encode("utf-8"))
        if activity is not None and not visible_stream_state["announced"]:
            visible_stream_state["announced"] = True
            activity(
                "model_output_stream",
                "Gemma",
                "scientific-review",
                "Visible non-thought model output is updating",
                str(visible_stream_path),
            )

    try:
        audit_outputs: dict[str, VerificationReport] = {}
        result = await _audit_report(
            settings,
            planning,
            report,
            validation,
            retrieval,
            computation,
            controller_artifacts,
            controller_dates,
            simple_mode=simple_mode,
            cancel_event=cancel_event,
            on_visible_text=record_visible_text,
            audit_outputs=audit_outputs,
        )
        if live_dir is not None:
            live_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            report_review = audit_outputs.get("gemma_report", result)
            report_review_path = live_dir / "gemma_report_review.json"
            write_json(report_review_path, report_review)
            display_review = audit_outputs.get("gemma_display")
            if display_review is not None:
                display_review_path = live_dir / "gemma_display_review.json"
                write_json(display_review_path, display_review)
            review_path = live_dir / "gemma_scientific_review.json"
            write_json(review_path, result)
            _, display_inputs = prepare_display_audit(report, computation)
            figure_text_inputs = []
            for item in display_inputs:
                if item["kind"] != "figure":
                    continue
                ocr = item.get("ocr") if isinstance(item.get("ocr"), dict) else {}
                ocr_text = str(ocr.get("text") or "")
                figure_text_inputs.append(
                    {
                        "display_id": item["display_id"],
                        "sha256": item["sha256"],
                        "media_type": item["media_type"],
                        "width": item["width"],
                        "height": item["height"],
                        "ocr_available": ocr.get("available") is True,
                        "ocr_character_count": len(ocr_text),
                        "ocr_text_sha256": (
                            hashlib.sha256(ocr_text.encode("utf-8")).hexdigest()
                            if ocr_text
                            else None
                        ),
                        "geometry_available": bool(ocr.get("words")),
                    }
                )
            table_previews = [
                {
                    "display_id": item["display_id"],
                    "sha256": item["sha256"],
                    "total_rows": item["total_rows"],
                    "total_columns": item["total_columns"],
                    "truncated": item["truncated"],
                }
                for item in display_inputs
                if item["kind"] == "table"
            ]
            display_audit_path: Path | None = None
            if display_review is not None:
                missing_ocr = _figures_missing_ocr(display_inputs)
                display_audit_path = live_dir.parent / "gemma_display_audit.json"
                write_json(
                    display_audit_path,
                    {
                        "audited_at": utc_now(),
                        "critic_model": (None if missing_ocr else settings.gemma.model),
                        "review_source": (
                            "controller_gate" if missing_ocr else "gemma_text_critic"
                        ),
                        "review_mode": "ocr_text_and_geometry",
                        "verdict": display_review.verdict,
                        "figures_missing_ocr": missing_ocr,
                        "figure_text_inputs": figure_text_inputs,
                        "table_previews": table_previews,
                    },
                )
            if activity is not None:
                activity(
                    "model_output",
                    "Gemma",
                    "scientific-review",
                    f"Independent review is available ({result.verdict})",
                    str(review_path),
                )
                if display_audit_path is not None:
                    activity(
                        "artifact_ready",
                        "Controller",
                        "scientific-review",
                        "Gemma OCR/geometry and table audit inputs were recorded",
                        str(display_audit_path),
                    )
        return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error_type = type(exc).__name__
        ledger.append("independent_critic_unavailable", {"error_type": error_type})
        return VerificationReport(
            verdict="inconclusive",
            unsupported_claims=[
                "Independent scientific review unavailable "
                f"({error_type}); no approval inferred."
            ],
        )


def _requires_pubmed_literature(task: TaskSpec) -> bool:
    if task.task_type == "software_engineering":
        return False
    context = f"{task.scientific_domain} {task.objective}".casefold()
    words = set(re.findall(r"[a-z0-9]+", context))
    biomedical_words = {
        "biomedical",
        "medicine",
        "medical",
        "clinical",
        "health",
        "patient",
        "patients",
        "disease",
        "diseases",
        "cancer",
        "tumor",
        "tumors",
        "tumour",
        "tumours",
        "gene",
        "genes",
        "genome",
        "genomes",
        "genomic",
        "genomics",
        "genetic",
        "genetics",
        "protein",
        "proteins",
        "rna",
        "drug",
        "drugs",
        "therapy",
        "therapies",
        "diagnosis",
        "diagnoses",
        "diagnostic",
        "diabetes",
        "diabetic",
        "hypertension",
        "hypertensive",
        "cardiovascular",
        "cardiology",
        "cardiac",
        "neurology",
        "neurological",
        "infection",
        "infections",
        "infectious",
        "pathogen",
        "pathogens",
        "immunotherapy",
        "biomarker",
        "biomarkers",
        "vaccine",
        "vaccines",
        "mortality",
        "mouse",
        "mice",
        "tissue",
        "tissues",
        "cells",
        "epidemiology",
        "epidemiological",
        "oncology",
        "biomedicine",
        "biology",
        "biological",
        "microbiology",
        "neuroscience",
        "pathology",
        "physiology",
        "immunology",
        "pharmacology",
        "pharmacological",
        "metabolomics",
        "proteomics",
        "transcriptomics",
        "pubmed",
    }
    biomedical_phrases = (
        r"\bblood\s+pressure\b",
        r"\badverse\s+events?\b",
    )
    return bool(words & biomedical_words) or any(
        re.search(pattern, context) for pattern in biomedical_phrases
    )


def _prepare_task_spec(objective: str, *, enable_code: bool) -> TaskSpec:
    """Bind inferred computation requirements to the run's authorization."""

    task = normalize_task(objective)
    constraints = [
        constraint
        for constraint in task.constraints
        if not constraint.startswith("Read-only MVP")
    ]
    if enable_code:
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
        return task.model_copy(
            update={
                "constraints": constraints,
                "acceptance_tests": acceptance_tests,
                "security_risk": "medium",
            }
        )

    constraints.append(
        "This run has no code-execution authorization; references to Python or R "
        "APIs are documentation topics and do not require runtime artifacts"
    )
    return task.model_copy(
        update={
            "constraints": constraints,
            "required_computation_languages": [],
        }
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
    on_provenance_ready: Callable[[Path], None] | None = None,
    activity: ActivityCallback | None = None,
    cancel_event: threading.Event | None = None,
    parent_provenance_dir: Path | None = None,
    revision_request: str | None = None,
) -> RunResult:
    def report_progress(phase: str, message: str) -> None:
        if progress is None:
            return
        try:
            progress(phase, message)
        except Exception:
            # Progress reporting is observational and must never change a run result.
            pass

    run_id = (
        f"{utc_now().replace(':', '').replace('+00:00', 'Z')}-{uuid.uuid4().hex[:8]}"
    )
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, mode=0o700)
    os.chmod(run_dir, 0o700)
    if on_provenance_ready is not None:
        try:
            on_provenance_ready(run_dir)
        except Exception:
            pass
    _cancel_checkpoint(cancel_event)
    ledger = EventLedger(run_dir / "tool_call_log.jsonl")
    research_budget = ResearchBudgetController(
        max_model_turns=settings.max_research_model_turns,
        max_tool_calls=settings.max_research_tool_calls,
        max_repeated_tool_results=settings.max_repeated_tool_results,
    )
    objective_bytes = objective.encode("utf-8")
    ledger.append(
        "run_started",
        {
            "run_id": run_id,
            "objective_sha256": hashlib.sha256(objective_bytes).hexdigest(),
            "objective_bytes": len(objective_bytes),
        },
    )
    write_json(
        run_dir / "input_manifest.json", build_input_manifest(settings.workspace)
    )
    write_json(
        run_dir / "environment.json",
        build_environment_snapshot(application_version=__version__),
    )
    selected_mcp = mcp_names if mcp_names is not None else settings.mcp_servers
    include_chrome = include_chrome or "chrome-devtools" in selected_mcp
    report_progress(
        "planning",
        (
            "The controller is loading the immutable parent protocol"
            if parent_provenance_dir is not None
            else "Qwen is preparing one lean plan"
            if simple_mode
            else "Qwen and Gemma are preparing independent plans"
        ),
    )
    planning_streams: dict[str, dict[str, object]] = {}

    def record_planning_visible_text(actor: str, text: str) -> None:
        key = "gemma" if actor == "Gemma" else "qwen"
        state = planning_streams.get(key)
        if state is None:
            live_dir = run_dir / "live"
            live_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
            path = live_dir / f"{key}_planning_visible_output.txt"
            path.write_text("", encoding="utf-8")
            path.chmod(0o600)
            state = {"path": path, "bytes": 0, "announced": False}
            planning_streams[key] = state
            if actor == "Gemma":
                report_progress(
                    "planning",
                    (
                        "Gemma is independently auditing the Qwen plan"
                        if simple_mode
                        else "Gemma is preparing an independent blinded plan"
                    ),
                )
        if int(state["bytes"]) >= 120_000:
            return
        encoded = text.encode("utf-8")
        remaining = 120_000 - int(state["bytes"])
        chunk = encoded[:remaining].decode("utf-8", errors="ignore")
        path = state["path"]
        assert isinstance(path, Path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(chunk)
        state["bytes"] = int(state["bytes"]) + len(chunk.encode("utf-8"))
        if activity is not None and not bool(state["announced"]):
            state["announced"] = True
            activity(
                "model_output_stream",
                actor,
                "planning",
                "Visible non-thought planning output is updating",
                str(path),
            )

    write_json(
        run_dir / "run_configuration.json",
        {
            "qwen": {"model": settings.qwen.model},
            "gemma": {"model": settings.gemma.model},
            "mcp_servers": list(selected_mcp),
            "chrome_enabled": include_chrome,
            "code_execution_enabled": enable_code,
            "package_installation_enabled": bool(
                enable_code
                and settings.environment.worker_url
                and settings.environment.worker_token
            ),
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
            "research_budget": {
                "model_turns": settings.max_research_model_turns,
                "tool_calls": settings.max_research_tool_calls,
                "identical_tool_result_streak": settings.max_repeated_tool_results,
            },
            "execution_mode": "simple" if simple_mode else "full",
            "run_kind": "revision" if parent_provenance_dir is not None else "analysis",
            "parent_run": (
                parent_provenance_dir.name
                if parent_provenance_dir is not None
                else None
            ),
        },
    )

    parent_report: ScientificReport | None = None
    parent_validation: DeterministicValidation | None = None
    parent_review: VerificationReport | None = None
    parent_retrieval: RetrievalEvidence | None = None
    parent_computation: ComputationEvidence | None = None
    lineage_artifact: ArtifactRef | None = None
    ancestor_protocol_artifacts: tuple[ArtifactRef, ...] = ()
    parent_controller_dates: tuple[str, ...] = ()
    if (parent_provenance_dir is None) != (revision_request is None):
        raise ValueError(
            "parent_provenance_dir and revision_request must be supplied together"
        )
    if parent_provenance_dir is not None:
        parent_root = parent_provenance_dir.resolve()
        if (
            not parent_root.is_dir()
            or parent_root.parent != settings.runs_dir.resolve()
        ):
            raise ValueError("parent provenance is outside this workspace")
        required_parent_files = [
            "planning_result.json",
            "scientific_report.json",
            "deterministic_validation.json",
            "gemma_review.json",
            "retrieval_evidence.json",
            "computation_evidence.json",
            "input_manifest.json",
        ]
        missing = [
            name for name in required_parent_files if not (parent_root / name).is_file()
        ]
        if missing:
            raise ValueError(
                "parent provenance is incomplete: " + ", ".join(sorted(missing))
            )
        current_inputs = json.loads(
            (run_dir / "input_manifest.json").read_text(encoding="utf-8")
        )
        parent_inputs = json.loads(
            (parent_root / "input_manifest.json").read_text(encoding="utf-8")
        )
        if current_inputs.get("files") != parent_inputs.get("files"):
            raise ValueError(
                "workspace inputs changed after the parent run; start a new analysis "
                "instead of revising the old evidence record"
            )
        planning = PlanningResult.model_validate_json(
            (parent_root / "planning_result.json").read_text(encoding="utf-8")
        )
        parent_report = _load_report_compat(parent_root / "scientific_report.json")
        parent_validation = DeterministicValidation.model_validate_json(
            (parent_root / "deterministic_validation.json").read_text(encoding="utf-8")
        )
        parent_review = VerificationReport.model_validate_json(
            (parent_root / "gemma_review.json").read_text(encoding="utf-8")
        )
        parent_retrieval = RetrievalEvidence.model_validate_json(
            (parent_root / "retrieval_evidence.json").read_text(encoding="utf-8")
        )
        parent_computation = ComputationEvidence.model_validate_json(
            (parent_root / "computation_evidence.json").read_text(encoding="utf-8")
        )
        lineage_path = run_dir / "parent_lineage.json"
        lineage_names = [
            "planning_result.json",
            "protocol.json",
            "scientific_report.json",
            "deterministic_validation.json",
            "gemma_review.json",
            "retrieval_evidence.json",
            "computation_evidence.json",
        ]
        write_json(
            lineage_path,
            {
                "parent_run": parent_root.name,
                "revision_request_sha256": hashlib.sha256(
                    (revision_request or "").encode("utf-8")
                ).hexdigest(),
                "parent_artifacts": {
                    name: sha256_file(parent_root / name)
                    for name in lineage_names
                    if (parent_root / name).is_file()
                },
            },
        )
        lineage_artifact = ArtifactRef(
            path=str(lineage_path.resolve()),
            sha256=sha256_file(lineage_path),
            description="controller-verified immutable parent report lineage",
        )
        (
            ancestor_protocol_artifacts,
            parent_controller_dates,
        ) = _load_ancestor_protocol_artifacts(parent_root, settings.runs_dir)
        if activity is not None:
            activity(
                "revision_parent_loaded",
                "Controller",
                "planning",
                "Immutable parent protocol and evidence were loaded",
                str(lineage_path),
            )
    else:
        task = _prepare_task_spec(objective, enable_code=enable_code)
        planning_input = task.model_dump_json()
        if simple_mode:
            planning = await build_simple_planning(
                settings, task, record_planning_visible_text
            )
        else:
            workflow = build_planning_workflow(settings, record_planning_visible_text)
            planning = await run_typed(workflow, planning_input, PlanningResult)
        _cancel_checkpoint(cancel_event)
        plan_repair_history = []
        while (
            planning.status == "requires_revision"
            and len(plan_repair_history) < settings.max_repair_rounds
        ):
            round_number = len(plan_repair_history) + 1
            previous = planning
            report_progress(
                "plan-review",
                f"Repairing concrete plan findings (round {round_number})",
            )
            ledger.append(
                "plan_repair_started",
                {"round": round_number, "reason": planning.audit.verdict},
            )
            planning = await _repair_plan(
                settings, planning, record_planning_visible_text
            )
            plan_repair_history.append(
                {
                    "round": round_number,
                    "rejected_planning": previous.model_dump(mode="json"),
                    "repaired_status": planning.status,
                    "repaired_audit": planning.audit.model_dump(mode="json"),
                    "repaired_lints": [
                        item.model_dump(mode="json") for item in planning.plan_lints
                    ],
                }
            )
            ledger.append(
                "plan_repair_finished",
                {"round": round_number, "status": planning.status},
            )
            _cancel_checkpoint(cancel_event)
        if plan_repair_history:
            write_json(run_dir / "plan_repair_history.json", plan_repair_history)
    write_json(run_dir / "planning_result.json", planning)
    if activity is not None:
        activity(
            "artifact_ready",
            "Controller",
            "planning",
            "Independent planning result is available",
            str(run_dir / "planning_result.json"),
        )
    protocol_path = run_dir / "protocol.json"
    protocol_locked_at = utc_now()
    write_json(
        protocol_path,
        {
            "locked_at": protocol_locked_at,
            "task": planning.master_plan.task,
            "plan": planning.master_plan.plan,
            "audit": planning.audit,
            "status": planning.status,
            "run_kind": "revision" if parent_report is not None else "analysis",
            "revision_scope_sha256": (
                hashlib.sha256((revision_request or "").encode("utf-8")).hexdigest()
                if revision_request is not None
                else None
            ),
        },
    )
    protocol_artifact = ArtifactRef(
        path=str(protocol_path.resolve()),
        sha256=sha256_file(protocol_path),
        description="controller protocol lock written before research execution",
    )
    controller_artifacts = tuple(
        artifact
        for artifact in (
            protocol_artifact,
            *ancestor_protocol_artifacts,
            lineage_artifact,
        )
        if artifact is not None
    )
    controller_dates = tuple(
        dict.fromkeys((protocol_locked_at[:10], *parent_controller_dates))
    )

    if planning.status != "supported":
        report_progress(
            "stopped", "Planning did not produce an evidence-ready protocol"
        )
        result = RunResult(
            run_id=run_id,
            status="inconclusive"
            if planning.status == "inconclusive"
            else "requires_more_evidence",
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
        research_budget,
        selected_mcp,
        include_chrome,
        prior_report=parent_report,
        validation=parent_validation,
        review=parent_review,
        evidence_dir=run_dir / "evidence" / "attempt-0",
        enable_code=enable_code,
        computation_dir=run_dir / "computations" / "attempt-0",
        existing_retrieval=parent_retrieval,
        existing_computation=parent_computation,
        controller_artifacts=controller_artifacts,
        controller_dates=controller_dates,
        simple_mode=simple_mode,
        revision_request=revision_request,
        live_dir=run_dir / "live",
        activity=activity,
        phase_progress=report_progress,
        cancel_event=cancel_event,
    )
    retrieval = _merge_retrieval_evidence(parent_retrieval, retrieval)
    computation = _merge_computation_evidence(parent_computation, computation)
    report_progress("validation", "Running deterministic claim and artifact checks")
    required_languages = tuple(planning.master_plan.task.required_computation_languages)
    objective_lower = objective.lower()
    require_reconciliation = {"python", "r"}.issubset(required_languages) and any(
        marker in objective_lower
        for marker in ("reconcil", "cross-check", "crosscheck", "cross-language")
    )
    require_pubmed_literature = _requires_pubmed_literature(planning.master_plan.task)
    validation = validate_report(
        report,
        retrieval,
        computation,
        required_languages=required_languages,
        require_reconciliation=require_reconciliation,
        require_pubmed_literature=require_pubmed_literature,
        controller_artifacts=controller_artifacts,
        controller_dates=controller_dates,
    )
    report_progress("scientific-review", "Gemma is independently auditing the result")
    review = await _audit_report_resilient(
        settings,
        planning,
        report,
        validation,
        retrieval,
        computation,
        ledger,
        controller_artifacts,
        controller_dates,
        simple_mode,
        run_dir / "live",
        activity,
        cancel_event,
    )
    _write_attempt_bundle(
        run_dir, 0, report, validation, review, retrieval, computation
    )
    repair_rounds = 0
    repair_generation_failed = False
    while (
        _needs_repair(validation, review) and repair_rounds < settings.max_repair_rounds
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
        try:
            report, repair_retrieval, repair_computation = await _produce_report(
                settings,
                planning,
                ledger,
                research_budget,
                selected_mcp,
                include_chrome,
                prior_report=report,
                validation=validation,
                review=review,
                evidence_dir=run_dir / "evidence" / f"attempt-{repair_rounds}",
                enable_code=enable_code,
                computation_dir=(run_dir / "computations" / f"attempt-{repair_rounds}"),
                existing_retrieval=retrieval,
                existing_computation=computation,
                controller_artifacts=controller_artifacts,
                controller_dates=controller_dates,
                simple_mode=simple_mode,
                live_dir=run_dir / "live",
                activity=activity,
                phase_progress=report_progress,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            # Preserve the last independently audited report. A model failure or
            # schema-invalid repair is a scientific escalation, never permission
            # to publish the rejected candidate and never an opaque service crash.
            repair_generation_failed = True
            failure_path = run_dir / "repair_model_unavailable.json"
            write_json(
                failure_path,
                {
                    "repair_round": repair_rounds,
                    "error_type": type(exc).__name__,
                    "last_deterministic_finding_codes": sorted(
                        {finding.code for finding in validation.findings}
                    ),
                    "last_review_verdict": review.verdict,
                    "last_blocking_finding_ids": sorted(
                        {finding.finding_id for finding in review.blocking_findings}
                    ),
                    "disposition": (
                        "The repair model did not return an admissible complete "
                        "report. The prior audited report is preserved but remains "
                        "unvalidated and requires human scientific adjudication."
                    ),
                },
            )
            ledger.append(
                "report_repair_generation_failed",
                {
                    "round": repair_rounds,
                    "error_type": type(exc).__name__,
                    "prior_report_preserved": True,
                },
            )
            if activity is not None:
                activity(
                    "artifact_ready",
                    "Controller",
                    "repair",
                    "Repair output was inadmissible; the prior audited report was preserved",
                    str(failure_path),
                )
            break
        retrieval = _merge_retrieval_evidence(retrieval, repair_retrieval)
        computation = _merge_computation_evidence(computation, repair_computation)
        validation = validate_report(
            report,
            retrieval,
            computation,
            required_languages=required_languages,
            require_reconciliation=require_reconciliation,
            require_pubmed_literature=require_pubmed_literature,
            controller_artifacts=controller_artifacts,
            controller_dates=controller_dates,
        )
        report_progress("scientific-review", "Gemma is auditing the repaired result")
        review = await _audit_report_resilient(
            settings,
            planning,
            report,
            validation,
            retrieval,
            computation,
            ledger,
            controller_artifacts,
            controller_dates,
            simple_mode,
            run_dir / "live",
            activity,
            cancel_event,
        )
        _write_attempt_bundle(
            run_dir,
            repair_rounds,
            report,
            validation,
            review,
            retrieval,
            computation,
        )

    quality_gate_exhausted = bool(
        (repair_rounds >= settings.max_repair_rounds or repair_generation_failed)
        and _needs_repair(validation, review)
    )
    if quality_gate_exhausted:
        exhausted_path = run_dir / "repair_exhausted.json"
        write_json(
            exhausted_path,
            {
                "repair_rounds": repair_rounds,
                "repair_generation_failed": repair_generation_failed,
                "deterministic_finding_codes": sorted(
                    {finding.code for finding in validation.findings}
                ),
                "review_verdict": review.verdict,
                "blocking_finding_ids": sorted(
                    {finding.finding_id for finding in review.blocking_findings}
                ),
                "disposition": (
                    "The bounded automatic repair budget ended with unresolved "
                    "quality findings. The result is not validated and requires "
                    "human scientific adjudication."
                ),
            },
        )
        if activity is not None:
            activity(
                "artifact_ready",
                "Controller",
                "finalizing",
                "Automatic repair budget exhausted; unresolved findings remain blocking",
                str(exhausted_path),
            )

    status = _final_run_status(
        validation,
        review,
        quality_gate_exhausted=quality_gate_exhausted,
    )

    _cancel_checkpoint(cancel_event)
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
    display_manifest = (
        materialize_displays(run_dir, report, computation)
        if validation.passed
        else None
    )
    reference_manifest = (
        materialize_references(run_dir, report, retrieval)
        if validation.passed
        else None
    )
    (run_dir / "report.md").write_text(
        render_report_markdown(report, display_manifest, reference_manifest),
        encoding="utf-8",
    )
    write_json(run_dir / "deterministic_validation.json", validation)
    write_json(run_dir / "retrieval_evidence.json", retrieval)
    write_json(run_dir / "computation_evidence.json", computation)
    write_json(run_dir / "gemma_review.json", review)
    write_json(run_dir / "run_result.json", result)
    _cancel_checkpoint(cancel_event)
    ledger.append("run_completed", {"status": status, "repair_rounds": repair_rounds})
    build_manifest(run_dir)
    completion_messages = {
        "supported": "Validated result is ready",
        "supported_with_comments": "Validated result is ready with nonblocking limitations",
        "contradicted": "The evidence contradicts a material claim",
        "inconclusive": "The available evidence remains inconclusive",
        "requires_more_evidence": "Unresolved evidence requirements remain",
        "requires_human_decision": "Unresolved quality findings require human scientific adjudication",
    }
    report_progress("complete", completion_messages[status])
    return result


def run(objective: str, settings: Settings, **kwargs) -> RunResult:
    return asyncio.run(run_scientific_task(objective, settings, **kwargs))
