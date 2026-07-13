"""Deterministic plan and claim-evidence checks."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .schemas import (
    ArtifactRef,
    DeterministicValidation,
    ComputationEvidence,
    LintFinding,
    PlanLintReport,
    PlanProposal,
    RetrievalEvidence,
    ScientificReport,
    TaskSpec,
)


_WORD = re.compile(r"[a-z0-9]{4,}")
_PROTOCOL_TIMING = re.compile(
    r"\b(?:lock(?:ed|ing)?|prespecif(?:ied|ication))\b.{0,100}"
    r"\b(?:before|prior to)\b.{0,100}\b(?:inspect(?:ion|ing)?|outcome|result)",
    re.IGNORECASE,
)
_METHODOLOGICAL_GENERALIZATION = re.compile(
    r"\b(?:robust to|valid despite|known to|generally reliable|assumption violation)",
    re.IGNORECASE,
)


def _terms(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def lint_plan(task: TaskSpec, plan: PlanProposal) -> PlanLintReport:
    findings: list[LintFinding] = []
    step_ids = [step.step_id for step in plan.steps]
    if len(step_ids) != len(set(step_ids)):
        findings.append(
            LintFinding(
                code="duplicate_step_id",
                location="steps",
                message="Plan step IDs must be unique.",
            )
        )

    for index, step in enumerate(plan.steps):
        location = f"steps[{index}]"
        if not step.validators:
            findings.append(
                LintFinding(
                    code="missing_validator",
                    location=location,
                    message="Every step must declare at least one validator.",
                )
            )
        if not step.stop_conditions:
            findings.append(
                LintFinding(
                    code="missing_stop_condition",
                    location=location,
                    message="Every step must declare a stopping condition.",
                )
            )
        if step.security_risk == "irreversible":
            findings.append(
                LintFinding(
                    code="irreversible_action",
                    location=location,
                    message="The agent cannot execute irreversible actions.",
                )
            )

    produced_text = " ".join(
        [*plan.expected_artifacts, *(output for step in plan.steps for output in step.outputs)]
    )
    produced_terms = _terms(produced_text)
    for index, deliverable in enumerate(task.deliverables):
        wanted = _terms(deliverable)
        if wanted and not (wanted & produced_terms):
            findings.append(
                LintFinding(
                    code="unmapped_deliverable",
                    location=f"task.deliverables[{index}]",
                    message=f"No declared output appears to produce: {deliverable}",
                )
            )

    if task.scientific_risk in {"confirmatory", "decision_critical"}:
        combined = " ".join(
            [
                *plan.assumptions,
                *plan.expected_artifacts,
                *(method for step in plan.steps for method in step.methods),
            ]
        ).lower()
        if not any(term in combined for term in ("protocol", "preregister", "method lock")):
            findings.append(
                LintFinding(
                    code="missing_method_lock",
                    location="plan",
                    message="Confirmatory work requires a protocol or method lock before results.",
                )
            )

    return PlanLintReport(passed=not any(f.blocking for f in findings), findings=findings)


def _normalize_url(value: str) -> str:
    return value.rstrip("/")


def _reject_nonfinite_json(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def validate_report(
    report: ScientificReport,
    retrieval: RetrievalEvidence | None = None,
    computation: ComputationEvidence | None = None,
    required_languages: tuple[str, ...] = (),
    require_reconciliation: bool = False,
    controller_artifacts: tuple[ArtifactRef, ...] = (),
    controller_dates: tuple[str, ...] = (),
) -> DeterministicValidation:
    findings: list[LintFinding] = []
    records = computation.records if computation else []
    for record in records:
        if record.status != "succeeded":
            continue
        for artifact in record.artifacts:
            path = Path(artifact.path)
            if (
                artifact.description != "sandbox-generated analysis artifact"
                or path.suffix.lower() != ".json"
            ):
                continue
            try:
                json.loads(
                    path.read_text(encoding="utf-8"),
                    parse_constant=_reject_nonfinite_json,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                findings.append(
                    LintFinding(
                        code="invalid_generated_json",
                        location=str(path),
                        message=(
                            "Generated JSON must be strict UTF-8 JSON and may not "
                            f"contain NaN or Infinity: {type(exc).__name__}."
                        ),
                    )
                )
    for language in required_languages:
        successful_outputs = [
            artifact
            for record in records
            if record.language == language and record.status == "succeeded"
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
        ]
        if not successful_outputs:
            findings.append(
                LintFinding(
                    code="required_computation_language_missing",
                    location="computation_evidence",
                    message=(
                        f"The locked task requires {language}, but no successful "
                        "execution from that language produced an analysis artifact."
                    ),
                )
            )
    if require_reconciliation:
        candidates = [
            artifact
            for record in records
            if record.status == "succeeded"
            for artifact in record.artifacts
            if artifact.description == "sandbox-generated analysis artifact"
            and any(
                marker in Path(artifact.path).name.lower()
                for marker in ("reconciliation", "crosscheck", "cross-check")
            )
        ]
        if not candidates:
            findings.append(
                LintFinding(
                    code="required_reconciliation_artifact_missing",
                    location="computation_evidence",
                    message=(
                        "The locked cross-language task requires a generated "
                        "machine-readable reconciliation artifact."
                    ),
                )
            )
        else:
            verdicts: list[bool] = []
            for artifact in candidates:
                path = Path(artifact.path)
                if path.suffix.lower() != ".json" or not path.is_file():
                    continue
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(value, dict):
                    continue
                for key in (
                    "all_pass",
                    "passed",
                    "within_tolerance",
                    "reconciliation_passed",
                ):
                    verdict = value.get(key)
                    if isinstance(verdict, bool):
                        verdicts.append(verdict)
                        break
            if not verdicts:
                findings.append(
                    LintFinding(
                        code="reconciliation_artifact_invalid",
                        location="computation_evidence",
                        message=(
                            "The reconciliation artifact must be JSON with a top-level "
                            "boolean all_pass, passed, within_tolerance, or "
                            "reconciliation_passed verdict."
                        ),
                    )
                )
            elif not all(verdicts):
                findings.append(
                    LintFinding(
                        code="cross_language_reconciliation_failed",
                        location="computation_evidence",
                        message=(
                            "At least one generated reconciliation artifact reports "
                            "that the prespecified tolerance was not met."
                        ),
                    )
                )
    source_ids = [source.source_id for source in report.sources]
    if len(source_ids) != len(set(source_ids)):
        findings.append(
            LintFinding(
                code="duplicate_source_id",
                location="sources",
                message="Source IDs must be unique.",
            )
        )
    known = set(source_ids)
    sources_by_id = {source.source_id: source for source in report.sources}
    claim_ids = [claim.claim_id for claim in report.claims]
    if len(claim_ids) != len(set(claim_ids)):
        findings.append(
            LintFinding(
                code="duplicate_claim_id",
                location="claims",
                message="Claim IDs must be unique.",
            )
        )

    provenance_text = " ".join(
        [report.executive_summary, report.narrative, *report.limitations]
    ).lower()
    if re.search(
        r"\b(?:hash(?:es|ing)?|manifest|provenance)\b.{0,100}"
        r"\b(?:deferred|unavailable|not generated|cannot be generated)\b",
        provenance_text,
    ):
        findings.append(
            LintFinding(
                code="false_provenance_deferral",
                location="report",
                message=(
                    "The controller always generates the provenance manifest; "
                    "the report must not claim hashing is deferred or unavailable."
                ),
            )
        )

    for index, claim in enumerate(report.claims):
        location = f"claims[{index}]"
        missing = sorted(set(claim.evidence_refs) - known)
        if missing:
            findings.append(
                LintFinding(
                    code="unknown_evidence_ref",
                    location=location,
                    message=f"Claim references unknown sources: {', '.join(missing)}",
                )
            )
        if (
            claim.claim_type not in {"hypothesis"}
            and claim.status.value in {"supported", "partially_supported"}
            and not claim.evidence_refs
        ):
            findings.append(
                LintFinding(
                    code="supported_without_evidence",
                    location=location,
                    message="A supported non-hypothesis claim must cite evidence.",
                )
            )
        if claim.claim_type == "hypothesis" and claim.status.value == "supported":
            findings.append(
                LintFinding(
                    code="hypothesis_marked_supported",
                    location=location,
                    message="A hypothesis must not be labeled supported without reclassification.",
                )
            )

        if claim.status.value in {
            "supported",
            "partially_supported",
        }:
            referenced_sources = [
                sources_by_id[source_id]
                for source_id in claim.evidence_refs
                if source_id in sources_by_id
            ]
            if claim.claim_type == "computed" and _PROTOCOL_TIMING.search(claim.text):
                findings.append(
                    LintFinding(
                        code="protocol_timing_not_computed",
                        location=location,
                        message=(
                            "A sandbox-generated analysis artifact cannot establish "
                            "that the protocol was locked before outcome inspection; "
                            "describe controller protocol provenance in Methods instead."
                        ),
                    )
                )
            if (
                claim.claim_type == "inference"
                and _METHODOLOGICAL_GENERALIZATION.search(claim.text)
                and not any(source.url for source in referenced_sources)
            ):
                findings.append(
                    LintFinding(
                        code="methodological_generalization_without_source",
                        location=location,
                        message=(
                            "A general claim about method robustness or validity needs "
                            "retrieved literature evidence, not only a computation artifact; "
                            "otherwise preserve it as an unresolved limitation."
                        ),
                    )
                )
            if claim.claim_type == "computed" and referenced_sources and not any(
                source.artifact_path for source in referenced_sources
            ):
                findings.append(
                    LintFinding(
                        code="computed_without_artifact",
                        location=location,
                        message="A computed claim must cite a sandbox-generated artifact.",
                    )
                )
            if claim.claim_type == "literature_supported" and referenced_sources and not any(
                source.url for source in referenced_sources
            ):
                findings.append(
                    LintFinding(
                        code="literature_without_url",
                        location=location,
                        message="A literature-supported claim must cite a retrieved URL.",
                    )
                )
            for source_id in claim.evidence_refs:
                source = sources_by_id.get(source_id)
                if source is None:
                    continue
                if source.url is not None:
                    if retrieval is None or retrieval.successful_calls == 0:
                        findings.append(
                            LintFinding(
                                code="supported_without_retrieval",
                                location=location,
                                message=(
                                    "A claim with an external source requires a "
                                    "successful retrieval tool call."
                                ),
                            )
                        )
                    retrieved_urls = {_normalize_url(url) for url in retrieval.urls}
                    source_url = _normalize_url(str(source.url))
                    if source_url not in retrieved_urls:
                        findings.append(
                            LintFinding(
                                code="source_url_not_retrieved",
                                location=f"{location}.evidence_refs",
                                message=(
                                    "Source URL was not present in retrieval output: "
                                    f"{source_url}"
                                ),
                            )
                        )
                    if retrieval.retrieval_dates and not any(
                        source.retrieved_at.startswith(date)
                        for date in retrieval.retrieval_dates
                    ):
                        findings.append(
                            LintFinding(
                                code="source_retrieval_date_mismatch",
                                location=f"sources[{source_id}].retrieved_at",
                                message=(
                                    "Source retrieval date does not match any recorded "
                                    f"tool-call date: {source.retrieved_at}"
                                ),
                            )
                        )
                elif source.artifact_path is not None:
                    artifact_path = os.path.normpath(source.artifact_path)
                    controller_paths = {
                        os.path.normpath(artifact.path)
                        for artifact in controller_artifacts
                    }
                    is_controller_artifact = artifact_path in controller_paths
                    if (
                        not is_controller_artifact
                        and (computation is None or computation.successful_calls == 0)
                    ):
                        findings.append(
                            LintFinding(
                                code="supported_without_computation",
                                location=location,
                                message=(
                                    "A claim with an artifact source requires a "
                                    "successful sandbox computation."
                                ),
                            )
                        )
                    known_artifacts = {
                        os.path.normpath(artifact.path)
                        for artifact in (computation.artifacts if computation else [])
                    } | controller_paths
                    if artifact_path not in known_artifacts:
                        findings.append(
                            LintFinding(
                                code="source_artifact_not_generated",
                                location=f"{location}.evidence_refs",
                                message=(
                                    "Source artifact was not produced by a successful "
                                    f"sandbox run: {source.artifact_path}"
                                ),
                            )
                        )
                    evidence_dates = (
                        set(controller_dates)
                        if is_controller_artifact
                        else {
                            record.started_at[:10]
                            for record in (computation.records if computation else [])
                            if record.status == "succeeded"
                        }
                    )
                    if evidence_dates and not any(
                        source.retrieved_at.startswith(date) for date in evidence_dates
                    ):
                        findings.append(
                            LintFinding(
                                code=(
                                    "source_controller_date_mismatch"
                                    if is_controller_artifact
                                    else "source_computation_date_mismatch"
                                ),
                                location=f"sources[{source_id}].retrieved_at",
                                message=(
                                    "Artifact evidence date does not match its recorded "
                                    f"controller or computation date: {source.retrieved_at}"
                                ),
                            )
                        )

    return DeterministicValidation(
        passed=not any(f.blocking for f in findings), findings=findings
    )
