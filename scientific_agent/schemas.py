"""Typed contracts shared by model and deterministic workflow nodes."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class EvidenceStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    INCONCLUSIVE = "inconclusive"


class ArtifactRef(BaseModel):
    path: str
    sha256: str | None = None
    description: str = ""


class TaskSpec(BaseModel):
    task_id: str
    objective: str = Field(min_length=3)
    deliverables: list[str]
    available_inputs: list[ArtifactRef] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    scientific_domain: str = "general"
    task_type: Literal[
        "literature_review",
        "data_analysis",
        "bioinformatics_pipeline",
        "software_engineering",
        "statistical_modeling",
        "figure_generation",
        "mixed",
    ] = "mixed"
    security_risk: Literal["low", "medium", "high"] = "low"
    scientific_risk: Literal[
        "exploratory", "confirmatory", "decision_critical"
    ] = "exploratory"
    acceptance_tests: list[str]


class CheckSpec(BaseModel):
    check_id: str
    description: str
    check_type: Literal[
        "schema", "source", "calculation", "test", "reproduction", "human"
    ]
    blocking: bool = True


class ExpectedArtifact(BaseModel):
    name: str
    description: str
    media_type: str = "application/json"


class ActionSpec(BaseModel):
    action_id: str
    plan_step_id: str
    objective: str
    tool_name: str
    arguments: dict
    input_artifacts: list[ArtifactRef] = Field(default_factory=list)
    expected_outputs: list[ExpectedArtifact] = Field(default_factory=list)
    preconditions: list[CheckSpec] = Field(default_factory=list)
    validators: list[CheckSpec] = Field(default_factory=list)
    scientific_risk: Literal["low", "medium", "high"] = "low"
    security_risk: Literal["read", "workspace_write", "external", "irreversible"]
    rollback_strategy: str | None = None
    rationale_summary: str


class PlanStep(BaseModel):
    step_id: str
    objective: str
    inputs: list[str] = Field(default_factory=list, max_length=8)
    outputs: list[str] = Field(max_length=8)
    methods: list[str] = Field(max_length=8)
    validators: list[CheckSpec] = Field(max_length=8)
    stop_conditions: list[str] = Field(max_length=8)
    scientific_risk: Literal["low", "medium", "high"] = "low"
    security_risk: Literal["read", "workspace_write", "external", "irreversible"] = (
        "read"
    )


class PlanProposal(BaseModel):
    plan_label: Literal["A", "B", "MASTER"]
    objective: str
    assumptions: list[str] = Field(default_factory=list, max_length=10)
    required_data: list[str] = Field(default_factory=list, max_length=10)
    alternatives_considered: list[str] = Field(default_factory=list, max_length=10)
    foreseeable_failure_modes: list[str] = Field(default_factory=list, max_length=10)
    steps: list[PlanStep] = Field(min_length=1, max_length=5)
    expected_artifacts: list[str] = Field(max_length=10)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=10)
    estimated_resources: list[str] = Field(default_factory=list, max_length=10)


class LintFinding(BaseModel):
    code: str
    location: str
    message: str
    blocking: bool = True


class PlanLintReport(BaseModel):
    passed: bool
    findings: list[LintFinding] = Field(default_factory=list)


class PlanBundle(BaseModel):
    task: TaskSpec
    plan_a: PlanProposal
    plan_b: PlanProposal
    lint_a: PlanLintReport
    lint_b: PlanLintReport


class ResolutionRecord(BaseModel):
    issue: str
    plan_a_position: str
    plan_b_position: str
    evidence_or_rationale: str
    decision: str
    remaining_uncertainty: str = ""


class MasterPlan(BaseModel):
    task: TaskSpec
    plan: PlanProposal
    resolutions: list[ResolutionRecord]
    method_lock_required: bool
    protocol_fields: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    finding_id: str
    location: str
    problem: str
    why_it_matters: str
    evidence: str
    falsification_test_or_correction: str


class VerificationReport(BaseModel):
    verdict: Literal[
        "pass", "pass_with_nonblocking_comments", "fail", "inconclusive"
    ]
    blocking_findings: list[Finding] = Field(default_factory=list)
    nonblocking_findings: list[Finding] = Field(default_factory=list)
    protocol_deviations: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    proposed_falsification_tests: list[CheckSpec] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def failed_verdict_has_finding(self) -> "VerificationReport":
        if self.verdict == "fail" and not self.blocking_findings:
            raise ValueError("a fail verdict requires at least one blocking finding")
        return self


class PlanningResult(BaseModel):
    master_plan: MasterPlan
    audit: VerificationReport
    plan_lints: list[PlanLintReport]
    status: Literal["supported", "requires_revision", "inconclusive"]


class SourceRecord(BaseModel):
    source_id: str
    title: str
    url: HttpUrl | None = None
    artifact_path: str | None = None
    source_type: Literal[
        "primary_study",
        "review",
        "guideline",
        "documentation",
        "dataset",
        "web_page",
        "other",
    ]
    retrieved_at: str
    supporting_passage: str = Field(
        description="Short paraphrase or compliant excerpt supporting the linked claims"
    )

    @model_validator(mode="after")
    def exactly_one_evidence_location(self) -> "SourceRecord":
        if (self.url is None) == (self.artifact_path is None):
            raise ValueError("exactly one of url or artifact_path is required")
        return self


class ClaimRecord(BaseModel):
    claim_id: str
    text: str
    claim_type: Literal[
        "observed", "computed", "literature_supported", "inference", "hypothesis"
    ]
    evidence_refs: list[str] = Field(default_factory=list)
    status: EvidenceStatus
    limitations: list[str] = Field(default_factory=list)


class ScientificReport(BaseModel):
    title: str
    executive_summary: str
    methods: list[str]
    claims: list[ClaimRecord]
    sources: list[SourceRecord]
    unresolved_issues: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    narrative: str


class DeterministicValidation(BaseModel):
    passed: bool
    findings: list[LintFinding] = Field(default_factory=list)


class RetrievalEvidence(BaseModel):
    successful_calls: int = 0
    tools: list[str] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    retrieval_dates: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class ComputationRecord(BaseModel):
    execution_id: str
    language: Literal["python", "r"]
    code_sha256: str
    started_at: str
    duration_seconds: float
    exit_code: int | None = None
    status: Literal["succeeded", "failed", "timed_out", "policy_denied"]
    stdout_path: str
    stderr_path: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class ComputationEvidence(BaseModel):
    successful_calls: int = 0
    records: list[ComputationRecord] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    status: Literal[
        "supported",
        "supported_with_comments",
        "contradicted",
        "inconclusive",
        "requires_more_evidence",
        "requires_human_decision",
    ]
    planning: PlanningResult
    report: ScientificReport | None = None
    deterministic_validation: DeterministicValidation | None = None
    retrieval_evidence: RetrievalEvidence | None = None
    computation_evidence: ComputationEvidence | None = None
    scientific_review: VerificationReport | None = None
    repair_rounds: int = 0
    provenance_dir: str
