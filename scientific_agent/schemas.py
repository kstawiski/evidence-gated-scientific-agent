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
    scientific_risk: Literal["exploratory", "confirmatory", "decision_critical"] = (
        "exploratory"
    )
    required_computation_languages: list[Literal["python", "r"]] = Field(
        default_factory=list
    )
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
    resolutions: list[ResolutionRecord] = Field(max_length=6)
    method_lock_required: bool
    protocol_fields: list[str] = Field(default_factory=list, max_length=12)


class Finding(BaseModel):
    finding_id: str = Field(max_length=120)
    location: str = Field(max_length=500)
    problem: str = Field(max_length=1200)
    why_it_matters: str = Field(max_length=1200)
    evidence: str = Field(max_length=1600)
    falsification_test_or_correction: str = Field(max_length=1600)


class VerificationReport(BaseModel):
    verdict: Literal[
        "pass", "pass_with_nonblocking_comments", "fail", "inconclusive"
    ] = Field(
        description=(
            "Use fail only with at least one concrete blocking_findings entry; "
            "use pass when no defect is found and inconclusive when evidence is "
            "insufficient."
        )
    )
    blocking_findings: list[Finding] = Field(default_factory=list, max_length=200)
    nonblocking_findings: list[Finding] = Field(default_factory=list, max_length=200)
    protocol_deviations: list[str] = Field(default_factory=list, max_length=100)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=100)
    proposed_falsification_tests: list[CheckSpec] = Field(
        default_factory=list, max_length=200
    )
    evidence_refs: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def failed_verdict_has_finding(self) -> "VerificationReport":
        if self.verdict == "fail" and not self.blocking_findings:
            raise ValueError("a fail verdict requires at least one blocking finding")
        return self


PLAN_AUDIT_CRITERIA = (
    "task_method_fit",
    "leakage_and_statistical_validity",
    "validator_independence_and_falsifiability",
    "method_lock_and_protocol_adequacy",
    "reproducibility_and_unresolved_ambiguity",
)


class PlanAuditFinding(BaseModel):
    location: str = Field(max_length=500)
    plan_evidence_quote: str = Field(max_length=800)
    problem: str = Field(max_length=1200)
    why_it_matters: str = Field(max_length=1200)
    falsification_test_or_correction: str = Field(max_length=1600)


class PlanAuditReview(BaseModel):
    criterion: Literal[
        "task_method_fit",
        "leakage_and_statistical_validity",
        "validator_independence_and_falsifiability",
        "method_lock_and_protocol_adequacy",
        "reproducibility_and_unresolved_ambiguity",
    ]
    status: Literal["pass", "fail", "inconclusive"]
    finding: PlanAuditFinding | None = None

    @model_validator(mode="after")
    def finding_matches_status(self) -> "PlanAuditReview":
        if self.status == "pass" and self.finding is not None:
            raise ValueError("a passing criterion cannot contain a finding")
        if self.status != "pass" and self.finding is None:
            raise ValueError("fail and inconclusive criteria require a finding")
        return self


class PlanAuditChecklist(BaseModel):
    reviews: list[PlanAuditReview] = Field(min_length=5, max_length=5)
    nonblocking_findings: list[Finding] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def contains_each_criterion_once(self) -> "PlanAuditChecklist":
        criteria = [review.criterion for review in self.reviews]
        if len(set(criteria)) != len(criteria):
            raise ValueError("plan audit criteria must be unique")
        if set(criteria) != set(PLAN_AUDIT_CRITERIA):
            raise ValueError("plan audit must contain every required criterion")
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
    doi: str | None = None
    pmid: str | None = Field(default=None, pattern=r"^[1-9][0-9]{0,8}$")
    pmcid: str | None = Field(default=None, pattern=r"^PMC[1-9][0-9]*$")
    citekey: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")
    license: str | None = None
    rights_status: (
        Literal[
            "pmc_oa_reuse_allowed",
            "metadata_abstract_only_no_reuse_rights",
            "private_user_provided",
        ]
        | None
    ) = None
    terms_warning: str | None = None
    retracted: bool | None = None
    local_pdf_path: str | None = None
    local_markdown_path: str | None = None
    full_text_status: (
        Literal[
            "full_text_with_pdf",
            "full_text_markdown_only",
            "abstract_only",
            "verified_manual_browser_pdf",
            "unavailable",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "External literature acquisition status only; must be null when "
            "artifact_path is used for local computational evidence."
        ),
    )
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
        if self.local_pdf_path is not None and self.url is None:
            raise ValueError("a local literature PDF requires a canonical source URL")
        if self.local_markdown_path is not None and self.url is None:
            raise ValueError(
                "local literature Markdown requires a canonical source URL"
            )
        if self.full_text_status is not None and self.url is None:
            raise ValueError(
                "full-text status applies only to external literature sources"
            )
        if (
            self.full_text_status
            in {
                "full_text_with_pdf",
                "verified_manual_browser_pdf",
            }
            and self.local_pdf_path is None
        ):
            raise ValueError("the full-text status requires a local PDF path")
        if (
            self.full_text_status
            in {
                "full_text_with_pdf",
                "full_text_markdown_only",
                "abstract_only",
                "verified_manual_browser_pdf",
            }
            and self.local_markdown_path is None
        ):
            raise ValueError("the acquisition status requires a local Markdown path")
        if self.source_type in {"documentation", "dataset", "web_page", "other"}:
            literature_only = {
                "pmid": self.pmid,
                "pmcid": self.pmcid,
                "citekey": self.citekey,
                "rights_status": self.rights_status,
                "terms_warning": self.terms_warning,
                "local_pdf_path": self.local_pdf_path,
                "local_markdown_path": self.local_markdown_path,
                "full_text_status": self.full_text_status,
            }
            populated = sorted(
                field for field, value in literature_only.items() if value is not None
            )
            if populated:
                raise ValueError(
                    "generic web, documentation, dataset, and other records must "
                    "leave typed literature-acquisition fields null: "
                    + ", ".join(populated)
                )
        return self


class ClaimRecord(BaseModel):
    claim_id: str
    text: str
    claim_type: Literal[
        "observed", "computed", "literature_supported", "inference", "hypothesis"
    ]
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="SourceRecord IDs only; display IDs are not evidence sources.",
    )
    status: EvidenceStatus
    limitations: list[str] = Field(default_factory=list)


class ReportDisplay(BaseModel):
    display_id: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    kind: Literal["figure", "table"]
    placement: Literal["methods", "results", "discussion"] = "results"
    title: str = Field(min_length=3, max_length=240)
    caption: str = Field(min_length=3, max_length=3000)
    artifact_path: str = Field(min_length=1)
    claim_ids: list[str] = Field(
        default_factory=list,
        description="ClaimRecord IDs supported or illustrated by this display.",
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="SourceRecord IDs only; never place display IDs here.",
    )
    alt_text: str = Field(default="", max_length=1200)


class ScientificReport(BaseModel):
    title: str = Field(min_length=3, max_length=300)
    executive_summary: str = Field(min_length=3)
    introduction: str = Field(min_length=3)
    methods: list[str] = Field(min_length=1)
    results: str = Field(min_length=3)
    discussion: str = Field(min_length=3)
    conclusions: str = Field(min_length=3)
    displays: list[ReportDisplay] = Field(default_factory=list)
    claims: list[ClaimRecord]
    sources: list[SourceRecord]
    unresolved_issues: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    narrative: str = Field(
        default="",
        description="Legacy pre-v0.4 report field; new reports use article sections.",
    )


class DeterministicValidation(BaseModel):
    passed: bool
    findings: list[LintFinding] = Field(default_factory=list)


class ReportDiscussionResponse(BaseModel):
    answer: str = Field(min_length=3, max_length=12_000)
    evidence_refs: list[str] = Field(default_factory=list, max_length=24)
    unresolved_uncertainties: list[str] = Field(default_factory=list, max_length=12)
    suggested_revision_prompt: str | None = Field(default=None, max_length=12_000)


class VisualEvidenceObservation(BaseModel):
    artifact_path: str = Field(min_length=1, max_length=4096)
    observed_content: str = Field(min_length=3, max_length=4000)
    scientific_interpretation: str = Field(min_length=3, max_length=4000)
    concerns: list[str] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=12)


class VisualEvidenceReport(BaseModel):
    observations: list[VisualEvidenceObservation] = Field(
        default_factory=list, max_length=100
    )
    cross_artifact_findings: list[str] = Field(default_factory=list, max_length=100)
    limitations: list[str] = Field(default_factory=list, max_length=100)
    unreviewed_requests: list[str] = Field(default_factory=list, max_length=100)


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
    status: Literal["succeeded", "failed", "timed_out", "cancelled", "policy_denied"]
    stdout_path: str
    stderr_path: str
    environment_locks: dict[str, str] = Field(default_factory=dict)
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
