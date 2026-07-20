import asyncio
import hashlib
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

import scientific_agent.orchestrator as orchestrator_module
from scientific_agent.config import Settings
from scientific_agent.orchestrator import (
    ScientificToolOrderGate,
    _audit_report_resilient,
    _can_continue_after_research_error,
    _compact_computation_summary,
    _final_run_status,
    _is_presentation_only_repair,
    _load_ancestor_protocol_artifacts,
    _load_retrieval_compat,
    _ensure_declared_display_mentions,
    _fallback_evidence_packet,
    _merge_computation_evidence,
    _merge_reviews,
    _needs_repair,
    _merge_retrieval_evidence,
    _normalize_inline_citation_provenance,
    _without_ocr_contradicted_typography,
    _without_inline_citation_policy_conflicts,
    _without_validation_conflicts,
    _prepare_task_spec,
    _prepare_revision_task_spec,
    _register_computation_path_evidence,
    _research_continuation_payload,
    _requires_current_run_computation,
    _revision_required_new_display_kinds,
    _revision_requires_any_new_display,
    _revision_requests_new_analysis,
    _remove_display_ids_from_claim_evidence,
    _requires_pubmed_literature,
    _write_attempt_bundle,
    ResearchBudgetController,
    ResearchBudgetExceeded,
)
from scientific_agent.provenance import EventLedger, sha256_file
from scientific_agent.knowledge import chunk_text
from scientific_agent.schemas import (
    ArtifactRef,
    CheckSpec,
    ComputationEvidence,
    ComputationRecord,
    DeterministicValidation,
    Finding,
    InlineCitation,
    MasterPlan,
    LintFinding,
    KnowledgeVisualEvidence,
    PLAN_AUDIT_CRITERIA,
    PlanLintReport,
    PlanAuditChecklist,
    PlanAuditFinding,
    PlanAuditReview,
    PlanningResult,
    PlanProposal,
    PlanStep,
    RetrievalEvidence,
    ReportDisplay,
    ScientificReport,
    ClaimRecord,
    SourceRecord,
    TaskSpec,
    VerificationReport,
    VisualEvidenceObservation,
    VisualEvidenceReport,
)
from scientific_agent.workflow import (
    DEFAULT_METHOD_LOCK_FIELDS,
    PLAN_AUDIT_MAX_PRIVATE_REASONING_BYTES_WITHOUT_FINAL,
    PLAN_AUDIT_MAX_TOKENS,
    PLAN_CRITIC_UNAVAILABLE,
    audit_master_plan,
    bind_controller_task,
    build_plan_audit_packet,
    build_simple_planning,
    build_planning_workflow,
    lint_bound_master,
    merge_and_lint,
    normalize_task,
    package_planning,
    plan_audit_to_verification,
)


def _task():
    return TaskSpec(
        task_id="t",
        objective="Produce a report",
        deliverables=["scientific report"],
        acceptance_tests=["validated"],
    )


def test_visible_tool_activity_exposes_useful_fields_without_code_or_url_secrets():
    code = "print('sensitive input')"
    message = orchestrator_module._visible_tool_call(
        "run_python_analysis",
        {"code": code, "timeout_seconds": 120, "api_key": "never-show"},
    )
    assert "run_python_analysis" in message
    assert "timeout_seconds=120" in message
    assert "code=" in message and "sha256" in message
    assert code not in message
    assert "never-show" not in message

    browser = orchestrator_module._visible_tool_call(
        "navigate_page",
        {"url": "https://example.org/article?token=never-show"},
    )
    assert "url_origin='https://example.org'" in browser
    assert "/article" not in browser
    assert "token" not in browser


def test_exact_computation_path_ref_is_registered_without_inventing_science(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text('{"estimate":5.0}\n', encoding="utf-8")
    artifact = ArtifactRef(
        path=str(result_path),
        sha256=sha256_file(result_path),
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00Z",
                duration_seconds=1,
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
        title="Artifact registration",
        executive_summary="A generated estimate was recorded.",
        introduction="This test checks structural provenance normalization.",
        methods=["A generated JSON artifact was used."],
        results="The estimate is reported.",
        discussion="Interpretation is restricted to the generated artifact.",
        conclusions="The artifact remains directly inspectable.",
        sources=[],
        claims=[
            ClaimRecord(
                claim_id="estimate",
                text="The estimate is 5.0.",
                claim_type="computed",
                evidence_refs=[str(result_path)],
                status="supported",
            )
        ],
    )

    normalized = _register_computation_path_evidence(report, computation)

    assert len(normalized.sources) == 1
    assert normalized.claims[0].evidence_refs == [normalized.sources[0].source_id]
    assert normalized.sources[0].artifact_path == str(result_path)
    assert "5.0" not in normalized.sources[0].supporting_passage

    result_path.write_text('{"estimate":999.0}\n', encoding="utf-8")
    tampered = _register_computation_path_evidence(report, computation)
    assert tampered == report


def test_unknown_path_ref_is_not_silently_registered():
    report = ScientificReport(
        title="Unknown artifact",
        executive_summary="An unknown path remains unknown.",
        introduction="This test preserves validation failures.",
        methods=["No generated computation was available."],
        results="No result is accepted.",
        discussion="The missing source remains unresolved.",
        conclusions="No claim is supported.",
        sources=[],
        claims=[
            ClaimRecord(
                claim_id="unknown",
                text="An unsupported value is 5.0.",
                claim_type="computed",
                evidence_refs=["/unknown/result.json"],
                status="unsupported",
            )
        ],
    )

    normalized = _register_computation_path_evidence(report, ComputationEvidence())

    assert normalized == report


def test_visible_tool_result_links_execution_source_without_output_content():
    message, artifact = orchestrator_module._visible_tool_result(
        "run_python_analysis",
        {
            "status": "succeeded",
            "execution_id": "exec-001",
            "duration_seconds": 1.25,
            "stdout": "sensitive output",
            "artifacts": [
                {
                    "path": "/run/exec-001/analysis.py",
                    "description": "python analysis source",
                }
            ],
        },
    )
    assert "succeeded" in message and "exec-001" in message
    assert "sensitive output" not in message
    assert artifact == "/run/exec-001/analysis.py"


def test_load_ancestor_protocol_artifacts_preserves_full_revision_lineage(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    root = runs / "root"
    child = runs / "child"
    grandchild = runs / "grandchild"
    for directory, locked_at in (
        (root, "2026-07-11T08:00:00Z"),
        (child, "2026-07-12T08:00:00Z"),
        (grandchild, "2026-07-13T08:00:00Z"),
    ):
        directory.mkdir()
        (directory / "protocol.json").write_text(
            json.dumps({"locked_at": locked_at}), encoding="utf-8"
        )
    (child / "parent_lineage.json").write_text(
        json.dumps({"parent_run": root.name}), encoding="utf-8"
    )
    (grandchild / "parent_lineage.json").write_text(
        json.dumps({"parent_run": child.name}), encoding="utf-8"
    )

    artifacts, dates = _load_ancestor_protocol_artifacts(grandchild, runs)

    assert [Path(artifact.path).parent.name for artifact in artifacts] == [
        "grandchild",
        "child",
        "root",
    ]
    assert dates == ("2026-07-13", "2026-07-12", "2026-07-11")


def test_load_retrieval_compat_reconstructs_legacy_chunk_ordinal(tmp_path):
    parent = tmp_path / "runs" / "parent"
    parent.mkdir(parents=True)
    text_path = parent / "knowledge" / "documents" / ("a" * 32) / "extracted.md"
    text_path.parent.mkdir(parents=True)
    text = "A legacy exact passage about progression-free survival."
    text_path.write_text(text, encoding="utf-8")
    chunk = chunk_text(text)[0]
    document_id = "a" * 32
    chunk_id = (
        "kc-"
        + hashlib.sha256(
            (
                f"{document_id}:{chunk['ordinal']}:{chunk['char_start']}:"
                f"{chunk['char_end']}:{chunk['sha256']}"
            ).encode()
        ).hexdigest()[:24]
    )
    passage_id = "kp-" + "b" * 24
    payload = {
        "successful_calls": 1,
        "tools": ["search_knowledge"],
        "urls": [f"https://bench.test/{passage_id}"],
        "retrieval_dates": ["2026-07-15"],
        "artifacts": [str(text_path)],
        "knowledge_snapshot_sha256": "c" * 64,
        "knowledge_passages": [
            {
                "passage_id": passage_id,
                "document_id": document_id,
                "title": "Legacy source",
                "chunk_id": chunk_id,
                "char_start": chunk["char_start"],
                "char_end": chunk["char_end"],
                "content_sha256": sha256_file(text_path),
                "chunk_sha256": chunk["sha256"],
                "source_url": f"https://bench.test/{passage_id}",
                "artifact_path": str(
                    parent / "knowledge" / "passages" / f"{passage_id}.md"
                ),
                "artifact_sha256": "d" * 64,
                "document_filename": "legacy.md",
                "document_text_path": str(text_path),
                "document_text_sha256": sha256_file(text_path),
                "document_original_path": str(parent / "knowledge" / "original.md"),
                "document_original_sha256": "e" * 64,
            }
        ],
    }
    evidence_path = parent / "retrieval_evidence.json"
    evidence_path.write_text(json.dumps(payload), encoding="utf-8")

    evidence = _load_retrieval_compat(evidence_path, parent)

    assert evidence.knowledge_passages[0].chunk_ordinal == 0


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
                validators=[
                    CheckSpec(check_id="c", description="check", check_type="source")
                ],
                stop_conditions=["done"],
            )
        ],
        expected_artifacts=["scientific report"],
    )


def _display_audit_fixture():
    task = _task().model_copy(
        update={"scientific_domain": "general science", "task_type": "data_analysis"}
    )
    planning = PlanningResult(
        master_plan=MasterPlan(
            task=task,
            plan=_plan("MASTER"),
            resolutions=[],
            method_lock_required=False,
        ),
        audit=VerificationReport(verdict="pass"),
        plan_lints=[],
        status="supported",
    )
    report = ScientificReport(
        title="Effect report",
        executive_summary="A supported effect was estimated.",
        introduction="This analysis estimates an effect.",
        methods=["A deterministic calculation was used."],
        results="Figure 1 shows the estimated effect.",
        discussion="The result remains exploratory.",
        conclusions="The evidence supports an exploratory result.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect estimate",
                caption="Effect estimate with uncertainty.",
                artifact_path="/run/output/figures/effect.png",
                alt_text="An effect plot with estimate and uncertainty.",
            )
        ],
        claims=[],
        sources=[],
    )
    return planning, report


def _pass_with_requested_display_clearances(kwargs):
    payload = kwargs.get("payload") if isinstance(kwargs, dict) else None
    required = (
        payload.get("required_clearance_refs", []) if isinstance(payload, dict) else []
    )
    return VerificationReport(verdict="pass", evidence_refs=list(required))


def _high_priority_layout_questions():
    return {
        "source": "controller_ocr_and_raster_geometry",
        "pixel_interpretation_authority": "Gemma",
        "top_text_clearance": {
            "required": True,
            "candidate_overlap_count": 5,
            "candidate_overlap_count_in_top_22_percent": 4,
            "priority": "high",
            "examples": [],
            "question": "Are all top labels mutually separated?",
        },
        "legend_data_clearance": {
            "required": True,
            "candidate": {
                "candidate_box_fraction": [0.65, 0.1, 0.98, 0.4],
                "cue_words": ["Treatment", "Control", "Observations", "Group"],
                "chromatic_pixel_fraction_beyond_key_zone": 0.02,
                "priority": "high",
            },
            "question": "Does the legend cover data or annotations?",
        },
        "annotation_data_clearance": {
            "required": True,
            "candidate_count": 1,
            "priority": "high",
            "examples": [],
            "question": "Does plotted geometry cross annotation text?",
        },
    }


def test_join_bundle_recovers_blinded_plans():
    bundle = merge_and_lint(
        {"keep_task": _task(), "planner_a": _plan("A"), "planner_b": _plan("B")}
    )
    assert bundle.plan_a.plan_label == "A"
    assert bundle.plan_b.plan_label == "B"


def test_pubmed_is_mandatory_for_biomedical_analysis_but_not_software_work():
    biomedical = _task().model_copy(
        update={
            "scientific_domain": "clinical oncology",
            "task_type": "data_analysis",
        }
    )
    software = _task().model_copy(
        update={
            "scientific_domain": "medical informatics",
            "task_type": "software_engineering",
        }
    )

    assert _requires_pubmed_literature(biomedical)
    assert not _requires_pubmed_literature(software)


@pytest.mark.parametrize(
    "objective,domain",
    [
        ("Generate a general statistical report.", "general"),
        ("Analyze generational changes in software adoption.", "general analysis"),
        ("Summarize generic benchmark results.", "data science"),
    ],
)
def test_pubmed_classifier_does_not_match_gene_inside_generic_words(objective, domain):
    task = _task().model_copy(
        update={"objective": objective, "scientific_domain": domain}
    )

    assert not _requires_pubmed_literature(task)


def test_pubmed_classifier_ignores_explicit_no_clinical_claim_instruction():
    task = _task().model_copy(
        update={
            "objective": (
                "Analyze a synthetic software validation dataset and do not make "
                "clinical claims."
            ),
            "scientific_domain": "general",
            "task_type": "mixed",
        }
    )

    assert not _requires_pubmed_literature(task)


def test_pubmed_classifier_honors_explicit_nonbiomedical_software_scope():
    task = _task().model_copy(
        update={
            "objective": (
                "Analyze validation.csv as synthetic software-validation data. "
                "It is not biomedical or clinical evidence; do not make clinical "
                "claims and do not search the literature."
            ),
            "scientific_domain": "general",
            "task_type": "mixed",
        }
    )

    assert not _requires_pubmed_literature(task)


@pytest.mark.parametrize(
    "objective,domain",
    [
        ("Estimate differential gene expression in breast cancer.", "genomics"),
        ("Analyze patient outcomes after treatment.", "clinical oncology"),
        ("Review diagnostic accuracy studies.", "health sciences"),
        ("Estimate blood pressure changes after immunotherapy.", "general"),
        ("Find biomarkers of diabetes mortality.", "general"),
        ("Measure vaccine adverse events in mice.", "general"),
        ("Compare pathogen burden across tissue samples.", "general"),
        ("Assess cardiovascular risk and hypertension.", "general"),
        ("Analyze neurological outcomes after infection.", "general"),
        ("Profile metabolomics and proteomics in cells.", "general"),
    ],
)
def test_pubmed_classifier_recognizes_biomedical_terms(objective, domain):
    task = _task().model_copy(
        update={"objective": objective, "scientific_domain": domain}
    )

    assert _requires_pubmed_literature(task)


def test_research_budget_is_cumulative_and_fails_closed():
    budget = ResearchBudgetController(2, 2, 2)

    budget.record_model_turn()
    budget.record_model_turn()
    budget.record_tool_call("search_pubmed", {"query": "cancer"})
    budget.record_tool_result("search_pubmed", {"query": "cancer"}, {"ids": ["1"]})
    budget.record_tool_call("read_text_file", {"path": "input.csv"})

    with pytest.raises(ResearchBudgetExceeded, match="model-turn budget"):
        budget.record_model_turn()
    with pytest.raises(ResearchBudgetExceeded, match="tool-call budget"):
        budget.record_tool_call("search_pubmed", {"query": "rna"})


def test_research_budget_blocks_only_identical_calls_with_identical_results():
    budget = ResearchBudgetController(8, 8, 2)
    arguments = {"query": "cancer"}

    budget.record_tool_call("search_pubmed", arguments)
    budget.record_tool_result("search_pubmed", arguments, {"ids": ["1"]})
    budget.record_tool_call("search_pubmed", arguments)
    budget.record_tool_result("search_pubmed", arguments, {"ids": ["1", "2"]})
    budget.record_tool_call("search_pubmed", arguments)
    budget.record_tool_result("search_pubmed", arguments, {"ids": ["1", "2"]})
    budget.record_tool_call("search_pubmed", arguments)
    budget.record_tool_result("search_pubmed", arguments, {"ids": ["1", "2"]})

    with pytest.raises(ResearchBudgetExceeded, match="no-progress budget"):
        budget.record_tool_call("search_pubmed", arguments)


def test_research_budget_checks_cancellation_before_limits():
    cancelled = threading.Event()
    cancelled.set()
    budget = ResearchBudgetController(1, 1, 1)

    with pytest.raises(asyncio.CancelledError):
        budget.record_model_turn(cancelled)
    with pytest.raises(asyncio.CancelledError):
        budget.record_tool_call("search_pubmed", {}, cancelled)


def test_tool_order_gate_defers_optional_pubmed_calls_until_required_code_succeeds():
    gate = ScientificToolOrderGate(frozenset({"python", "r"}))
    empty = ComputationEvidence()
    search_result = {"articles": [{"pmid": "1"}]}
    acquisition_result = {
        "source_record": {"local_markdown_path": "/workspace/references/paper.md"}
    }
    gate.record_result("search_pubmed", search_result)
    gate.record_result("acquire_pubmed_article", acquisition_result)

    denied_search = gate.before_tool("search_pubmed", empty)
    denied_acquisition = gate.before_tool("acquire_pubmed_article", empty)

    assert denied_search is not None
    assert denied_acquisition is not None
    assert denied_search["error"] == "REQUIRED_COMPUTATION_PENDING"
    assert denied_acquisition["missing_required_languages"] == ["python", "r"]
    assert gate.before_tool("run_python_analysis", empty) is None
    gate.record_result("search_pubmed", denied_search)
    assert gate.pubmed_search_attempts == 1


def test_tool_order_gate_allows_broader_zero_hit_search_and_lifts_after_code():
    gate = ScientificToolOrderGate(frozenset({"python"}))
    empty = ComputationEvidence()
    gate.record_result("search_pubmed", {"articles": []})

    assert gate.before_tool("search_pubmed", empty) is None

    successful = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="a" * 64,
                started_at="2026-07-15T00:00:00+00:00",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path="/tmp/stdout",
                stderr_path="/tmp/stderr",
            )
        ]
    )
    gate.record_result("search_pubmed", {"articles": [{"pmid": "1"}]})

    assert gate.before_tool("search_pubmed", successful) is None


def test_tool_order_gate_requires_current_child_computation_not_inherited_evidence(
    tmp_path,
):
    gate = ScientificToolOrderGate(frozenset(), require_current_computation=True)
    gate.record_result("search_pubmed", {"articles": [{"pmid": "1"}]})
    inherited = ComputationEvidence(successful_calls=1)

    denied = gate.before_tool(
        "search_pubmed", ComputationEvidence(), existing=inherited
    )

    assert denied is not None
    assert denied["error"] == "REQUIRED_COMPUTATION_PENDING"
    assert denied["current_computation_required"] is True
    result = tmp_path / "survival.json"
    result.write_text('{"hazard_ratio": 1.2}\n', encoding="utf-8")
    current = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-current",
                language="python",
                code_sha256="c" * 64,
                started_at="2026-07-19T00:00:00+00:00",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout"),
                stderr_path=str(tmp_path / "stderr"),
                artifacts=[
                    ArtifactRef(
                        path=str(result),
                        description="sandbox-generated analysis artifact",
                    )
                ],
            )
        ]
    )
    assert gate.before_tool("search_pubmed", current, existing=inherited) is None


def test_survival_code_preflight_rejects_known_invalid_or_ungrounded_patterns():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        ungrounded_categorical_columns=frozenset({"Gender", "T"}),
    )
    code = """
from lifelines import CoxPHFitter
labels = df['Gender'].map({1: 'Male', 2: 'Female'})
stage = df['T'].map({0: 'Ta', 1: 'T1'})
model = CoxPHFitter(penalizer=0.1)
median = kmf.median_survival_time
at_one = kmf.predict([1.0])[0]
limits = model.confidence_intervals_
"""

    denied = gate.before_tool(
        "run_python_analysis",
        ComputationEvidence(),
        arguments={"code": code},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    joined = " ".join(denied["issues"])
    assert "median_survival_time_" in joined
    assert "receive a scalar" in joined
    assert "coefficient-scale" in joined
    assert "do not authorize a penalized Cox" in joined
    assert "Gender has no complete input-profile codebook" in joined
    assert "T has no complete input-profile codebook" in joined


def test_survival_code_preflight_accepts_raw_labels_and_safe_lifelines_api():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        ungrounded_categorical_columns=frozenset({"T"}),
    )
    code = """
raw_stage = df['T'].map({0: 'T=0', 1: 'T=1'})
median = kmf.median_survival_time_
at_one = float(kmf.predict(1.0))
limits = model.summary[['exp(coef) lower 95%', 'exp(coef) upper 95%']]
coefficient_limits_on_ratio_scale = np.exp(model.confidence_intervals_)
"""

    assert (
        gate.before_tool(
            "run_python_analysis",
            ComputationEvidence(),
            arguments={"code": code},
        )
        is None
    )


def test_survival_code_preflight_requires_controller_grounded_time_origin():
    gate = ScientificToolOrderGate(frozenset(), survival_task=True)
    code = """
from lifelines import KaplanMeierFitter
KaplanMeierFitter().fit(df['RFS_time'], df['RFS_event'])
"""

    denied = gate.before_tool(
        "run_python_analysis",
        ComputationEvidence(),
        arguments={"code": code},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    assert "defines the time origin" in " ".join(denied["issues"])
    assert "External retrieval" in " ".join(denied["issues"])


def test_survival_code_preflight_denies_manual_cif_without_time_origin():
    gate = ScientificToolOrderGate(frozenset(), survival_task=True)
    code = """
def compute_cif(times, events, event_of_interest):
    at_risk = len(times)
    cif = np.zeros(len(times))
    cumulative_incidence = 0.0
    for index, _ in enumerate(times):
        cumulative_incidence += (events[index] == event_of_interest) / at_risk
        cif[index] = cumulative_incidence
        at_risk -= 1
    return cif
"""

    denied = gate.before_tool(
        "run_python_analysis",
        ComputationEvidence(),
        arguments={"code": code},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    assert "immutable input manifest" in " ".join(denied["issues"])


def test_survival_code_preflight_accepts_source_grounded_time_origin():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        survival_time_origin_source_paths=frozenset({"codebook.md"}),
    )
    gate.record_result(
        "search_pubmed",
        {
            "query": "time origin diagnosis",
            "articles": [{"abstract": "The study modeled recurrence-free survival."}],
        },
    )
    assert not gate.survival_time_origin_grounded
    gate.record_result(
        "read_text_file",
        {
            "path": "codebook.md",
            "content": "RFS was defined as time from initial TURBT to recurrence.",
        },
    )

    assert (
        gate.before_tool(
            "run_python_analysis",
            ComputationEvidence(),
            arguments={
                "code": "from lifelines import CoxPHFitter\nCoxPHFitter().fit(df)"
            },
        )
        is None
    )


def test_survival_code_preflight_rejects_unrelated_retrieval_time_origin():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        survival_time_origin_source_paths=frozenset({"codebook.md"}),
    )
    gate.record_result(
        "search_pubmed",
        {
            "articles": [
                {
                    "abstract": (
                        "The 60-month RFS and PFS increased from 0.39 and 0.85 "
                        "at baseline in an unrelated cohort."
                    )
                }
            ]
        },
    )
    gate.record_result(
        "read_text_file",
        {
            "path": "references/unrelated.md",
            "content": "RFS was measured from diagnosis in an external cohort.",
        },
    )

    denied = gate.before_tool(
        "run_python_analysis",
        ComputationEvidence(),
        arguments={"code": "from lifelines import CoxPHFitter\nCoxPHFitter().fit(df)"},
    )

    assert not gate.survival_time_origin_grounded
    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"


def test_r_survival_code_preflight_requires_controller_grounded_time_origin():
    gate = ScientificToolOrderGate(frozenset(), survival_task=True)

    denied = gate.before_tool(
        "run_r_analysis",
        ComputationEvidence(),
        arguments={"code": "fit <- coxph(Surv(time, event) ~ age, data=df)"},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    assert "defines the time origin" in " ".join(denied["issues"])


@pytest.mark.parametrize(
    "code, expected",
    [
        (
            'Path("/output/data").mkdir(parents=True, exist_ok=True)',
            "dir.create",
        ),
        (
            "utils::zip(zipfile=target, files=c(a, b), junk.paths=FALSE)",
            "junk.paths=",
        ),
        (
            "utils::zip(\n  zipfile=target,\n  files=c(a, b),\n  recurse=FALSE\n)",
            "recurse=",
        ),
        ('ragg::png("figure.png")', "does not export png"),
        (
            'ragg::agg_png("figure.png", width=6, height=4, units="in", dpi=320)',
            "uses res=",
        ),
        (
            "geom_errorbarh(aes(x=estimate, xmin=ci_low, xmax=ci_high))",
            "does not accept an x aesthetic",
        ),
        ("stopifnot(ci[1] < estimate < ci[2])", "does not support chained"),
        (
            'systemfonts::font_add_google("Open Sans", "OpenSans")',
            "does not export font_add_google",
        ),
        (
            "p_err <- test_result$p.value\np_err <- ggplot(results, aes(x, y))",
            "overwrites numeric p-value",
        ),
        (
            'ggplot2::annotate("text", x=grid::unit(0.9, "npc"), y=1, label="p")',
            "use data coordinates",
        ),
        (
            'plot_data <- tidyr::pivot_longer(data, cols=c(baseline_error_rate, optimized_error_rate), names_to=c("engine", "metric"), names_sep="_")',
            "multiple underscores",
        ),
        (
            'filtered <- data %>% subset(metric == "error_rate")',
            "without loading dplyr",
        ),
        (
            'jsonlite::write_json(results, "/output/data/results.json", auto_unbox=TRUE)',
            "defaults to digits=4",
        ),
        (
            'library(jsonlite)\nwrite_json(results, "/output/data/results.json", auto_unbox=TRUE)',
            "defaults to digits=4",
        ),
        (
            'write.csv(plot_data, "/output/tables/plot_data.csv", row.names=FALSE)',
            "belong below /output/data",
        ),
    ],
)
def test_r_code_preflight_rejects_known_generated_api_defects(code, expected):
    gate = ScientificToolOrderGate(frozenset())

    denied = gate.before_tool(
        "run_r_analysis", ComputationEvidence(), arguments={"code": code}
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    assert expected in " ".join(denied["issues"])


def test_r_code_preflight_accepts_supported_output_apis():
    code = """
target <- "/output/figures/result.png"
dir.create(dirname(target), recursive=TRUE, showWarnings=FALSE)
stopifnot(ci[1] < estimate && estimate < ci[2])
systemfonts::match_font("Open Sans")
p_value_error <- test_result$p.value
plot_error <- ggplot(results, aes(x=estimate, y=metric))
plot_error <- plot_error + annotate("text", x=0.1, y=1, label="p < 0.001")
ragg::agg_png(target, width=6, height=4, units="in", res=320,
              background="white")
print(plot_object)
dev.off()
utils::zip(zipfile="/output/deliverables/results.zip",
           files=c("/output/data/results.json", target))
jsonlite::write_json(results, "/output/data/results.json",
                     auto_unbox=TRUE, digits=16)
"""
    gate = ScientificToolOrderGate(frozenset())

    denied = gate.before_tool(
        "run_r_analysis", ComputationEvidence(), arguments={"code": code}
    )

    assert denied is None


def test_r_code_preflight_accepts_loaded_magrittr_pipe():
    gate = ScientificToolOrderGate(frozenset())

    denied = gate.before_tool(
        "run_r_analysis",
        ComputationEvidence(),
        arguments={"code": "library(dplyr)\nfiltered <- data %>% filter(value > 0)"},
    )

    assert denied is None


def test_r_survival_code_preflight_denies_manual_cif_without_time_origin():
    gate = ScientificToolOrderGate(frozenset(), survival_task=True)

    denied = gate.before_tool(
        "run_r_analysis",
        ComputationEvidence(),
        arguments={"code": "cif <- numeric(length(times)); at_risk <- length(times)"},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"


def test_survival_code_preflight_rejects_direct_semantic_indicators_and_origin():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        ungrounded_categorical_columns=frozenset(
            {"Gender", "T", "Grading", "No_tumors", "Diameter", "BCG"}
        ),
    )
    code = """
from lifelines.statistics import proportional_hazard_test
df_cox['Gender_female'] = (df_cox['Gender'] == 2).astype(int)
df_cox['T_1'] = (df_cox['T'] == 1).astype(int)  # Tis
df_cox['Grading_2'] = (df_cox['Grading'] == 2).astype(int)  # intermediate grade
df_cox['Multifocal'] = df_cox['No_tumors'].copy()
df_cox['Large_diameter'] = df_cox['Diameter'].copy()
df_cox['BCG_yes'] = df_cox['BCG'].copy()
try:
    diagnostics = proportional_hazard_test(model, data)
except Exception as exc:
    diagnostics = {'note': f'PH diagnostic failed: {exc}'}
results = {
    'analysis_metadata': {
        'time_origin': 'Inferred as initial TURBT; not explicitly verified'
    },
    'diagnostics': diagnostics,
}
"""

    denied = gate.before_tool(
        "run_python_analysis",
        ComputationEvidence(),
        arguments={"code": code},
    )

    assert denied is not None
    assert denied["error"] == "SCIENTIFIC_CODE_PREFLIGHT_FAILED"
    joined = " ".join(denied["issues"])
    assert "unverified time origin" in joined
    assert "Gender has no complete input-profile codebook" in joined
    assert "T has no complete input-profile codebook" in joined
    assert "No_tumors has no complete input-profile codebook" in joined
    assert "Diameter has no complete input-profile codebook" in joined
    assert "BCG has no complete input-profile codebook" in joined
    assert "proportional-hazards diagnostic failure" in joined


def test_survival_code_preflight_accepts_raw_numeric_indicator_names():
    gate = ScientificToolOrderGate(
        frozenset(),
        survival_task=True,
        ungrounded_categorical_columns=frozenset({"Gender", "T", "Grading"}),
    )
    code = """
df_cox['Gender_2'] = (df_cox['Gender'] == 2).astype(int)
df_cox['T_1'] = (df_cox['T'] == 1).astype(int)
df_cox['Grading_2'] = (df_cox['Grading'] == 2).astype(int)
results = {'time_origin': 'Verified from uploaded codebook: initial procedure'}
"""

    assert (
        gate.before_tool(
            "run_python_analysis",
            ComputationEvidence(),
            arguments={"code": code},
        )
        is None
    )


def test_non_survival_code_preflight_ignores_unrelated_time_origin_field():
    gate = ScientificToolOrderGate(frozenset(), survival_task=False)

    assert (
        gate.before_tool(
            "run_python_analysis",
            ComputationEvidence(),
            arguments={"code": "result = {'time_origin': 'unknown'}"},
        )
        is None
    )


def test_simple_tool_order_gate_caps_post_computation_pubmed_attempts():
    gate = ScientificToolOrderGate(
        frozenset(),
        max_pubmed_search_attempts=3,
        max_pubmed_acquisition_attempts=2,
    )
    for _ in range(3):
        gate.record_result("search_pubmed", {"articles": []})
    for _ in range(2):
        gate.record_result(
            "acquire_pubmed_article",
            {"source_record": {"local_markdown_path": "/references/paper.md"}},
        )

    denied_search = gate.before_tool("search_pubmed", ComputationEvidence())
    denied_acquisition = gate.before_tool(
        "acquire_pubmed_article", ComputationEvidence()
    )

    assert denied_search["error"] == "RETRIEVAL_ATTEMPT_LIMIT_REACHED"
    assert denied_acquisition["error"] == "RETRIEVAL_ATTEMPT_LIMIT_REACHED"


def test_tool_order_gate_requires_passing_cross_language_reconciliation(tmp_path):
    gate = ScientificToolOrderGate(
        frozenset({"python", "r"}), require_reconciliation=True
    )
    gate.record_result("search_pubmed", {"articles": [{"pmid": "1"}]})
    records = [
        ComputationRecord(
            execution_id=f"exec-{index:03d}",
            language=language,
            code_sha256=str(index) * 64,
            started_at="2026-07-15T00:00:00+00:00",
            duration_seconds=1,
            exit_code=0,
            status="succeeded",
            stdout_path=str(tmp_path / f"stdout-{index}"),
            stderr_path=str(tmp_path / f"stderr-{index}"),
        )
        for index, language in enumerate(("python", "r"), start=1)
    ]
    languages_only = ComputationEvidence(records=records)

    denied = gate.before_tool("search_pubmed", languages_only)

    assert denied is not None
    assert denied["missing_required_languages"] == []
    assert denied["reconciliation_required"] is True
    python_output = tmp_path / "python.json"
    r_output = tmp_path / "r.json"
    python_output.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    r_output.write_text('{"primary":{"point_estimate":5.0}}\n', encoding="utf-8")
    reconciliation = tmp_path / "cross_language_reconciliation.json"
    reconciliation.write_text(
        json.dumps(
            {
                "all_pass": True,
                "comparisons": [
                    {
                        "metric": "primary_point_estimate",
                        "python": {
                            "language": "python",
                            "artifact_sha256": sha256_file(python_output),
                            "json_path": "primary.point_estimate",
                            "value": 5.0,
                        },
                        "r": {
                            "language": "r",
                            "artifact_sha256": sha256_file(r_output),
                            "json_path": "primary.point_estimate",
                            "value": 5.0,
                        },
                        "absolute_difference": 0.0,
                        "tolerance": 1e-6,
                        "passed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    python_artifact = ArtifactRef(
        path=str(python_output),
        sha256=sha256_file(python_output),
        description="sandbox-generated analysis artifact",
    )
    r_artifact = ArtifactRef(
        path=str(r_output),
        sha256=sha256_file(r_output),
        description="sandbox-generated analysis artifact",
    )
    artifact = ArtifactRef(
        path=str(reconciliation),
        sha256=sha256_file(reconciliation),
        description="sandbox-generated analysis artifact",
    )
    reconciled = ComputationEvidence(
        records=[
            records[0].model_copy(update={"artifacts": [python_artifact, artifact]}),
            records[1].model_copy(update={"artifacts": [r_artifact]}),
        ],
        artifacts=[python_artifact, r_artifact, artifact],
    )
    assert gate.before_tool("search_pubmed", reconciled) is None

    reconciliation.write_text(
        json.dumps(
            {
                **json.loads(reconciliation.read_text(encoding="utf-8")),
                "padding": "x" * (90 * 1024),
            }
        ),
        encoding="utf-8",
    )
    assert gate.before_tool("search_pubmed", reconciled) is not None
    padded_artifact = artifact.model_copy(
        update={"sha256": sha256_file(reconciliation)}
    )
    padded = reconciled.model_copy(
        update={
            "records": [
                reconciled.records[0].model_copy(
                    update={"artifacts": [python_artifact, padded_artifact]}
                ),
                reconciled.records[1],
            ],
            "artifacts": [python_artifact, r_artifact, padded_artifact],
        }
    )
    assert gate.before_tool("search_pubmed", padded) is None


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


def _passing_plan_audit():
    return PlanAuditChecklist(
        reviews=[
            PlanAuditReview(criterion=criterion, status="pass")
            for criterion in PLAN_AUDIT_CRITERIA
        ]
    )


def test_plan_audit_packet_is_compact_blinded_and_includes_lint():
    master = MasterPlan(
        task=_task(),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )

    packet = build_plan_audit_packet(master)
    encoded = json.dumps(packet)

    assert packet["deterministic_lint"]["passed"] is True
    assert "task_id" not in encoded
    assert "plan_label" not in encoded
    assert "sha256" not in encoded
    assert "planner_a" not in encoded
    assert "planner_b" not in encoded


def test_plan_audit_controller_derives_verdict_and_falsification_test():
    master = MasterPlan(
        task=_task(),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )
    reviews = list(_passing_plan_audit().reviews)
    reviews[1] = PlanAuditReview(
        criterion=reviews[1].criterion,
        status="fail",
        finding=PlanAuditFinding(
            location="plan.steps[0].validators",
            plan_evidence_quote="check",
            problem="The validator is not independent of the proposed method.",
            why_it_matters="The same failure mode could affect result and validation.",
            falsification_test_or_correction="Add an independent recomputation.",
        ),
    )

    report = plan_audit_to_verification(
        PlanAuditChecklist(reviews=reviews), master=master
    )

    assert report.verdict == "fail"
    assert report.blocking_findings[0].location == "plan.steps[0].validators"
    assert report.proposed_falsification_tests[0].blocking is True


def test_task_normalizer_locks_explicit_computation_languages():
    task = normalize_task("Cross-check this analysis independently in Python and R.")
    assert task.required_computation_languages == ["python", "r"]


def test_task_normalizer_requires_method_lock_for_prespecified_analysis():
    task = normalize_task(
        "Analyze the dataset using a prespecified endpoint and lock the method "
        "before inspecting outcomes."
    )

    assert task.scientific_risk == "confirmatory"


def test_task_normalizer_locks_known_effect_evaluation_objective():
    task = normalize_task(
        "Analyze /workspace/known_effect.csv as a prespecified two-group pre/post "
        "study. Before inspecting group outcomes, lock a Welch two-sample t-test "
        "on change and execute it independently in Python and R."
    )

    assert task.scientific_risk == "confirmatory"


def test_task_normalizer_respects_explicit_nonconfirmatory_scope():
    task = normalize_task(
        "Analyze the dataset as exploratory and not confirmatory; generate "
        "hypotheses only."
    )

    assert task.scientific_risk == "exploratory"


def test_task_normalizer_recognizes_explicit_decision_critical_work():
    task = normalize_task(
        "Evaluate the model for a decision-critical regulatory decision."
    )

    assert task.scientific_risk == "decision_critical"


@pytest.mark.parametrize(
    "objective",
    [
        "Compare exploratory and confirmatory study designs.",
        "Perform an exploratory analysis to plan a future confirmatory trial.",
        "Analyze retrospective data to determine whether a confirmatory study is warranted.",
        "This analysis will not inform a regulatory decision.",
        "Evaluate why this is not a clinical decision.",
    ],
)
def test_task_normalizer_does_not_promote_referenced_or_negated_stakes(objective):
    assert normalize_task(objective).scientific_risk == "exploratory"


@pytest.mark.parametrize(
    "objective",
    [
        "This is not exploratory; analyze the results to guide a clinical decision.",
        "Do not delay; analyze the results to guide a clinical decision.",
        "Never ignore the result; analyze it to support a regulatory decision.",
    ],
)
def test_task_normalizer_preserves_unnegated_decision_stakes(objective):
    assert normalize_task(objective).scientific_risk == "decision_critical"


def test_secondary_nonprespecified_endpoint_does_not_downgrade_primary_analysis():
    task = normalize_task(
        "Conduct a confirmatory analysis of the primary outcome; one secondary "
        "endpoint is not prespecified."
    )

    assert task.scientific_risk == "confirmatory"


@pytest.mark.parametrize(
    "objective",
    [
        "Do not conduct a confirmatory analysis; perform exploratory hypothesis generation.",
        "Never run a confirmatory analysis; analyze descriptively.",
        "This is a non-decision-critical exploratory analysis.",
    ],
)
def test_task_normalizer_does_not_promote_negated_current_stakes(objective):
    assert normalize_task(objective).scientific_risk == "exploratory"


@pytest.mark.parametrize(
    "objective",
    [
        "Plan a future confirmatory trial, but conduct a confirmatory analysis now.",
        "Compare exploratory and confirmatory study designs; then conduct a "
        "confirmatory analysis of the current endpoint.",
        "Conduct a confirmatory evaluation of the current intervention.",
        "Run a confirmatory inference for the primary outcome.",
        "Undertake confirmatory evaluation now.",
    ],
)
def test_task_normalizer_recognizes_current_confirmatory_clause(objective):
    assert normalize_task(objective).scientific_risk == "confirmatory"


@pytest.mark.parametrize(
    "objective",
    [
        "This analysis cannot inform a clinical decision.",
        "This is not expected to be decision-critical.",
        "Do not use the prespecified endpoint; analyze an exploratory endpoint instead.",
    ],
)
def test_task_normalizer_binds_common_negation_to_the_affected_stake(objective):
    assert normalize_task(objective).scientific_risk == "exploratory"


@pytest.mark.parametrize(
    "objective",
    [
        "This analysis doesn't inform a clinical decision.",
        "We shouldn't conduct a confirmatory analysis.",
        "Do not conduct and run a confirmatory analysis.",
        "Neither inform nor guide a clinical decision.",
    ],
)
def test_task_normalizer_handles_contracted_and_coordinated_negation(objective):
    assert normalize_task(objective).scientific_risk == "exploratory"


def test_task_normalizer_does_not_treat_not_only_as_negation():
    task = normalize_task(
        "Not only conduct a confirmatory analysis; also report sensitivity analyses."
    )

    assert task.scientific_risk == "confirmatory"


@pytest.mark.parametrize(
    "objective",
    [
        "Analyze the primary endpoint, which was prespecified before data collection.",
        "The primary endpoint was prespecified before data collection; analyze it now.",
        "Analyze the endpoint specified a priori in the protocol.",
    ],
)
def test_task_normalizer_recognizes_noun_first_method_locks(objective):
    assert normalize_task(objective).scientific_risk == "confirmatory"


@pytest.mark.parametrize(
    "objective",
    [
        "Analyze the results to guide clinical decisions.",
        "Evaluate the model to guide patient-care decisions.",
        "Conduct a confirmatory analysis and use it to guide clinical decisions.",
        "Do not delay and analyze the results to guide a clinical decision.",
    ],
)
def test_task_normalizer_recognizes_plural_or_independent_decision_stakes(objective):
    assert normalize_task(objective).scientific_risk == "decision_critical"


@pytest.mark.parametrize(
    "objective",
    [
        "Do not delay and conduct a confirmatory analysis now.",
        "Perform the analysis exactly as preregistered.",
        "Execute the locked analysis plan on the primary endpoint.",
    ],
)
def test_task_normalizer_recognizes_independent_or_locked_analysis(objective):
    assert normalize_task(objective).scientific_risk == "confirmatory"


def test_task_normalizer_handles_must_not_contraction():
    assert (
        normalize_task("We mustn't conduct a confirmatory analysis.").scientific_risk
        == "exploratory"
    )


@pytest.mark.parametrize(
    "objective",
    [
        "Analyze the result to guide patient care.",
        "Evaluate the model to support treatment decisions.",
        "Conduct the locked analysis for use in clinical decision-making.",
        "Analyze the result for use in regulatory decisions.",
    ],
)
def test_task_normalizer_recognizes_common_decision_stakes(objective):
    assert normalize_task(objective).scientific_risk == "decision_critical"


@pytest.mark.parametrize(
    "objective",
    [
        "Analyze the primary outcome defined a priori.",
        "Execute the locked statistical analysis plan.",
        "Perform the analysis according to the preregistration.",
        "Execute the statistical analysis plan finalized before outcomes were reviewed.",
    ],
)
def test_task_normalizer_recognizes_common_method_lock_phrasing(objective):
    assert normalize_task(objective).scientific_risk == "confirmatory"


def test_task_normalizer_recognizes_current_confirmatory_analysis():
    task = normalize_task("Conduct a confirmatory analysis of the primary outcome.")

    assert task.scientific_risk == "confirmatory"


def test_controller_enforces_required_method_lock_and_protocol_fields():
    task = normalize_task("Analyze a prespecified primary endpoint.")
    master = MasterPlan(
        task=task.model_copy(update={"scientific_risk": "exploratory"}),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
        protocol_fields=[],
    )

    bound = bind_controller_task(master, task)

    assert bound.method_lock_required is True
    assert bound.protocol_fields
    assert bound.task.scientific_risk == "confirmatory"


def test_controller_unions_partial_model_protocol_with_all_required_fields():
    task = normalize_task("Analyze a prespecified primary endpoint.")
    master = MasterPlan(
        task=task,
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
        protocol_fields=["software versions and random seeds", "custom audit field"],
    )

    bound = bind_controller_task(master, task)

    assert bound.protocol_fields[:9] == DEFAULT_METHOD_LOCK_FIELDS
    assert bound.protocol_fields[9] == "custom audit field"


def test_bound_master_lint_accepts_complete_controller_protocol_only():
    task = normalize_task("Analyze a prespecified primary endpoint.")
    incomplete = MasterPlan(
        task=task,
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=True,
        protocol_fields=DEFAULT_METHOD_LOCK_FIELDS[:-1],
    )
    complete = bind_controller_task(incomplete, task)

    assert "missing_method_lock" in {
        item.code for item in lint_bound_master(incomplete).findings
    }
    assert lint_bound_master(complete).passed
    assert build_plan_audit_packet(complete)["deterministic_lint"]["passed"] is True


def test_run_authorization_clears_languages_for_documentation_only_task():
    task = _prepare_task_spec(
        "Compare the official Python and R API documentation.", enable_code=False
    )

    assert task.required_computation_languages == []
    assert any("no code-execution authorization" in item for item in task.constraints)


def test_task_spec_exposes_exact_virtual_input_manifest_to_planners():
    task = _prepare_task_spec(
        "Analyze the uploaded dataset in Python.",
        enable_code=True,
        input_manifest={
            "files": [
                {
                    "path": "known_effect.csv",
                    "bytes": 123,
                    "sha256": "b" * 64,
                }
            ]
        },
    )

    assert [item.model_dump() for item in task.available_inputs] == [
        {
            "path": "/workspace/known_effect.csv",
            "sha256": "b" * 64,
            "description": "immutable uploaded workspace input",
        }
    ]


def test_code_authorization_binds_r_first_figure_policy_to_task_contract():
    task = _prepare_task_spec(
        "Analyze the uploaded dataset and create a scientific figure.",
        enable_code=True,
    )

    assert any(
        "Reader-facing scientific figures use R by default" in item
        for item in task.constraints
    )
    assert any(
        "missing R packages are installed from canonical CRAN" in item
        for item in task.acceptance_tests
    )


def test_task_spec_exposes_controller_input_profile_to_planners():
    from scientific_agent.schemas import InputProfile

    profile = InputProfile(
        total_files=1,
        profiled_files=0,
        limitations=["task-specific reader required"],
    )
    task = _prepare_task_spec(
        "Analyze the uploaded dataset.",
        enable_code=True,
        input_profile=profile,
    )

    assert task.input_profile == profile


def test_task_spec_binds_requested_output_artifacts_before_planning():
    task = _prepare_task_spec(
        "Analyze the uploaded dataset.",
        enable_code=True,
        requested_outputs=("pptx_presentation", "analysis_notebook", "data_bundle"),
    )

    assert task.deliverables == [
        "Evidence-backed scientific report with claim and source ledgers",
        "PowerPoint presentation (.pptx)",
        "Reproducible analysis notebook (.ipynb)",
        "Machine-readable result bundle (.zip)",
    ]
    assert orchestrator_module._task_requests_visual_evidence(task)


def test_revision_task_is_a_new_survival_addendum_not_the_parent_plan():
    parent_task = normalize_task("Describe the uploaded cohort.")
    parent = PlanningResult(
        master_plan=MasterPlan(
            task=parent_task,
            plan=_plan("MASTER"),
            resolutions=[],
            method_lock_required=False,
        ),
        audit=VerificationReport(verdict="pass"),
        plan_lints=[],
        status="supported",
    )

    task = _prepare_revision_task_spec(
        "Add more survival and competing risk analyses.",
        parent,
        enable_code=True,
    )

    assert task.objective == "Add more survival and competing risk analyses."
    assert task.objective != parent_task.objective
    assert "New machine-readable analysis results produced in this child run" in (
        task.deliverables
    )
    contract = " ".join([*task.constraints, *task.acceptance_tests]).casefold()
    assert "time origin" in contract
    assert "recurrence and progression are not competing events" in contract
    assert "univariate association" in contract
    assert "ratio confidence" in contract

    unauthorized = _prepare_revision_task_spec(
        "Compute adjusted odds ratios.",
        parent,
        enable_code=False,
    )
    assert "New machine-readable analysis results produced in this child run" in (
        unauthorized.deliverables
    )
    assert (
        "code execution is not authorized"
        in " ".join(unauthorized.constraints).casefold()
    )


def test_revision_analysis_intent_distinguishes_computation_from_editorial_changes():
    for request in (
        "Rephrase the conclusion to add clarity.",
        "Please expand the discussion with more context.",
        "Add a sentence noting the sample size.",
        "Add discussion without rerunning the analysis.",
        "Do not perform a new analysis; clarify the old table.",
        "Update the hazard ratio wording.",
        "Add a paragraph discussing the Cox model.",
        "Improve the fit of the layout.",
        "Clarify the estimate wording.",
        "Do not rerun the Cox model but clarify its caption.",
    ):
        assert not _revision_requests_new_analysis(request), request

    for request in (
        "Add more survival and competing risk analyses.",
        "Compute adjusted odds ratios.",
        "Fit a logistic regression.",
        "Rerun the primary analysis.",
        "Provide a Fine-Gray competing risk analysis.",
        "Report the cause-specific hazard ratios.",
        "Present Kaplan-Meier survival curves by arm.",
        "Model overall survival with a Cox model.",
        "Determine the cumulative incidence of relapse.",
        "Produce a survival analysis for the cohort.",
        "Test proportional hazards.",
        "Do not alter the prose but rerun the Cox model.",
    ):
        assert _revision_requests_new_analysis(request), request


def test_revision_display_gate_matches_requested_scope():
    assert _revision_required_new_display_kinds("Add a Cox hazard-ratio table") == (
        "table",
    )
    assert _revision_required_new_display_kinds("Add a Kaplan-Meier figure") == (
        "figure",
    )
    generic = "Add more survival and competing risk analyses"
    assert _revision_required_new_display_kinds(generic) == ()
    assert _revision_requires_any_new_display(generic)
    assert (
        _revision_required_new_display_kinds("Rerun the Cox model without a new figure")
        == ()
    )


def test_controller_task_cannot_be_shortened_by_plan_synthesis():
    full = normalize_task(
        "Analyze the dataset in Python and R, then save a reconciliation artifact."
    )
    shortened = full.model_copy(
        update={
            "objective": "Analyze the dataset.",
            "required_computation_languages": [],
        }
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


@pytest.mark.asyncio
async def test_plan_repair_preserves_controller_task_and_uses_dedicated_prompt(
    monkeypatch,
):
    controller_task = _task()
    rewritten_task = controller_task.model_copy(update={"objective": "Different task"})
    revised_master = MasterPlan(
        task=rewritten_task,
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )
    seen = {}

    async def fake_request(_endpoint, **kwargs):
        seen["system_prompt"] = kwargs["system_prompt"]
        seen["payload"] = kwargs["payload"]
        return revised_master

    async def fake_audit(_settings, _master, _visible=None):
        return VerificationReport(verdict="pass")

    monkeypatch.setattr(orchestrator_module, "request_structured", fake_request)
    monkeypatch.setattr(orchestrator_module, "_audit_plan", fake_audit)
    initial = PlanningResult(
        master_plan=revised_master.model_copy(update={"task": controller_task}),
        audit=VerificationReport(
            verdict="fail",
            blocking_findings=[
                Finding(
                    finding_id="plan-1",
                    location="plan.steps[0]",
                    problem="The method is underspecified.",
                    why_it_matters="The result would not be reproducible.",
                    evidence="The supplied method lacks an operational rule.",
                    falsification_test_or_correction="Specify the operational rule.",
                )
            ],
            nonblocking_findings=[
                Finding(
                    finding_id="plan-output",
                    location="plan.expected_artifacts",
                    problem="The raster dimensions are not fixed.",
                    why_it_matters="Rendering would not be reproducible.",
                    evidence="No dimensions appear in the plan.",
                    falsification_test_or_correction="Specify dimensions and DPI.",
                )
            ],
        ),
        plan_lints=[
            PlanLintReport(
                passed=False,
                findings=[
                    LintFinding(
                        code="unmapped_deliverable",
                        location="task.deliverables[0]",
                        message="No declared output produces the requested bundle.",
                    )
                ],
            )
        ],
        status="requires_revision",
    )

    repaired = await orchestrator_module._repair_plan(Settings(), initial)

    assert repaired.master_plan.task == controller_task
    assert repaired.status == "supported"
    assert "every concrete blocking" in seen["system_prompt"]
    assert "never silently drop or regress" in seen["system_prompt"]
    assert [
        item["finding_id"] for item in seen["payload"]["cumulative_repair_findings"]
    ] == ["plan-1", "plan-output"]
    assert seen["payload"]["cumulative_plan_lints"] == [
        {
            "code": "unmapped_deliverable",
            "location": "task.deliverables[0]",
            "message": "No declared output produces the requested bundle.",
            "blocking": True,
        }
    ]


def test_plan_repair_lints_accumulate_and_deduplicate():
    first = LintFinding(code="lint-a", location="plan", message="Fix A")
    second = LintFinding(code="lint-b", location="plan.steps[0]", message="Fix B")

    merged = orchestrator_module._merge_plan_repair_lints(
        (first,),
        [PlanLintReport(passed=False, findings=[first, second])],
    )

    assert merged == (first, second)


def test_plan_repair_findings_accumulate_without_losing_same_criterion():
    formula = Finding(
        finding_id="plan-audit-reproducibility_and_unresolved_ambiguity",
        location="plan.steps[0].validators[0]",
        problem="The Hedges correction formula is wrong.",
        why_it_matters="A correct result would fail validation.",
        evidence="The denominator is 4*(N-9).",
        falsification_test_or_correction="Use the task-specified denominator.",
    )
    site = Finding(
        finding_id="plan-audit-reproducibility_and_unresolved_ambiguity",
        location="plan.unresolved_questions",
        problem="The role of site remains unresolved.",
        why_it_matters="The executable protocol remains ambiguous.",
        evidence="The plan asks whether site should be included.",
        falsification_test_or_correction="Explicitly include or omit site.",
    )
    arm_mapping = Finding(
        finding_id="deterministic-arbitrary_semantic_arm_mapping",
        location="plan.steps[0].validators[1]",
        problem="Semantic arm mapping is not explicit.",
        why_it_matters="The contrast could be reversed.",
        evidence="The mapping relies on category order.",
        falsification_test_or_correction="Use explicit normalized role labels.",
    )

    first = orchestrator_module._merge_plan_repair_findings(
        (), VerificationReport(verdict="fail", blocking_findings=[formula])
    )
    second = orchestrator_module._merge_plan_repair_findings(
        first,
        VerificationReport(
            verdict="fail",
            blocking_findings=[arm_mapping],
            nonblocking_findings=[site],
        ),
    )
    deduplicated = orchestrator_module._merge_plan_repair_findings(
        second,
        VerificationReport(verdict="fail", blocking_findings=[formula]),
    )

    assert [finding.problem for finding in second] == [
        formula.problem,
        arm_mapping.problem,
        site.problem,
    ]
    assert deduplicated == second


@pytest.mark.asyncio
async def test_plan_repair_cannot_downgrade_controller_method_lock(monkeypatch):
    controller_task = normalize_task("Analyze a prespecified primary endpoint.")
    downgraded = MasterPlan(
        task=controller_task.model_copy(update={"scientific_risk": "exploratory"}),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
        protocol_fields=[],
    )

    async def fake_request(*_args, **_kwargs):
        return downgraded

    async def fake_audit(*_args, **_kwargs):
        return VerificationReport(verdict="pass")

    monkeypatch.setattr(orchestrator_module, "request_structured", fake_request)
    monkeypatch.setattr(orchestrator_module, "_audit_plan", fake_audit)
    initial = PlanningResult(
        master_plan=downgraded.model_copy(update={"task": controller_task}),
        audit=VerificationReport(
            verdict="fail",
            blocking_findings=[
                Finding(
                    finding_id="lock-missing",
                    location="method_lock_required",
                    problem="The required method lock is missing.",
                    why_it_matters="The analysis is confirmatory.",
                    evidence="The controller classified the task as confirmatory.",
                    falsification_test_or_correction="Restore the controller method lock.",
                )
            ],
        ),
        plan_lints=[],
        status="requires_revision",
    )

    repaired = await orchestrator_module._repair_plan(Settings(), initial)

    assert repaired.master_plan.task == controller_task
    assert repaired.master_plan.method_lock_required is True
    assert len(repaired.master_plan.protocol_fields) == 9


@pytest.mark.asyncio
async def test_plan_repair_keeps_a_concrete_inconclusive_finding_fixable(monkeypatch):
    task = _task()
    master = MasterPlan(
        task=task,
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )
    blocking = Finding(
        finding_id="search-rule",
        location="plan.steps[0]",
        problem="The search query is unspecified.",
        why_it_matters="Retrieval is not reproducible.",
        evidence="No query appears in the plan.",
        falsification_test_or_correction="Add a fixed query.",
    )

    async def fake_request(*_args, **_kwargs):
        return master

    async def fake_audit(*_args, **_kwargs):
        return VerificationReport(verdict="inconclusive", blocking_findings=[blocking])

    monkeypatch.setattr(orchestrator_module, "request_structured", fake_request)
    monkeypatch.setattr(orchestrator_module, "_audit_plan", fake_audit)
    initial = PlanningResult(
        master_plan=master,
        audit=VerificationReport(verdict="inconclusive", blocking_findings=[blocking]),
        plan_lints=[],
        status="requires_revision",
    )

    repaired = await orchestrator_module._repair_plan(Settings(), initial)

    assert repaired.status == "requires_revision"


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


def test_research_failure_packet_preserves_bounded_retrieval_evidence(tmp_path):
    artifact = tmp_path / "retrieval.json"
    artifact.write_text(
        json.dumps({"content": "retrieved evidence " * 10_000}), encoding="utf-8"
    )
    retrieval = RetrievalEvidence(
        successful_calls=1,
        tools=["brave_web_search"],
        artifacts=[str(artifact)],
    )

    packet = _fallback_evidence_packet(
        RuntimeError("transport failed"), ComputationEvidence(), retrieval
    )
    decoded = json.loads(packet)

    assert decoded["error_type"] == "RuntimeError"
    assert decoded["retrieval_previews"][0]["path"] == str(artifact)
    assert "retrieved evidence" in decoded["retrieval_previews"][0]["content"]
    assert len(packet.encode("utf-8")) < 80 * 1024


def test_research_budget_exhaustion_can_continue_only_with_existing_evidence(tmp_path):
    assert _can_continue_after_research_error(
        repairing=True,
        computation=ComputationEvidence(),
        retrieval=RetrievalEvidence(),
    )
    assert _can_continue_after_research_error(
        repairing=False,
        computation=ComputationEvidence(),
        retrieval=RetrievalEvidence(successful_calls=1),
    )
    assert not _can_continue_after_research_error(
        repairing=False,
        computation=ComputationEvidence(),
        retrieval=RetrievalEvidence(),
    )
    assert not _can_continue_after_research_error(
        repairing=True,
        computation=ComputationEvidence(),
        retrieval=RetrievalEvidence(successful_calls=2),
        require_current_computation=True,
    )
    intake = tmp_path / "data_structure.json"
    intake.write_text('{"n_total": 2879, "rfs_events": 1352}\n', encoding="utf-8")
    intake_only = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-intake",
                language="python",
                code_sha256="e" * 64,
                started_at="2026-07-19T00:00:00+00:00",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout-intake"),
                stderr_path=str(tmp_path / "stderr-intake"),
                artifacts=[
                    ArtifactRef(
                        path=str(intake),
                        description="sandbox-generated analysis artifact",
                    )
                ],
            )
        ]
    )
    assert not _can_continue_after_research_error(
        repairing=True,
        computation=intake_only,
        retrieval=RetrievalEvidence(successful_calls=1),
        require_current_computation=True,
    )
    result = tmp_path / "result.json"
    result.write_text('{"hazard_ratio": 1.2}\n', encoding="utf-8")
    reportable = ComputationEvidence(
        records=[
            ComputationRecord(
                execution_id="exec-result",
                language="python",
                code_sha256="d" * 64,
                started_at="2026-07-19T00:00:00+00:00",
                duration_seconds=1,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout"),
                stderr_path=str(tmp_path / "stderr"),
                artifacts=[
                    ArtifactRef(
                        path=str(result),
                        description="sandbox-generated analysis artifact",
                    )
                ],
            )
        ]
    )
    assert _can_continue_after_research_error(
        repairing=True,
        computation=reportable,
        retrieval=RetrievalEvidence(),
        require_current_computation=True,
    )


def test_current_run_computation_requirement_covers_analysis_and_child_revisions():
    analysis_task = _task().model_copy(update={"task_type": "data_analysis"})
    writing_task = _task().model_copy(update={"task_type": "literature_review"})

    assert _requires_current_run_computation(
        analysis_task,
        enable_code=True,
        repairing=False,
        revision_request=None,
    )
    assert _requires_current_run_computation(
        writing_task,
        enable_code=True,
        repairing=True,
        revision_request="Add more survival analyses.",
    )
    assert not _requires_current_run_computation(
        writing_task,
        enable_code=True,
        repairing=True,
        revision_request="Clarify the survival-analysis limitations.",
    )


def test_research_continuation_handoff_forbids_reporting_before_computation():
    task = _task().model_copy(update={"task_type": "statistical_modeling"})
    planning = PlanningResult(
        master_plan=MasterPlan(
            task=task,
            plan=_plan("MASTER"),
            resolutions=[],
            method_lock_required=False,
        ),
        audit=VerificationReport(verdict="pass"),
        plan_lints=[],
        status="supported",
    )

    payload = _research_continuation_payload(
        planning=planning,
        retrieval=RetrievalEvidence(successful_calls=1),
        computation=ComputationEvidence(),
        error=RuntimeError("secret transport detail"),
        revision_request="Add more survival analyses.",
        environment_records=[
            {
                "language": "r",
                "repository": "cran",
                "requested": ["survival"],
                "status": "succeeded",
                "stdout": "unbounded installer output",
            }
        ],
    )

    assert "run_python_analysis or run_r_analysis" in payload["mandatory_next_action"]
    assert "Do not draft the report" in payload["mandatory_next_action"]
    assert payload["user_revision_request"] == "Add more survival analyses."
    assert payload["successful_package_installations"] == [
        {"language": "r", "repository": "cran", "requested": ["survival"]}
    ]
    assert "unbounded installer output" not in json.dumps(payload)
    assert "secret transport detail" not in json.dumps(payload)


@pytest.mark.asyncio
async def test_plan_critic_failure_becomes_explicit_inconclusive_audit(monkeypatch):
    master = MasterPlan(
        task=_task(),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )

    observed = {}

    async def failed_critic(*_args, **kwargs):
        observed.update(kwargs)
        raise RuntimeError("private endpoint detail must not be copied")

    monkeypatch.setattr("scientific_agent.workflow.request_structured", failed_critic)

    audit = await audit_master_plan(Settings(), master)
    result = package_planning({"keep_master": master, "plan_auditor": audit})

    assert audit.verdict == "inconclusive"
    assert audit.blocking_findings[0].finding_id == PLAN_CRITIC_UNAVAILABLE
    assert "private endpoint detail" not in audit.model_dump_json()
    assert result.status == "inconclusive"
    assert observed["temperature"] == Settings().gemma.temperature
    assert observed["max_tokens"] == PLAN_AUDIT_MAX_TOKENS
    assert observed["max_private_reasoning_bytes_without_final"] == (
        PLAN_AUDIT_MAX_PRIVATE_REASONING_BYTES_WITHOUT_FINAL
    )


def test_concrete_inconclusive_plan_finding_requires_another_revision():
    master = MasterPlan(
        task=_task(),
        plan=_plan("MASTER"),
        resolutions=[],
        method_lock_required=False,
    )
    blocking = Finding(
        finding_id="retrieval-rule",
        location="plan.steps[0]",
        problem="The search rule is not operationally defined.",
        why_it_matters="The retrieval would not be reproducible.",
        evidence="No fixed query is declared.",
        falsification_test_or_correction="Declare the fixed query and source set.",
    )

    result = package_planning(
        {
            "keep_master": master,
            "plan_auditor": VerificationReport(
                verdict="inconclusive", blocking_findings=[blocking]
            ),
        }
    )

    assert result.status == "requires_revision"


def test_article_and_display_reviews_merge_to_strictest_verdict():
    article = VerificationReport(
        verdict="pass_with_nonblocking_comments",
        unsupported_claims=["article caveat"],
    )
    display = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="display-1",
                location="Figure 1",
                problem="The legend overlaps an annotation.",
                why_it_matters="The statistical annotation is unreadable.",
                evidence="visual input 1",
                falsification_test_or_correction="Move the legend and rerender.",
            )
        ],
        unsupported_claims=["display mismatch"],
    )

    merged = _merge_reviews(article, display)

    assert merged.verdict == "fail"
    assert merged.unsupported_claims == ["article caveat", "display mismatch"]


def test_review_merge_caps_unique_findings_with_explicit_overflow_record():
    def finding(index):
        return Finding(
            finding_id=f"finding-{index}",
            location=f"item {index}",
            problem=f"Problem {index}",
            why_it_matters="It changes scientific interpretation.",
            evidence=f"Evidence {index}",
            falsification_test_or_correction=f"Check item {index}.",
        )

    first = VerificationReport(
        verdict="fail", blocking_findings=[finding(index) for index in range(200)]
    )
    second = VerificationReport(
        verdict="fail",
        blocking_findings=[finding(index) for index in range(200, 400)],
    )

    merged = _merge_reviews(first, second)

    assert len(merged.blocking_findings) == 200
    assert merged.blocking_findings[-1].finding_id == (
        "controller-blocking-findings-overflow"
    )
    assert "201 additional" in merged.blocking_findings[-1].problem


def test_review_merge_caps_falsification_tests_with_explicit_overflow_record():
    def checks(start):
        return [
            CheckSpec(
                check_id=f"check-{index}",
                description=f"Run falsification test {index}",
                check_type="test",
            )
            for index in range(start, start + 200)
        ]

    merged = _merge_reviews(
        VerificationReport(
            verdict="pass_with_nonblocking_comments",
            proposed_falsification_tests=checks(0),
        ),
        VerificationReport(
            verdict="pass_with_nonblocking_comments",
            proposed_falsification_tests=checks(200),
        ),
    )

    assert len(merged.proposed_falsification_tests) == 200
    overflow = merged.proposed_falsification_tests[-1]
    assert overflow.check_id == "controller-falsification-tests-overflow"
    assert "201 additional" in overflow.description
    assert overflow.blocking is False


def test_review_merge_preserves_findings_from_multiple_full_batches():
    def findings(prefix):
        return [
            Finding(
                finding_id=f"{prefix}-{index}",
                location=f"Figure {index}",
                problem="Concrete display defect",
                why_it_matters="It changes interpretation.",
                evidence="Visible mismatch",
                falsification_test_or_correction="Correct and rerender.",
            )
            for index in range(8)
        ]

    merged = _merge_reviews(
        VerificationReport(verdict="fail", blocking_findings=findings("a")),
        VerificationReport(verdict="fail", blocking_findings=findings("b")),
    )

    assert len(merged.blocking_findings) == 16
    assert merged.blocking_findings[-1].finding_id == "b-7"


def test_noop_typography_correction_cannot_block_a_report():
    display = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="display-noop",
                location="Figure 1 title",
                problem="Typo in title",
                why_it_matters="The alleged spelling error affects presentation.",
                evidence="Visible text: 'Effect Estimate'",
                falsification_test_or_correction="Correct 'Effect' to 'Effect'.",
            )
        ],
    )

    merged = _merge_reviews(VerificationReport(verdict="pass"), display)

    assert merged.verdict == "inconclusive"
    assert merged.blocking_findings == []
    assert any("no-op typography" in item for item in merged.unsupported_claims)


def test_critic_cannot_reverse_deterministic_table_precision_rule():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="more-decimals",
                location="Table 1",
                problem="The table loses precision.",
                why_it_matters="Exact verification allegedly requires more digits.",
                evidence="5.00 (expected 5.0000+)",
                falsification_test_or_correction=(
                    "Use format(value, '.4f') so at least 4 decimal places remain."
                ),
            )
        ],
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="Table 1",
                message="Reader table contains excessive precision.",
            )
        ],
    )

    filtered = _without_validation_conflicts(review, validation)

    assert filtered.verdict == "inconclusive"
    assert filtered.blocking_findings == []
    assert any("contradictory blocker" in item for item in filtered.unsupported_claims)


def _report_with_literature_and_computation_sources() -> ScientificReport:
    return ScientificReport(
        title="Effect report",
        executive_summary="The effect was estimated.",
        introduction="A published method motivated the analysis.",
        methods=["The effect was computed from the supplied data."],
        results="The estimated effect was 5.0.",
        discussion="The estimate is interpreted within the supplied data.",
        conclusions="The supplied data support an effect estimate.",
        claims=[
            ClaimRecord(
                claim_id="computed-effect",
                text="The estimated effect was 5.0.",
                claim_type="computed",
                evidence_refs=["src-python-results"],
                status="supported",
            )
        ],
        sources=[
            SourceRecord(
                source_id="src-python-results",
                title="Python result",
                artifact_path="/run/output/results.json",
                source_type="dataset",
                retrieved_at="2026-07-16T00:00:00Z",
                supporting_passage="Machine-readable computation output.",
            ),
            SourceRecord(
                source_id="src-nakagawa",
                title="Published effect-size method",
                url="https://pubmed.ncbi.nlm.nih.gov/17944619/",
                pmid="17944619",
                source_type="primary_study",
                retrieved_at="2026-07-16T00:00:00Z",
                supporting_passage="The source describes an effect-size method.",
            ),
        ],
    )


def test_critic_cannot_require_inline_citation_to_computation_artifact():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="missing-computation-citation",
                location="Results",
                problem="The numerical result lacks an InlineCitation.",
                why_it_matters="The number should remain traceable.",
                evidence="The result is supported by src-python-results.",
                falsification_test_or_correction=(
                    "Add an InlineCitation to the computation artifact "
                    "src-python-results."
                ),
            )
        ],
    )

    filtered = _without_inline_citation_policy_conflicts(
        review, _report_with_literature_and_computation_sources()
    )

    assert filtered.verdict == "inconclusive"
    assert filtered.blocking_findings == []
    assert any(
        "reserved for URL-backed" in item for item in filtered.unsupported_claims
    )


def test_controller_removes_policy_impossible_computation_inline_citation():
    report = _report_with_literature_and_computation_sources().model_copy(
        update={
            "inline_citations": [
                InlineCitation(
                    citation_id="computed",
                    section="results",
                    anchor_text="The estimated effect was 5.0.",
                    source_ids=["src-python-results", "src-nakagawa"],
                    claim_ids=["computed-effect"],
                )
            ]
        }
    )

    normalized = _normalize_inline_citation_provenance(report)

    assert normalized.inline_citations == []
    assert normalized.claims == report.claims


def test_misattributed_literature_citation_keeps_blocker_with_valid_correction():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="misattributed-literature",
                location="Results",
                problem=(
                    "The computed estimate is misattributed to src-nakagawa, "
                    "which does not support this dataset-specific value."
                ),
                why_it_matters="The citation implies the paper produced the result.",
                evidence="The value was generated in src-python-results.",
                falsification_test_or_correction=(
                    "Replace the InlineCitation source with src-python-results."
                ),
            )
        ],
    )

    filtered = _without_inline_citation_policy_conflicts(
        review, _report_with_literature_and_computation_sources()
    )

    assert filtered.verdict == "fail"
    assert len(filtered.blocking_findings) == 1
    correction = filtered.blocking_findings[0].falsification_test_or_correction
    assert correction.startswith("Remove the literature InlineCitation")
    assert "ClaimRecord.evidence_refs" in correction


def test_legitimate_missing_literature_inline_citation_remains_blocking():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="missing-literature-citation",
                location="Introduction",
                problem="The methodological statement lacks an InlineCitation.",
                why_it_matters="The statement relies on external literature.",
                evidence="src-nakagawa directly supports the statement.",
                falsification_test_or_correction=(
                    "Add an InlineCitation to src-nakagawa on that statement."
                ),
            )
        ],
    )

    filtered = _without_inline_citation_policy_conflicts(
        review, _report_with_literature_and_computation_sources()
    )

    assert filtered == review


def test_live_update_instruction_cannot_reverse_table_precision_rule():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="live-more-decimals",
                location="displays[1]",
                problem="The table allegedly loses precision.",
                why_it_matters="Exact verification allegedly requires more digits.",
                evidence="5.00 (expected 5.0000+)",
                falsification_test_or_correction=(
                    "Update the Python/R table generation logic to use "
                    "`format(value, '.4f')` or similar to ensure at least 4 "
                    "decimal places are preserved in the CSV output."
                ),
            )
        ],
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="displays[1]",
                message="Reader table contains excessive precision.",
            )
        ],
    )

    filtered = _without_validation_conflicts(review, validation)

    assert filtered.verdict == "inconclusive"
    assert filtered.blocking_findings == []


def test_precision_filter_preserves_aligned_and_cross_location_findings():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="aligned-rounding",
                location="displays[1]",
                problem="The reader-facing estimate has excessive precision.",
                why_it_matters="It implies unsupported numerical certainty.",
                evidence="The table prints 5.0000.",
                falsification_test_or_correction="Round 5.0000 to 5.00.",
            ),
            Finding(
                finding_id="different-table",
                location="displays[2]",
                problem="A different table allegedly needs more precision.",
                why_it_matters="Its values are difficult to compare.",
                evidence="The table prints two decimals.",
                falsification_test_or_correction="Use at least 4 decimal places.",
            ),
        ],
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="displays[1]",
                message="Reader table contains excessive precision.",
            )
        ],
    )

    filtered = _without_validation_conflicts(review, validation)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == [
        "aligned-rounding",
        "different-table",
    ]


def test_precision_filter_does_not_reverse_a_negated_correction():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="negated-more-decimals",
                location="Table 1",
                problem="The table has excessive precision.",
                why_it_matters="It implies unsupported certainty.",
                evidence="The table prints four decimals.",
                falsification_test_or_correction=(
                    "Do not use at least four decimal places; round to two."
                ),
            )
        ],
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="Table 1",
                message="Reader table contains excessive precision.",
            )
        ],
    )

    filtered = _without_validation_conflicts(review, validation)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == [
        "negated-more-decimals"
    ]


@pytest.mark.parametrize(
    "correction",
    [
        "Rather than increase the precision to four decimals, round to two.",
        "Instead of showing more decimal places, round to two.",
        "There is no need for more decimal places; use two.",
        ("Update the table logic to not use format(value, '.4f'); use '.2f'."),
        (
            "Set the table not to use format(value, '.4f'), because two "
            "decimals are sufficient."
        ),
        (
            "Modify the table so it does not use format(value, '.4f'); "
            "round to two decimals."
        ),
        ("Update the table because it shouldn't use format(value, '.4f'); use '.2f'."),
        ("Change the table because it cannot use format(value, '.4f'); round to two."),
        "Set the table because it can't use format(value, '.4f'); use two decimals.",
        "Use no more decimal places; round the table to two.",
    ],
)
def test_precision_filter_preserves_contrastive_rounding_corrections(correction):
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="contrastive-rounding",
                location="Table 1",
                problem="The table has excessive precision.",
                why_it_matters="It implies unsupported certainty.",
                evidence="The table prints four decimals.",
                falsification_test_or_correction=correction,
            )
        ],
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="Table 1",
                message="Reader table contains excessive precision.",
            )
        ],
    )

    filtered = _without_validation_conflicts(review, validation)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == [
        "contrastive-rounding"
    ]


def test_ocr_corroborated_correction_defeats_hallucinated_typo_blocker():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="manga-typo",
                location="Panel B",
                problem=(
                    "Direct visual inspection of the same text label reads MANGA "
                    "although visual inspection of that same label shows ANCOVA."
                ),
                why_it_matters="The alleged typo changes the method name.",
                evidence="The figure allegedly says MANGA: 5.042.",
                falsification_test_or_correction=(
                    "Correct the plotting code from 'MANGA' to 'ANCOVA'."
                ),
            )
        ],
    )
    inputs = [
        {
            "kind": "figure",
            "ocr": {"available": True, "text": "Panel B ANCOVA 5.042"},
        }
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "inconclusive"
    assert filtered.blocking_findings == []
    assert any("controller OCR" in item for item in filtered.unsupported_claims)


def test_ocr_from_another_figure_cannot_suppress_typography_blocker():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="fig-1-typo",
                location="Figure fig-1",
                problem=(
                    "The same text label is transcribed as MANGA although visual "
                    "inspection of that same label shows ANCOVA."
                ),
                why_it_matters="The method name would be wrong.",
                evidence="Figure fig-1 allegedly says MANGA.",
                falsification_test_or_correction="Replace 'MANGA' with 'ANCOVA'.",
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "Panel A treatment effect"},
        },
        {
            "display_id": "fig-2",
            "kind": "figure",
            "ocr": {"available": True, "text": "Panel B ANCOVA"},
        },
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == ["fig-1-typo"]


def test_display_id_prefix_does_not_scope_ocr_to_the_wrong_figure():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="fig-10-typo",
                location="Figure fig-10",
                problem=(
                    "The same text label is transcribed as MANGA although visual "
                    "inspection of that same label shows ANCOVA."
                ),
                why_it_matters="The method name would be wrong.",
                evidence="Figure fig-10 allegedly says MANGA.",
                falsification_test_or_correction="Replace 'MANGA' with 'ANCOVA'.",
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "ANCOVA"},
        },
        {
            "display_id": "fig-2",
            "kind": "figure",
            "ocr": {"available": True, "text": "Other figure"},
        },
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == ["fig-10-typo"]


def test_figure_ocr_cannot_suppress_a_table_typography_blocker():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="table-typo",
                location="Table table-1",
                problem=(
                    "The same text label is transcribed as MANGA although visual "
                    "inspection of that same label shows ANCOVA."
                ),
                why_it_matters="The method name would be wrong.",
                evidence="Table table-1 allegedly says MANGA.",
                falsification_test_or_correction="Replace 'MANGA' with 'ANCOVA'.",
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "ANCOVA"},
        },
        {"display_id": "table-1", "kind": "table", "preview": "MANGA"},
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == ["table-typo"]


def test_caption_ocr_cannot_suppress_an_axis_label_blocker():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="axis-typo",
                location="Figure fig-1 y-axis",
                problem=(
                    "This axis label says MANGA, while visual evidence elsewhere "
                    "in the caption shows ANCOVA."
                ),
                why_it_matters="The axis method name would be wrong.",
                evidence="The y-axis allegedly says MANGA.",
                falsification_test_or_correction="Replace 'MANGA' with 'ANCOVA'.",
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "Caption ANCOVA"},
        }
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == ["axis-typo"]


def test_metadata_raster_mismatch_remains_a_real_display_blocker():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="metadata-title-mismatch",
                location="Figure fig-1 title",
                problem=(
                    "The ReportDisplay metadata title says MANGA, while visual "
                    "inspection of that same title shows ANCOVA."
                ),
                why_it_matters="Metadata and the rendered figure disagree.",
                evidence="The registered title is MANGA but the raster reads ANCOVA.",
                falsification_test_or_correction=(
                    "Replace the metadata title 'MANGA' with 'ANCOVA'."
                ),
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "ANCOVA"},
        }
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == [
        "metadata-title-mismatch"
    ]


def test_ocr_disagreement_alone_cannot_overrule_direct_visual_review():
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="direct-visual-typo",
                location="Figure fig-1",
                problem="Direct raster review identifies a MANGA text-label typo.",
                why_it_matters="The method name would be wrong.",
                evidence="The raster allegedly says MANGA.",
                falsification_test_or_correction="Replace 'MANGA' with 'ANCOVA'.",
            )
        ],
    )
    inputs = [
        {
            "display_id": "fig-1",
            "kind": "figure",
            "ocr": {"available": True, "text": "Panel B ANCOVA"},
        }
    ]

    filtered = _without_ocr_contradicted_typography(review, inputs)

    assert filtered.verdict == "fail"
    assert [item.finding_id for item in filtered.blocking_findings] == [
        "direct-visual-typo"
    ]


@pytest.mark.anyio
async def test_raw_contradictory_critic_outputs_remain_browsable(tmp_path, monkeypatch):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: (
            [image],
            [
                {
                    "display_id": "effect-figure",
                    "kind": "figure",
                    "sha256": "a" * 64,
                    "media_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "ocr": {
                        "available": True,
                        "text": "Panel B ANCOVA 5.042",
                        "words": [],
                    },
                }
            ],
        ),
    )
    calls = 0

    async def fake_request(_endpoint, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return VerificationReport(
                verdict="fail",
                blocking_findings=[
                    Finding(
                        finding_id="more-decimals",
                        location="Table 1",
                        problem="The table loses precision.",
                        why_it_matters="Exact verification allegedly needs digits.",
                        evidence="5.00 (expected 5.0000+)",
                        falsification_test_or_correction=(
                            "Use format(value, '.4f') for at least 4 decimal places."
                        ),
                    )
                ],
            )
        return VerificationReport(
            verdict="fail",
            blocking_findings=[
                Finding(
                    finding_id="manga-typo",
                    location="Panel B",
                    problem=(
                        "Direct visual inspection of the same text label reads MANGA "
                        "although visual inspection of that same label shows ANCOVA."
                    ),
                    why_it_matters="The alleged typo changes the method name.",
                    evidence="The raster allegedly says MANGA.",
                    falsification_test_or_correction=(
                        "Correct the plotting code from 'MANGA' to 'ANCOVA'."
                    ),
                )
            ],
        )

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="Table 1",
                message="Reader table contains excessive precision.",
            )
        ],
    )
    retrieval = RetrievalEvidence()
    computation = ComputationEvidence()
    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        validation,
        retrieval,
        computation,
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "inconclusive"
    assert (
        json.loads((tmp_path / "live" / "gemma_report_review_raw.json").read_text())[
            "verdict"
        ]
        == "fail"
    )
    assert (
        json.loads((tmp_path / "live" / "gemma_report_review.json").read_text())[
            "verdict"
        ]
        == "inconclusive"
    )

    _write_attempt_bundle(
        tmp_path,
        0,
        report,
        validation,
        review,
        retrieval,
        computation,
    )

    async def passing_request(_endpoint, **kwargs):
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", passing_request
    )
    second_review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        retrieval,
        computation,
        EventLedger(tmp_path / "events-second.jsonl"),
        live_dir=tmp_path / "live",
    )
    _write_attempt_bundle(
        tmp_path,
        1,
        report,
        DeterministicValidation(passed=True),
        second_review,
        retrieval,
        computation,
    )

    assert (
        json.loads((tmp_path / "live" / "gemma_report_review_raw.json").read_text())[
            "verdict"
        ]
        == "pass"
    )
    assert (
        json.loads(
            (
                tmp_path / "attempts" / "attempt-0" / "gemma_report_review_raw.json"
            ).read_text()
        )["verdict"]
        == "fail"
    )
    assert (
        json.loads(
            (
                tmp_path / "attempts" / "attempt-1" / "gemma_report_review_raw.json"
            ).read_text()
        )["verdict"]
        == "pass"
    )
    assert (
        json.loads(
            (tmp_path / "live" / "gemma_display_batch_001_raw.json").read_text()
        )["verdict"]
        == "pass"
    )
    assert (
        json.loads(
            (
                tmp_path / "attempts" / "attempt-0" / "gemma_display_batch_001_raw.json"
            ).read_text()
        )["verdict"]
        == "fail"
    )
    assert (
        json.loads(
            (
                tmp_path / "attempts" / "attempt-1" / "gemma_display_batch_001_raw.json"
            ).read_text()
        )["verdict"]
        == "pass"
    )


@pytest.mark.anyio
async def test_gemma_multimodal_display_audit_uses_raster_and_failure_blocks(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    display_inputs = [
        {
            "display_id": "effect-figure",
            "kind": "figure",
            "sha256": "a" * 64,
            "media_type": "image/png",
            "width": 800,
            "height": 600,
            "ocr": {
                "available": True,
                "text": "Effect estimate 95% CI 4.07 to 5.93",
                "words": [
                    {
                        "text": "Effect",
                        "confidence": 96.0,
                        "left": 10,
                        "top": 10,
                        "width": 50,
                        "height": 12,
                    }
                ],
            },
        }
    ]
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: ([image], display_inputs),
    )
    calls = []

    async def fake_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        if kwargs["on_visible_text"] is not None:
            kwargs["on_visible_text"](f"{endpoint.model} visible output")
        if len(calls) == 2:
            return VerificationReport(
                verdict="fail",
                blocking_findings=[
                    Finding(
                        finding_id="gemma-display-1",
                        location="Figure 1",
                        problem="The OCR text omits the comparison label.",
                        why_it_matters="The estimand is ambiguous.",
                        evidence="The extracted text names an interval but no groups.",
                        falsification_test_or_correction="Add the comparison label and rerender.",
                    )
                ],
            )
        return VerificationReport(verdict="pass")

    settings = Settings()
    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    activity = []
    review = await _audit_report_resilient(
        settings,
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
        activity=lambda *args: activity.append(args),
    )

    assert review.verdict == "fail"
    assert len(calls) == 2
    assert calls[0][0] is settings.gemma
    assert calls[0][1].get("image_paths", ()) == ()
    assert calls[1][0] is settings.gemma
    assert calls[1][1]["image_paths"] == (image,)
    assert calls[1][1]["temperature"] == 0.4
    assert "sole visual critic" in calls[1][1]["system_prompt"]
    blinded_payload = json.dumps(calls[1][1]["payload"]).casefold()
    assert "qwen" not in blinded_payload
    assert "gemma" not in blinded_payload
    assert not (tmp_path / "gemma_visual_audit.json").exists()
    assert not (tmp_path / "qwen_visual_audit.json").exists()
    display_audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert display_audit["critic_model"] == settings.gemma.model
    assert display_audit["review_source"] == "gemma_multimodal_critic"
    assert display_audit["review_mode"] == "raster_with_ocr_geometry_and_table_previews"
    assert display_audit["visual_critic"] == "Gemma"
    assert display_audit["qwen_image_inputs"] == 0
    assert display_audit["verdict"] == "fail"
    assert display_audit["figure_text_inputs"][0]["ocr_available"] is True
    assert display_audit["figure_text_inputs"][0]["geometry_available"] is True
    assert (
        settings.gemma.model
        in (tmp_path / "live" / "gemma_visible_output.txt").read_text()
    )
    assert {item[1] for item in activity if item[0] == "model_output_stream"} == {
        "Gemma",
    }


@pytest.mark.anyio
async def test_missing_figure_ocr_still_runs_gemma_multimodal_review(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: (
            [image],
            [
                {
                    "display_id": "effect-figure",
                    "kind": "figure",
                    "sha256": "a" * 64,
                    "media_type": "image/png",
                    "width": 800,
                    "height": 600,
                }
            ],
        ),
    )
    settings = Settings()
    calls = []

    async def fake_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    review = await _audit_report_resilient(
        settings,
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "pass"
    assert review.blocking_findings == []
    assert len(calls) == 2
    assert calls[0][0] is settings.gemma
    assert calls[0][1].get("image_paths", ()) == ()
    assert calls[1][0] is settings.gemma
    assert calls[1][1]["image_paths"] == (image,)
    display_audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert display_audit["critic_model"] == settings.gemma.model
    assert display_audit["review_source"] == "gemma_multimodal_critic"
    assert display_audit["figures_missing_ocr"] == ["effect-figure"]


@pytest.mark.anyio
async def test_bare_gemma_display_pass_fails_closed_and_preserves_raw_output(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    display_input = {
        "display_id": "effect-figure",
        "kind": "figure",
        "sha256": "a" * 64,
        "media_type": "image/png",
        "width": 800,
        "height": 600,
        "ocr": {"available": True, "text": "Effect plot", "words": []},
        "layout_review_questions": _high_priority_layout_questions(),
    }
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: ([image], [display_input]),
    )
    calls = []

    async def fake_request(_endpoint, **kwargs):
        calls.append(kwargs)
        return VerificationReport(verdict="pass", evidence_refs=[])

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )

    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "inconclusive"
    assert any(
        item.finding_id.startswith("controller-display-clearance-missing-")
        for item in review.blocking_findings
    )
    display_payload = calls[1]["payload"]
    assert display_payload["display_inputs"][0]["layout_review_questions"] == (
        _high_priority_layout_questions()
    )
    assert display_payload["required_clearance_refs"] == [
        "display-reviewed:effect-figure",
        "visual-clearance:effect-figure:top-text",
        "visual-clearance:effect-figure:legend-data",
        "visual-clearance:effect-figure:annotation-data",
    ]
    raw = json.loads(
        (tmp_path / "live" / "gemma_display_batch_001_raw.json").read_text()
    )
    normalized = json.loads(
        (tmp_path / "live" / "gemma_display_batch_001.json").read_text()
    )
    assert raw["verdict"] == "pass"
    assert raw["evidence_refs"] == []
    assert normalized["verdict"] == "inconclusive"


@pytest.mark.anyio
async def test_gemma_can_visually_clear_layout_warning_without_forced_repair(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: (
            [image],
            [
                {
                    "display_id": "effect-figure",
                    "kind": "figure",
                    "sha256": "a" * 64,
                    "media_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "ocr": {"available": True, "text": "Effect plot", "words": []},
                    "layout_review_questions": _high_priority_layout_questions(),
                }
            ],
        ),
    )

    async def fake_request(_endpoint, **kwargs):
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )

    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "pass"
    normalized = json.loads(
        (tmp_path / "live" / "gemma_display_batch_001.json").read_text()
    )
    assert normalized["verdict"] == "pass"
    assert normalized["evidence_refs"] == [
        "display-reviewed:effect-figure",
        "visual-clearance:effect-figure:top-text",
        "visual-clearance:effect-figure:legend-data",
        "visual-clearance:effect-figure:annotation-data",
    ]


@pytest.mark.anyio
async def test_display_audit_failure_preserves_completed_report_review(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: (
            [image],
            [
                {
                    "display_id": "effect-figure",
                    "kind": "figure",
                    "sha256": "a" * 64,
                    "media_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "ocr": {
                        "available": True,
                        "text": "Mean difference 5.00, 95% CI 4.07 to 5.93",
                        "words": [{"text": "Mean", "left": 10, "top": 10}],
                    },
                }
            ],
        ),
    )
    calls = 0

    async def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return VerificationReport(verdict="pass")
        raise RuntimeError("invalid display review")

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "inconclusive"
    assert calls == 3
    assert (
        json.loads((tmp_path / "live" / "gemma_report_review.json").read_text())[
            "verdict"
        ]
        == "pass"
    )
    display_review = json.loads(
        (tmp_path / "live" / "gemma_display_review.json").read_text()
    )
    assert display_review["verdict"] == "inconclusive"
    assert (
        "display review batch 1/1 unavailable"
        in display_review["unsupported_claims"][0]
    )
    display_audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert display_audit["critic_model"] is None
    assert display_audit["review_source"] == "controller_gate"
    assert display_audit["review_mode"] == "multimodal_unavailable"
    assert display_audit["batches_attempted"] == 1
    assert display_audit["batches_succeeded"] == 0


@pytest.mark.anyio
async def test_display_audit_retries_one_fresh_call_after_unusable_response(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"test image bytes")
    display_inputs = [
        {
            "display_id": "effect-figure",
            "kind": "figure",
            "sha256": "a" * 64,
            "media_type": "image/png",
            "width": 800,
            "height": 600,
            "ocr": {"available": True, "text": "Treatment effect 5.00", "words": []},
        }
    ]
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: ([image], display_inputs),
    )
    calls = 0

    async def flaky_request(*_args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return VerificationReport(verdict="pass")
        if calls == 2:
            raise RuntimeError("schema-invalid display review")
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", flaky_request
    )
    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert calls == 3
    assert review.verdict == "pass"
    display_audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert display_audit["batches_attempted"] == 1
    assert display_audit["batches_succeeded"] == 1


@pytest.mark.anyio
async def test_display_provenance_uses_exact_review_inputs_without_second_read(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    image = tmp_path / "effect.png"
    image.write_bytes(b"reviewed raster")
    preparations = 0

    def prepare(*_args):
        nonlocal preparations
        preparations += 1
        if preparations > 1:
            raise OSError("artifact changed after review")
        return (
            [image],
            [
                {
                    "display_id": "effect-figure",
                    "kind": "figure",
                    "sha256": "a" * 64,
                    "media_type": "image/png",
                    "width": 800,
                    "height": 600,
                    "ocr": {
                        "available": True,
                        "text": "Treatment effect 5.00",
                        "words": [{"text": "Treatment"}],
                    },
                }
            ],
        )

    monkeypatch.setattr("scientific_agent.orchestrator.prepare_display_audit", prepare)

    async def passing_review(*_args, **kwargs):
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", passing_review
    )

    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert review.verdict == "pass"
    assert preparations == 1
    audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert audit["review_source"] == "gemma_multimodal_critic"
    assert audit["review_mode"] == "raster_with_ocr_geometry_and_table_previews"
    assert audit["figure_text_inputs"][0]["display_id"] == "effect-figure"


@pytest.mark.anyio
async def test_invalid_display_preparation_preserves_completed_article_audit(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()

    def invalid_display(*_args):
        raise ValueError("TIFF is not an inline report format")

    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit", invalid_display
    )
    calls = 0

    async def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return VerificationReport(verdict="pass")

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    review = await _audit_report_resilient(
        Settings(),
        planning,
        report,
        DeterministicValidation(passed=False),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
        live_dir=tmp_path / "live",
    )

    assert calls == 1
    assert review.verdict == "inconclusive"
    assert (
        json.loads((tmp_path / "live" / "gemma_report_review.json").read_text())[
            "verdict"
        ]
        == "pass"
    )
    display_audit = json.loads((tmp_path / "gemma_display_audit.json").read_text())
    assert display_audit["critic_model"] is None
    assert display_audit["review_source"] == "controller_gate"
    assert display_audit["review_mode"] == "invalid_display_inputs"
    assert display_audit["input_error"] == (
        "ValueError: TIFF is not an inline report format"
    )


@pytest.mark.anyio
async def test_gemma_only_visual_review_batches_more_than_five_images(
    tmp_path, monkeypatch
):
    planning, report = _display_audit_fixture()
    images = []
    inputs = []
    for index in range(6):
        image = tmp_path / f"figure-{index}.png"
        image.write_bytes(f"image {index}".encode())
        images.append(image)
        inputs.append(
            {
                "display_id": f"figure-{index}",
                "kind": "figure",
                "sha256": f"{index:x}" * 64,
                "media_type": "image/png",
                "width": 800,
                "height": 600,
                "ocr": {"available": False, "text": "", "words": []},
            }
        )
    report = report.model_copy(
        update={
            "displays": [
                ReportDisplay(
                    display_id=f"figure-{index}",
                    kind="figure",
                    title=f"Figure {index}",
                    caption="Current batch figure.",
                    artifact_path=f"/run/output/figures/figure-{index}.png",
                    alt_text="A test figure.",
                )
                for index in range(6)
            ]
        }
    )
    monkeypatch.setattr(
        "scientific_agent.orchestrator.prepare_display_audit",
        lambda *_args: (images, inputs),
    )
    settings = Settings()
    calls = []

    async def fake_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return _pass_with_requested_display_clearances(kwargs)

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    review = await _audit_report_resilient(
        settings,
        planning,
        report,
        DeterministicValidation(passed=True),
        RetrievalEvidence(),
        ComputationEvidence(),
        EventLedger(tmp_path / "events.jsonl"),
    )

    assert review.verdict == "pass"
    assert len(calls) == 3
    assert all(endpoint is settings.gemma for endpoint, _kwargs in calls)
    assert calls[0][1].get("image_paths", ()) == ()
    assert calls[1][1]["image_paths"] == tuple(images[:5])
    assert calls[2][1]["image_paths"] == tuple(images[5:])
    assert calls[1][1]["payload"]["visual_input_order"] == [
        f"figure-{index}" for index in range(5)
    ]
    assert calls[2][1]["payload"]["visual_input_order"] == ["figure-5"]
    assert {item["display_id"] for item in calls[1][1]["payload"]["displays"]} == {
        f"figure-{index}" for index in range(5)
    }
    assert [item["display_id"] for item in calls[2][1]["payload"]["displays"]] == [
        "figure-5"
    ]
    assert "judge only the current batch" in calls[1][1]["system_prompt"]


def test_visual_batching_respects_total_bytes_not_only_image_count(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orchestrator_module, "MAX_TOTAL_IMAGE_BYTES", 10)
    images = []
    inputs = []
    for index in range(3):
        image = tmp_path / f"bytes-{index}.png"
        image.write_bytes(b"123456")
        images.append(image)
        inputs.append({"display_id": f"figure-{index}"})

    batches, rejected = orchestrator_module._bounded_visual_batches(images, inputs)

    assert rejected == []
    assert [batch_images for batch_images, _ in batches] == [
        [images[0]],
        [images[1]],
        [images[2]],
    ]


def test_extreme_pixel_count_input_is_rejected_without_aborting_run(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "compressed-large-pixel-count.png"
    Image.new("L", (32, 32), color=0).save(image)
    destination = tmp_path / "converted"
    destination.mkdir()
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)

    assert orchestrator_module._valid_model_raster(image) is False
    converted, failures = orchestrator_module._convert_image_frames(
        image, destination, "uploaded image", 1
    )
    assert converted == []
    assert failures == [
        "uploaded image could not be converted (DecompressionBombError)"
    ]
    images, _inputs, omitted = orchestrator_module._collect_input_visuals(
        workspace,
        ComputationEvidence(),
        TaskSpec(
            task_id="nonvisual",
            objective="Summarize the tabular dataset trends",
            deliverables=["scientific report"],
            acceptance_tests=["The report is evidence grounded"],
        ),
    )
    assert images == []
    assert any("invalid or oversized" in item for item in omitted)


@pytest.mark.anyio
async def test_source_images_are_reviewed_only_by_gemma_and_cached_by_hash(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "source.png"
    Image.new("RGB", (32, 24), color=(120, 80, 20)).save(image)
    Image.new("RGB", (32, 24), color=(20, 80, 120)).save(
        workspace / "source-figure.tiff"
    )
    (workspace / "visual-proof.pdf").write_bytes(b"%PDF-1.4\n")
    planning, _report = _display_audit_fixture()
    planning = planning.model_copy(
        update={
            "master_plan": planning.master_plan.model_copy(
                update={
                    "task": planning.master_plan.task.model_copy(
                        update={
                            "objective": "Inspect the attached figure and proof PDF",
                            "deliverables": ["visual evidence report"],
                        }
                    )
                }
            )
        }
    )
    settings = Settings(workspace=workspace)
    calls = []

    async def fake_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return VisualEvidenceReport(
            observations=[
                VisualEvidenceObservation(
                    artifact_path=artifact_path,
                    observed_content="A two-panel scientific chart is visible.",
                    scientific_interpretation=(
                        "The chart can inform the requested visual-fidelity review."
                    ),
                )
                for artifact_path in kwargs["payload"]["visual_input_order"]
            ]
        )

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    run_dir = tmp_path / "run"
    live_dir = run_dir / "live"
    first = await orchestrator_module._review_input_visual_evidence(
        settings,
        planning,
        ComputationEvidence(),
        "bounded research context",
        run_dir,
        live_dir=live_dir,
    )
    second = await orchestrator_module._review_input_visual_evidence(
        settings,
        planning,
        ComputationEvidence(),
        "bounded research context",
        run_dir,
        live_dir=live_dir,
    )

    assert first == second
    assert len(calls) == 1
    assert calls[0][0] is settings.gemma
    assert image in calls[0][1]["image_paths"]
    assert len(calls[0][1]["image_paths"]) == 2
    assert calls[0][1]["payload"]["visual_input_order"] == [
        "visual-001",
        "visual-002",
    ]
    assert all(
        item["artifact_path"].startswith("visual-")
        for item in calls[0][1]["payload"]["visual_inputs"]
    )
    assert all(
        not item["artifact_path"].startswith("/data/")
        for item in calls[0][1]["payload"]["visual_inputs"]
    )
    reviewed_paths = {item.artifact_path for item in first.observations}
    assert "/workspace/source.png" in reviewed_paths
    assert len(reviewed_paths) == 2
    assert all(not path.startswith("visual-") for path in reviewed_paths)
    assert "sole image-understanding scientist" in calls[0][1]["system_prompt"]
    assert any("visual-proof.pdf" in item for item in first.unreviewed_requests)
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert audit["critic_model"] == settings.gemma.model
    assert audit["qwen_image_inputs"] == 0
    assert audit["batches_attempted"] == 1
    assert audit["batches_succeeded"] == 1


@pytest.mark.anyio
async def test_selected_knowledge_visuals_are_bounded_and_sent_only_to_gemma(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    run_dir = tmp_path / "run"
    visual_dir = run_dir / "knowledge" / "visuals"
    visual_dir.mkdir(parents=True)
    visuals = []
    for index in range(orchestrator_module.MAX_INPUT_VISUALS + 2):
        knowledge_visual_id = f"kvp-{index:024x}"
        path = visual_dir / f"{knowledge_visual_id}.png"
        Image.new("RGB", (32, 24), color=(index, 80, 120)).save(path)
        digest = sha256_file(path)
        visuals.append(
            KnowledgeVisualEvidence(
                knowledge_visual_id=knowledge_visual_id,
                document_id=f"{index:032x}",
                visual_id=f"kv-{index:024x}",
                title=f"Selected visual {index}",
                source_type="primary_study",
                source_label=f"page {index + 1}",
                document_filename=f"source-{index}.pdf",
                document_original_sha256="a" * 64,
                visual_sha256=digest,
                source_url=(
                    f"https://bench.test/api/runs/r1/knowledge/visuals/"
                    f"{knowledge_visual_id}"
                ),
                artifact_path=str(path),
                artifact_sha256=digest,
                snapshot_sha256="b" * 64,
                retrieved_at="2026-07-16T12:00:00+00:00",
                retrieval_method="visual_descriptor",
            )
        )
    settings = Settings(workspace=workspace)
    calls = []

    async def fake_request(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return VisualEvidenceReport(
            observations=[
                VisualEvidenceObservation(
                    artifact_path=visual_id,
                    observed_content="A scientific raster is visible.",
                    scientific_interpretation=(
                        "The visible content can inform an evidence-bounded report."
                    ),
                )
                for visual_id in kwargs["payload"]["visual_input_order"]
            ]
        )

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    report = await orchestrator_module._review_input_visual_evidence(
        settings,
        _visual_review_planning(),
        ComputationEvidence(),
        "Qwen requested Gemma inspection but made no pixel observations.",
        run_dir,
        knowledge_visuals=tuple(visuals),
        live_dir=run_dir / "live",
    )

    assert calls
    assert all(call[0] is settings.gemma for call in calls)
    sent_paths = [path for _, call in calls for path in call["image_paths"]]
    assert sent_paths == [Path(item.artifact_path) for item in visuals[:20]]
    assert sum(len(call["payload"]["visual_inputs"]) for _, call in calls) == 20
    assert all(
        item["source"] == "selected_knowledge_visual"
        for _, call in calls
        for item in call["payload"]["visual_inputs"]
    )
    assert len(report.observations) == 20
    assert all(
        item.source_url in " ".join(report.unreviewed_requests) for item in visuals[20:]
    )
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert audit["visual_critic"] == "Gemma"
    assert audit["qwen_image_inputs"] == 0


def _visual_review_planning():
    planning, _report = _display_audit_fixture()
    return planning.model_copy(
        update={
            "master_plan": planning.master_plan.model_copy(
                update={
                    "task": planning.master_plan.task.model_copy(
                        update={
                            "objective": "Inspect all attached source figures",
                            "deliverables": ["visual evidence report"],
                        }
                    )
                }
            )
        }
    )


@pytest.mark.anyio
async def test_partial_source_visual_review_has_explicit_partial_provenance(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(6):
        Image.new("RGB", (32, 24), color=(index * 20, 80, 120)).save(
            workspace / f"source-{index}.png"
        )
    calls = 0

    async def review(_endpoint, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second batch unavailable")
        return VisualEvidenceReport(
            observations=[
                VisualEvidenceObservation(
                    artifact_path=visual_id,
                    observed_content="A scientific source figure is visible.",
                    scientific_interpretation="It supports a bounded visual audit.",
                )
                for visual_id in kwargs["payload"]["visual_input_order"]
            ]
        )

    monkeypatch.setattr("scientific_agent.orchestrator.request_structured", review)
    run_dir = tmp_path / "run"

    report = await orchestrator_module._review_input_visual_evidence(
        Settings(workspace=workspace),
        _visual_review_planning(),
        ComputationEvidence(),
        "bounded research context",
        run_dir,
    )

    assert len(report.observations) == 5
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert audit["batches_attempted"] == 2
    assert audit["batches_succeeded"] == 1
    assert audit["review_source"] == "gemma_multimodal_input_critic_partial"
    assert len(audit["batch_reports"]) == 1
    assert len(audit["batch_errors"]) == 1


@pytest.mark.anyio
async def test_source_visual_finding_aggregation_is_bounded_and_preserves_batches(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(6):
        Image.new("RGB", (32, 24), color=(index * 20, 80, 120)).save(
            workspace / f"source-{index}.png"
        )
    calls = 0

    async def review(_endpoint, **kwargs):
        nonlocal calls
        calls += 1
        return VisualEvidenceReport(
            observations=[
                VisualEvidenceObservation(
                    artifact_path=visual_id,
                    observed_content="A scientific source figure is visible.",
                    scientific_interpretation="It supports a bounded visual audit.",
                )
                for visual_id in kwargs["payload"]["visual_input_order"]
            ],
            cross_artifact_findings=[
                f"batch {calls} finding {index}" for index in range(100)
            ],
        )

    monkeypatch.setattr("scientific_agent.orchestrator.request_structured", review)
    run_dir = tmp_path / "run"

    report = await orchestrator_module._review_input_visual_evidence(
        Settings(workspace=workspace),
        _visual_review_planning(),
        ComputationEvidence(),
        "bounded research context",
        run_dir,
    )

    assert len(report.cross_artifact_findings) == 100
    assert "101 additional" in report.cross_artifact_findings[-1]
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert len(audit["batch_reports"]) == 2


def test_visual_evidence_accepts_single_finding_strings():
    report = VisualEvidenceReport.model_validate(
        {
            "observations": [
                {
                    "artifact_path": "visual-001",
                    "observed_content": "A complete chart is visible.",
                    "scientific_interpretation": "It reports a descriptive estimate.",
                    "concerns": "The unit is not visible.",
                    "limitations": "Only one panel was supplied.",
                }
            ],
            "cross_artifact_findings": "The two views agree.",
            "limitations": "The evidence is descriptive.",
            "unreviewed_requests": "A second page was not supplied.",
        }
    )

    assert report.observations[0].concerns == ["The unit is not visible."]
    assert report.observations[0].limitations == ["Only one panel was supplied."]
    assert report.cross_artifact_findings == ["The two views agree."]
    assert report.limitations == ["The evidence is descriptive."]
    assert report.unreviewed_requests == ["A second page was not supplied."]


@pytest.mark.anyio
async def test_failed_input_visual_review_is_retried_instead_of_cached(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = workspace / "source.png"
    Image.new("RGB", (32, 24), color=(120, 80, 20)).save(image)
    planning, _report = _display_audit_fixture()
    planning = planning.model_copy(
        update={
            "master_plan": planning.master_plan.model_copy(
                update={
                    "task": planning.master_plan.task.model_copy(
                        update={
                            "objective": "Inspect the attached source figure",
                            "deliverables": ["visual evidence report"],
                        }
                    )
                }
            )
        }
    )
    calls = 0

    async def fake_request(_endpoint, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient critic outage")
        return VisualEvidenceReport(
            observations=[
                VisualEvidenceObservation(
                    artifact_path=kwargs["payload"]["visual_input_order"][0],
                    observed_content="A source chart is visible.",
                    scientific_interpretation=(
                        "The chart can support a bounded visual description."
                    ),
                )
            ]
        )

    monkeypatch.setattr(
        "scientific_agent.orchestrator.request_structured", fake_request
    )
    settings = Settings(workspace=workspace)
    run_dir = tmp_path / "run"

    first = await orchestrator_module._review_input_visual_evidence(
        settings,
        planning,
        ComputationEvidence(),
        "bounded research context",
        run_dir,
    )
    second = await orchestrator_module._review_input_visual_evidence(
        settings,
        planning,
        ComputationEvidence(),
        "bounded research context",
        run_dir,
    )

    assert calls == 2
    assert first.observations == []
    assert len(second.observations) == 1
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert audit["batches_succeeded"] == 1


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
        introduction="The draft addresses a test objective.",
        methods=["Test method"],
        results="The draft result remains provisional.",
        discussion="The draft interpretation remains provisional.",
        conclusions="No supported conclusion is drawn.",
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


def test_compact_computation_summary_includes_only_cited_verified_json(tmp_path):
    cited_path = tmp_path / "cited.json"
    cited_path.write_text('{"estimate": 5.0, "ci": [4.1, 5.9]}', encoding="utf-8")
    unreferenced_path = tmp_path / "unreferenced.json"
    unreferenced_path.write_text('{"estimate": 999}', encoding="utf-8")
    cited = ArtifactRef(
        path=str(cited_path),
        sha256=sha256_file(cited_path),
        description="sandbox-generated analysis artifact",
    )
    unreferenced = ArtifactRef(
        path=str(unreferenced_path),
        sha256=sha256_file(unreferenced_path),
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="python",
                code_sha256="c" * 64,
                started_at="2026-07-17T20:00:00Z",
                duration_seconds=1.0,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[cited, unreferenced],
            )
        ],
        artifacts=[cited, unreferenced],
    )

    summary = _compact_computation_summary(
        computation,
        referenced_json_paths={str(cited_path)},
    )

    assert summary["referenced_json_values"] == [
        {
            "path": str(cited_path),
            "sha256": sha256_file(cited_path),
            "bytes": cited_path.stat().st_size,
            "value": {"estimate": 5.0, "ci": [4.1, 5.9]},
        }
    ]
    assert summary["referenced_json_unavailable"] == []


def test_compact_computation_summary_rejects_hash_mismatched_json(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text('{"estimate": 5.0}', encoding="utf-8")
    result = ArtifactRef(
        path=str(result_path),
        sha256="0" * 64,
        description="sandbox-generated analysis artifact",
    )
    computation = ComputationEvidence(
        successful_calls=1,
        records=[
            ComputationRecord(
                execution_id="exec-001",
                language="r",
                code_sha256="d" * 64,
                started_at="2026-07-17T20:00:00Z",
                duration_seconds=1.0,
                exit_code=0,
                status="succeeded",
                stdout_path=str(tmp_path / "stdout.txt"),
                stderr_path=str(tmp_path / "stderr.txt"),
                artifacts=[result],
            )
        ],
        artifacts=[result],
    )

    summary = _compact_computation_summary(
        computation,
        referenced_json_paths={str(result_path)},
    )

    assert summary["referenced_json_values"] == []
    assert summary["referenced_json_unavailable"] == [
        {"path": str(result_path), "reason": "sha256_mismatch"}
    ]


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


def test_simple_planning_uses_one_qwen_plan_and_bounded_gemma_audit(monkeypatch):
    calls = []
    audits = []

    async def fake_request(*_args, **kwargs):
        calls.append(kwargs["system_prompt"])
        return _plan("MASTER")

    async def fake_audit(_settings, master, _visible=None):
        audits.append(master)
        return VerificationReport(verdict="pass")

    monkeypatch.setattr("scientific_agent.workflow.request_structured", fake_request)
    monkeypatch.setattr("scientific_agent.workflow.audit_master_plan", fake_audit)
    import asyncio

    controller_report = (
        "Evidence-backed scientific report with claim and source ledgers"
    )
    task = _task().model_copy(update={"deliverables": [controller_report]})
    result = asyncio.run(build_simple_planning(Settings(), task))
    assert result.status == "supported"
    assert len(calls) == 1
    assert audits == [result.master_plan]
    assert result.audit.verdict == "pass"
    assert controller_report in result.master_plan.plan.expected_artifacts


def test_simple_planning_requires_revision_for_a_concrete_plan_audit_failure(
    monkeypatch,
):
    async def fake_request(*_args, **_kwargs):
        return _plan("MASTER")

    async def fake_audit(_settings, _master, _visible=None):
        return VerificationReport(
            verdict="fail",
            blocking_findings=[
                Finding(
                    finding_id="plan-method",
                    location="plan.steps[0]",
                    problem="The method is not operationally defined.",
                    why_it_matters="The analysis would not be reproducible.",
                    evidence="The plan names a heuristic without defining it.",
                    falsification_test_or_correction="Define and validate the heuristic.",
                )
            ],
        )

    monkeypatch.setattr("scientific_agent.workflow.request_structured", fake_request)
    monkeypatch.setattr("scientific_agent.workflow.audit_master_plan", fake_audit)

    result = asyncio.run(build_simple_planning(Settings(), _task()))

    assert result.status == "requires_revision"


def test_presentation_only_repair_does_not_require_new_research():
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="unregistered_report_artifact",
                message="Register the existing figure.",
                location="/run/output/figures/effect.png",
            ),
            LintFinding(
                code="non_display_artifact_in_reader_facing_folder",
                message="Move machine JSON below output/data.",
                location="/run/output/tables/results.json",
                blocking=False,
            ),
        ],
    )
    assert _is_presentation_only_repair(
        validation,
        VerificationReport(verdict="inconclusive"),
    )
    assert not _is_presentation_only_repair(
        validation,
        VerificationReport(
            verdict="inconclusive",
            proposed_falsification_tests=[
                CheckSpec(
                    check_id="recompute",
                    description="Recompute the endpoint.",
                    check_type="test",
                )
            ],
        ),
    )
    assert not _is_presentation_only_repair(
        validation,
        VerificationReport(
            verdict="fail",
            blocking_findings=[
                Finding(
                    finding_id="bad-figure",
                    location="Figure 1",
                    problem="The plotted groups overlap and obscure the estimate.",
                    why_it_matters="The reader cannot recover the result.",
                    evidence="The supplied raster visibly overlaps.",
                    falsification_test_or_correction="Regenerate the figure.",
                )
            ],
        ),
    )


def test_concrete_critic_failure_requires_repair_even_when_validation_passes():
    validation = DeterministicValidation(passed=True)
    blocking = Finding(
        finding_id="figure-typo",
        location="Figure 1 title",
        problem="The title contains a typographical error.",
        why_it_matters="The display is not publication quality.",
        evidence="The rendered title reads 'Analasys'.",
        falsification_test_or_correction="Correct and rerender the title.",
    )

    assert _needs_repair(
        validation,
        VerificationReport(verdict="fail", blocking_findings=[blocking]),
    )
    assert _needs_repair(
        validation,
        VerificationReport(verdict="inconclusive", blocking_findings=[blocking]),
    )
    assert not _needs_repair(
        validation,
        VerificationReport(
            verdict="pass_with_nonblocking_comments",
            nonblocking_findings=[blocking],
        ),
    )


def test_deterministic_only_repair_exhaustion_requires_human_decision():
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="table_excessive_precision",
                location="Table 1",
                message="Reader-facing precision remains excessive.",
            )
        ],
    )
    review = VerificationReport(verdict="pass")

    assert (
        _final_run_status(validation, review, quality_gate_exhausted=True)
        == "requires_human_decision"
    )


def test_research_repair_instruction_disambiguates_significant_digits():
    guidance = orchestrator_module.TABLE_PRECISION_REPAIR_GUIDANCE

    assert "at most four significant digits, not four decimal places" in guidance
    assert "10.897 -> 10.9 or 10.90" in guidance
    assert "utils::zip(zipfile=target_zip, files=files)" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )
    assert "compare basename(zip_list$Name)" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )
    assert "scales::label_number_auto()" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )
    assert "default digits=4 is not full precision" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )
    assert "do not force a distant zero" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )
    assert "never a chained lower < estimate < upper" in (
        orchestrator_module.R_REPAIR_EXECUTION_GUIDANCE
    )


@pytest.mark.anyio
async def test_schema_invalid_repair_preserves_audited_report_and_escalates(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    runs = tmp_path / "runs"
    workspace.mkdir()
    runs.mkdir()
    planning, report = _display_audit_fixture()
    validation = DeterministicValidation(
        passed=False,
        findings=[
            LintFinding(
                code="literature_source_not_locally_acquired",
                location="sources[0]",
                message="The cited article lacks an acquired record.",
            )
        ],
    )
    review = VerificationReport(
        verdict="fail",
        blocking_findings=[
            Finding(
                finding_id="source-provenance",
                location="sources[0]",
                problem="The source provenance is not admissible.",
                why_it_matters="The claim cannot be independently checked.",
                evidence="No controller-verified local article record exists.",
                falsification_test_or_correction="Acquire the article or remove the claim.",
            )
        ],
    )
    produce_calls = 0
    audit_calls = 0

    async def fake_planning(*_args, **_kwargs):
        return planning

    async def fake_produce(*_args, **_kwargs):
        nonlocal produce_calls
        produce_calls += 1
        if produce_calls == 1:
            return report, RetrievalEvidence(), ComputationEvidence(), (), ""
        raise RuntimeError("schema-invalid repair")

    async def fake_audit(*_args, **_kwargs):
        nonlocal audit_calls
        audit_calls += 1
        return review

    monkeypatch.setattr(orchestrator_module, "build_simple_planning", fake_planning)
    monkeypatch.setattr(orchestrator_module, "_produce_report", fake_produce)
    monkeypatch.setattr(
        orchestrator_module, "validate_report", lambda *_a, **_k: validation
    )
    monkeypatch.setattr(orchestrator_module, "_audit_report_resilient", fake_audit)

    result = await orchestrator_module.run_scientific_task(
        "Produce a report",
        Settings(
            workspace=workspace,
            runs_dir=runs,
            max_repair_rounds=4,
            mcp_servers=(),
        ),
        mcp_names=(),
        simple_mode=True,
    )

    provenance = Path(result.provenance_dir)
    failure = json.loads(
        (provenance / "repair_model_unavailable.json").read_text(encoding="utf-8")
    )
    assert result.status == "requires_human_decision"
    assert result.report == report
    assert result.repair_rounds == 1
    assert produce_calls == 2
    assert audit_calls == 1
    assert failure["error_type"] == "RuntimeError"
    assert failure["last_deterministic_finding_codes"] == [
        "literature_source_not_locally_acquired"
    ]
    assert (provenance / "scientific_report.json").is_file()


def test_display_ids_are_removed_from_claim_evidence_refs():
    report = ScientificReport(
        title="Effect report",
        executive_summary="A supported effect was estimated.",
        introduction="This analysis estimates an effect.",
        methods=["A prespecified calculation was used."],
        results="Figure 1 shows the effect.",
        discussion="The result remains exploratory.",
        conclusions="The evidence supports an exploratory result.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                title="Effect estimate",
                caption="Effect estimate with uncertainty.",
                artifact_path="/run/output/figures/effect.png",
                alt_text="An effect plot with estimate and uncertainty.",
            )
        ],
        claims=[
            ClaimRecord(
                claim_id="effect",
                text="The effect was estimated.",
                claim_type="computed",
                evidence_refs=["result-source", "effect-figure"],
                status="supported",
            )
        ],
        sources=[
            SourceRecord(
                source_id="result-source",
                title="Result artifact",
                artifact_path="/run/output/result.json",
                source_type="dataset",
                retrieved_at="2026-07-14T00:00:00Z",
                supporting_passage="Computed result artifact.",
            )
        ],
    )

    normalized = _remove_display_ids_from_claim_evidence(report)

    assert normalized.claims[0].evidence_refs == ["result-source"]
    assert report.claims[0].evidence_refs == ["result-source", "effect-figure"]


def test_declared_display_mentions_are_added_without_new_claims():
    report = ScientificReport(
        title="Effect report",
        executive_summary="A supported effect was estimated.",
        introduction="This analysis estimates an effect.",
        methods=["A prespecified calculation was used."],
        results="The effect estimate was computed.",
        discussion="The result remains exploratory.",
        conclusions="The evidence supports an exploratory result.",
        displays=[
            ReportDisplay(
                display_id="effect-figure",
                kind="figure",
                placement="results",
                title="Effect estimate",
                caption="Effect estimate with uncertainty.",
                artifact_path="/run/output/figures/effect.png",
                alt_text="An effect plot with estimate and uncertainty.",
            ),
            ReportDisplay(
                display_id="effect-table",
                kind="table",
                placement="results",
                title="Exact estimates",
                caption="Exact effect estimates.",
                artifact_path="/run/output/tables/effect.csv",
            ),
        ],
        claims=[],
        sources=[],
    )

    normalized = _ensure_declared_display_mentions(report)

    assert "Figure 1" in normalized.results
    assert "Table 1" in normalized.results
    assert normalized.claims == []
    assert report.results == "The effect estimate was computed."
