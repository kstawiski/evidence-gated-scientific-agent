import asyncio
import json
import threading
from pathlib import Path

import pytest
from PIL import Image

import scientific_agent.orchestrator as orchestrator_module
from scientific_agent.config import Settings
from scientific_agent.orchestrator import (
    _audit_report_resilient,
    _can_continue_after_research_error,
    _compact_computation_summary,
    _final_run_status,
    _is_presentation_only_repair,
    _load_ancestor_protocol_artifacts,
    _ensure_declared_display_mentions,
    _fallback_evidence_packet,
    _merge_computation_evidence,
    _merge_reviews,
    _needs_repair,
    _merge_retrieval_evidence,
    _prepare_task_spec,
    _remove_display_ids_from_claim_evidence,
    _requires_pubmed_literature,
    _write_attempt_bundle,
    ResearchBudgetController,
    ResearchBudgetExceeded,
)
from scientific_agent.provenance import EventLedger
from scientific_agent.schemas import (
    ArtifactRef,
    CheckSpec,
    ComputationEvidence,
    ComputationRecord,
    DeterministicValidation,
    Finding,
    MasterPlan,
    LintFinding,
    PLAN_AUDIT_CRITERIA,
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
    PLAN_CRITIC_UNAVAILABLE,
    audit_master_plan,
    bind_controller_task,
    build_plan_audit_packet,
    build_simple_planning,
    build_planning_workflow,
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
        ),
        plan_lints=[],
        status="requires_revision",
    )

    repaired = await orchestrator_module._repair_plan(Settings(), initial)

    assert repaired.master_plan.task == controller_task
    assert repaired.status == "supported"
    assert "every concrete blocking" in seen["system_prompt"]


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


def test_research_budget_exhaustion_can_continue_only_with_existing_evidence():
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
        return VerificationReport(verdict="pass")

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
    assert calls == 2
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
        return VerificationReport(verdict="pass")

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
    assert "sole image-understanding scientist" in calls[0][1]["system_prompt"]
    assert any("visual-proof.pdf" in item for item in first.unreviewed_requests)
    audit = json.loads((run_dir / "gemma_input_visual_review.json").read_text())
    assert audit["critic_model"] == settings.gemma.model
    assert audit["qwen_image_inputs"] == 0
    assert audit["batches_attempted"] == 1
    assert audit["batches_succeeded"] == 1


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
            )
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

    async def fake_planning(*_args, **_kwargs):
        return planning

    async def fake_produce(*_args, **_kwargs):
        nonlocal produce_calls
        produce_calls += 1
        if produce_calls == 1:
            return report, RetrievalEvidence(), ComputationEvidence(), ()
        raise RuntimeError("schema-invalid repair")

    async def fake_audit(*_args, **_kwargs):
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
