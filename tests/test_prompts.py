from scientific_agent.prompts import (
    INPUT_VISUAL_AUDITOR,
    PLAN_AUDITOR,
    PLANNER_A,
    PLANNER_B,
    REPORT_DISCUSSION,
    REPORT_AUDITOR,
    REPORTER,
    REPAIRER,
    RESEARCHER,
    SIMPLE_REPORTER,
    SIMPLE_PLANNER,
)


def test_researcher_forbids_mixed_effect_scales_and_zero_rounded_p_values():
    assert "never plot an\nunstandardized estimate" in RESEARCHER
    assert "Hedges g" in RESEARCHER
    assert "never as zero after fixed-decimal rounding" in RESEARCHER


def test_planners_do_not_invent_input_names_or_qwen_visual_audits():
    for prompt in (PLANNER_A, PLANNER_B, SIMPLE_PLANNER):
        assert "filename" in prompt
        assert "Qwen cannot" in prompt
        assert "image pixels" in prompt
        assert "controller-routed" in prompt
        assert "Gemma audit" in prompt
    assert "Do not list a Gemma audit as a\nQwen-produced output" in SIMPLE_PLANNER


def test_plan_auditor_preserves_controller_ownership_of_visual_checkpoint():
    assert "controller automatically routes bounded rasters" in PLAN_AUDITOR
    assert "never ask Qwen to interpret pixels" in PLAN_AUDITOR
    assert "not as a model-generated output artifact" in PLAN_AUDITOR


def test_input_visual_auditor_requires_exact_schema_and_artifact_paths():
    assert "short controller-issued artifact_path identifier" in INPUT_VISUAL_AUDITOR
    assert "`observed_content`" in INPUT_VISUAL_AUDITOR
    assert "`scientific_interpretation`" in INPUT_VISUAL_AUDITOR
    assert "and `concerns`" in INPUT_VISUAL_AUDITOR
    assert "do not substitute `observation`" in INPUT_VISUAL_AUDITOR
    assert "Do not reproduce a host filesystem path" in INPUT_VISUAL_AUDITOR


def test_researcher_encourages_each_default_mcp_without_assuming_availability():
    assert "Research connections are normally enabled by default" in RESEARCHER
    assert "When available\nin this run" in RESEARCHER


def test_researcher_prioritizes_locked_computation_over_optional_retrieval():
    assert (
        "after at most one PubMed\nsearch and one article-acquisition call"
        in RESEARCHER
    )
    assert "every required computation language" in RESEARCHER
    assert "before retrieving optional additional papers" in RESEARCHER
    assert "Context7" in RESEARCHER
    assert "Brave Search" in RESEARCHER
    assert "Chrome DevTools" in RESEARCHER
    assert "do not call\nan irrelevant connection mechanically" in RESEARCHER


def test_researcher_broadens_zero_hit_pubmed_queries_before_giving_up():
    assert "two to four discriminating\nconcepts" in RESEARCHER
    assert (
        "If a search returns\nzero articles, retry with a materially broader query"
        in RESEARCHER
    )
    assert "bounded to three distinct\nqueries" in RESEARCHER


def test_researcher_does_not_promote_secondary_bibliographies_to_retrieved_sources():
    assert "merely appears in a retrieved page's bibliography" in RESEARCHER
    assert "a lead, not a verified source record" in RESEARCHER
    assert "unless a tool separately returned that record or its content" in RESEARCHER


def test_research_packet_cannot_self_certify_the_final_report():
    assert (
        "The research packet is evidence input, not the scientific article"
        in RESEARCHER
    )
    assert 'declarations such as "complete", "validated", "pass"' in RESEARCHER
    assert "never self-certify a calculation validator" in RESEARCHER
    assert "Treat any researcher-authored labels" in REPORTER
    assert "never copy a self-certified validator result" in REPORTER


def test_reporter_keeps_literature_acquisition_fields_off_artifact_sources():
    assert "For every artifact-backed SourceRecord" in REPORTER
    assert "full_text_status" in REPORTER
    assert "fields apply only to external literature records" in REPORTER


def test_reporter_forbids_browser_snapshots_from_impersonating_acquired_articles():
    assert "Chrome snapshot\nhash" in REPORTER
    assert "Never translate a browser snapshot" in REPORTER
    assert "do not reclassify\nit as web_page" in REPORTER
    assert "Never downgrade a\nDOI-bearing scholarly article" in REPAIRER


def test_reporter_avoids_unsupported_design_and_clinical_framing():
    assert "do not introduce intervention language" in REPORTER
    assert "do not introduce clinical-importance language" in REPORTER


def test_simple_reporter_does_not_suppress_material_limitations_for_brevity():
    assert "Include every material limitation" in SIMPLE_REPORTER
    assert "prefer scientific completeness" in SIMPLE_REPORTER
    assert "no more than two short limitations" not in SIMPLE_REPORTER


def test_report_auditor_accepts_controller_protocol_authority():
    assert (
        "controller is\nthe expected authority for protocol locking" in REPORT_AUDITOR
    )
    assert (
        "never require protocol.json to be\ncreated by a sandbox computation"
        in REPORT_AUDITOR
    )


def test_report_contract_scopes_method_recommendations_and_literature_reviews():
    assert '"robust default" language' in REPORTER
    assert "literature-only evidence synthesis" in REPORTER
    assert "block invented review methods" in REPORT_AUDITOR


def test_report_contract_requires_independent_equation_verification():
    assert "equations, algebraic reductions, boundary conditions" in REPORTER
    assert "Equality of one intermediate term" in REPORTER
    assert "recheck every reported equation" in REPORT_AUDITOR
    assert "tautological or duplicated equation operands" in REPORT_AUDITOR


def test_report_discussion_preserves_material_report_uncertainty():
    assert "a scientifically supported report can still" in REPORT_DISCUSSION
    assert "Do not leave this list empty merely because" in REPORT_DISCUSSION
    assert "already explicit, claim-bounding inherent limitation" in REPORT_DISCUSSION
