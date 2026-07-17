from scientific_agent.prompts import (
    DISPLAY_AUDITOR,
    INPUT_VISUAL_AUDITOR,
    PLAN_REPAIRER,
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
    assert (
        "Never abbreviate an unstandardized mean difference as bare `d`" in RESEARCHER
    )
    assert "why -> what -> local meaning" in REPORTER
    assert "Reserve mechanisms, broad\nclinical or biological implications" in REPORTER
    assert "missing component or Discussion-style drift" in REPORT_AUDITOR
    assert "workspace-relative" in RESEARCHER
    assert "inside Python/R sandbox code" in RESEARCHER
    assert "exact controller-registered absolute host path" in RESEARCHER
    assert "Arbitrary absolute paths remain denied" in RESEARCHER
    assert "only for code submitted to run_python_analysis" in RESEARCHER
    assert "parent.mkdir(parents=True, exist_ok=True)" in RESEARCHER
    assert "dir.create(dirname(target), recursive=TRUE" in RESEARCHER
    assert "Never reconstruct, hand-copy, or hard-code subject-level" in RESEARCHER
    assert "not at a placeholder coordinate plus a text label" in RESEARCHER
    assert "never copy a pooled or\nwhole-cohort diagnostic" in RESEARCHER
    assert "an R validation call must not plot" in RESEARCHER
    assert "independent numeric JSON and reconciliation first" in RESEARCHER
    assert "exit successfully without display generation" in RESEARCHER
    assert "never\ncombine the only required Python/R validation result" in RESEARCHER
    assert "candidate_role_labels" in SIMPLE_PLANNER
    assert "never copy example labels from the audit" in PLAN_REPAIRER
    assert "`4*N - 9` is not `4*(N - 9)`" in PLAN_REPAIRER
    assert "Never classify a dataset as observational" in PLANNER_A
    assert "use\ndesign-unspecified language" in SIMPLE_PLANNER
    assert "require design-unspecified language instead" in PLAN_AUDITOR
    assert "including booleans, to native Python scalars" in RESEARCHER
    assert "`jsonlite::write_json(..., auto_unbox = TRUE)`" in RESEARCHER
    assert "never length-one arrays such as `[5.0]`" in RESEARCHER
    assert "never emit a non-null full_text_status" in REPORTER


def test_researcher_documents_matplotlib_hlines_return_type():
    assert "Every object key must be a string" in RESEARCHER
    assert "MultiIndex or group-by tuple keys" in RESEARCHER
    assert "`Axes.errorbar()` accepts `linewidth` or `elinewidth`" in RESEARCHER
    assert "never the scatter-style\n`linewidths` keyword" in RESEARCHER
    assert "`get_xdata()` and `get_ydata()`, not `get_xy()`" in RESEARCHER
    assert "`Axes.hlines()` returns one `LineCollection`" in RESEARCHER
    assert "inspect `get_segments()`" in RESEARCHER
    assert "cross-check every reported test statistic" in REPORT_AUDITOR
    assert "referenced_json_values" in REPORT_AUDITOR
    assert "bounded, hash-verified values" in REPORT_AUDITOR
    assert "horizontal versus\nvertical bars" in DISPLAY_AUDITOR
    assert "zero-spread group alongside a nonzero reported\nSD" in DISPLAY_AUDITOR
    assert "must never be shifted or duplicated around\nboth group means" in (
        DISPLAY_AUDITOR
    )
    assert "Do not use `twinx()`, `twiny()`" in RESEARCHER
    assert "separate, plainly labeled effect-estimate panel" in RESEARCHER
    assert "Treat a twin/secondary axis over the raw-data panel as blocking" in (
        DISPLAY_AUDITOR
    )
    assert "jitter only the categorical position coordinate" in RESEARCHER
    assert "consecutive integer centers" in RESEARCHER
    assert "jitter\nenvelopes overlap" in RESEARCHER
    assert "point estimate, and both confidence-interval endpoints" in RESEARCHER
    assert "without `openssl`, `digest`" in RESEARCHER
    assert "shifted or duplicated contrast interval" in REPORT_AUDITOR
    assert "Shared boolean quality-control\nfields" in RESEARCHER
    assert '`license: "unknown"` is data' in REPORTER
    assert "preserve literal `license:" in REPAIRER
    assert "pmid, pmcid, citekey, license, rights_status" in REPORTER


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


def test_planning_prompts_forbid_order_based_semantic_arm_assignment():
    assert "never assign control or" in SIMPLE_PLANNER
    assert "Never propose lexical" in PLAN_AUDITOR
    assert "Never resolve semantic arm identity" in PLAN_REPAIRER
    for prompt in (SIMPLE_PLANNER, PLAN_AUDITOR, PLAN_REPAIRER):
        normalized = " ".join(prompt.split())
        assert "observed baseline" in normalized
        assert "effect direction" in normalized
        assert "Shapiro-Wilk" in normalized
        assert "sensitivity analysis" in normalized


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
    assert (
        "Every substantive statement supported by a knowledge-base passage" in REPORTER
    )
    assert "controller renders validated Vancouver-style linked numbers" in REPORTER


def test_reporter_forbids_browser_snapshots_from_impersonating_acquired_articles():
    assert "Chrome snapshot\nhash" in REPORTER
    assert "Never translate a browser snapshot" in REPORTER
    assert "do not reclassify\nit as web_page" in REPORTER
    assert "Never downgrade a\nDOI-bearing scholarly article" in REPAIRER


def test_reporter_avoids_unsupported_design_and_clinical_framing():
    assert "do not introduce intervention language" in REPORTER
    assert "do not introduce clinical-importance language" in REPORTER
    assert "Group means do not establish a uniform" in REPORTER
    assert "individual-level values" in REPORTER


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


def test_report_auditor_does_not_invent_submission_readiness_rules():
    assert "Do not import generic journal, submission-package" in REPORT_AUDITOR
    assert "Never combine manuscript and supplement word counts" in REPORT_AUDITOR
    assert "A placeholder explicitly permitted" in REPORT_AUDITOR
    assert "unresolved nonblocking question" in REPORT_AUDITOR
    assert "fixed schema with task-specific top-level headings" in REPORT_AUDITOR
    assert "deterministic display validator is authoritative" in REPORT_AUDITOR
    assert "never demand more decimal" in REPORT_AUDITOR


def test_display_auditor_does_not_force_ocr_contradicted_typo_repairs():
    assert "controller OCR disagree" in DISPLAY_AUDITOR
    assert "return inconclusive for that label" in DISPLAY_AUDITOR


def test_display_auditor_requires_explicit_per_figure_layout_clearance():
    assert "layout_review_questions" in DISPLAY_AUDITOR
    assert (
        "deterministic attention\nsignals, not pixel interpretations" in DISPLAY_AUDITOR
    )
    assert "display-reviewed:<display_id>" in DISPLAY_AUDITOR
    assert "visual-clearance:<display_id>:top-text" in DISPLAY_AUDITOR
    assert "visual-clearance:<display_id>:legend-data" in DISPLAY_AUDITOR
    assert "visual-clearance:<display_id>:annotation-data" in DISPLAY_AUDITOR
    assert "primary Qwen agent never receives raster images" in DISPLAY_AUDITOR
    assert "A bare pass" in DISPLAY_AUDITOR
    assert "do not\ninvent `findings`, `findings_list`" in DISPLAY_AUDITOR
    assert "must\ncontain at least one complete `blocking_findings` object" in (
        DISPLAY_AUDITOR
    )


def test_report_contract_does_not_register_extracted_source_images_as_displays():
    for prompt in (REPORTER, REPAIRER):
        assert "logical /output/figures or /output/tables" in prompt
        assert "archive extraction copy" in prompt
        assert "intermediate visual-review raster" in prompt


def test_report_writer_respects_non_readiness_scope_and_separate_word_counts():
    for prompt in (REPORTER, REPAIRER):
        assert "excludes submission readiness" in prompt
        assert "Never combine main-manuscript and supplement word counts" in prompt
        assert "placeholder explicitly permitted" in prompt
        assert "confirmatory, exploratory, or decision-critical status" in prompt
        assert "Never relabel one as another" in prompt


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
