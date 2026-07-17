import hashlib
import json

from evals.run_deployed_eval import (
    _known_effect_matches_reference,
    _language_result,
    _metric,
    _reconciliation_delta,
    score,
)


class _Artifacts:
    def __init__(self, values, parent_detail=None):
        self.values = values
        self.parent_detail = parent_detail

    def download_artifact(self, run_id, path):
        assert run_id in {"run-1", "run-parent"}
        return json.dumps(self.values[path]).encode()

    def request(self, method, path):
        assert method == "GET"
        assert path == "/api/runs/run-parent"
        assert self.parent_detail is not None
        return self.parent_detail


class _ByteArtifacts:
    def __init__(self, values):
        self.values = values

    def download_artifact(self, run_id, path):
        assert run_id == "run-1"
        return self.values[path]


def test_score_handles_terminal_inconclusive_run_with_null_result_sections():
    detail = {
        "id": "run-1",
        "status": "inconclusive",
        "artifacts": [],
        "report": None,
        "result": {
            "deterministic_validation": None,
            "scientific_review": None,
            "computation_evidence": None,
        },
    }

    result = score(
        "known-effect",
        {"enable_code": False},
        None,
        detail,
        _Artifacts({}),
    )

    assert result["passed"] is False
    assert result["checks"]["terminal_status"] is True
    assert result["checks"]["deterministic_validation"] is False


def test_language_result_prefers_complete_reference_over_display_values():
    display_path = "computations/a/output/data/display_values.json"
    result_path = "computations/a/output/results/python/primary.json"
    complete = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference_treatment_minus_control": 5.0,
        "t_statistic": 10.897247358851683,
        "welch_df": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "j_correction": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    computation = {
        "records": [
            {
                "language": "python",
                "status": "succeeded",
                "artifacts": [
                    {
                        "path": display_path,
                        "description": "sandbox-generated analysis artifact",
                    },
                    {
                        "path": result_path,
                        "description": "sandbox-generated analysis artifact",
                    },
                ],
            }
        ]
    }

    selected = _language_result(
        computation,
        {
            display_path: {"mean_difference": 5.0, "hedges_g": 3.38},
            result_path: complete,
        },
        "python",
    )

    assert selected is complete


def test_language_result_follows_reconciliation_artifact_hash():
    full_path = "computations/a/output/data/python_results.json"
    rounded_path = "computations/b/output/data/rounded_results.json"
    full = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference": 5.0,
        "welch_t_statistic": 10.897247358851683,
        "degrees_of_freedom": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "hedges_g_correction_J": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    rounded = {"mean_difference": 5.0, "hedges_g": 3.378}
    computation = {
        "records": [
            {
                "language": "python",
                "status": "succeeded",
                "artifacts": [
                    {
                        "path": full_path,
                        "sha256": "a" * 64,
                        "description": "sandbox-generated analysis artifact",
                    },
                    {
                        "path": rounded_path,
                        "sha256": "b" * 64,
                        "description": "sandbox-generated analysis artifact",
                    },
                ],
            }
        ]
    }
    reconciliation = {
        "comparisons": [
            {"python": {"artifact_sha256": "a" * 64}},
        ]
    }

    selected = _language_result(
        computation,
        {rounded_path: rounded, full_path: full},
        "python",
        reconciliation,
    )

    assert selected is full


def test_language_result_uses_complete_bound_artifact_independent_of_order():
    partial_path = "computations/a/output/data/partial.json"
    complete_path = "computations/b/output/data/complete.json"
    partial = {"mean_difference": 5.0, "hedges_g": 3.378}
    complete = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference": 5.0,
        "welch_t_statistic": 10.897247358851683,
        "degrees_of_freedom": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "hedges_g_correction_J": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    computation = {
        "records": [
            {
                "language": "python",
                "status": "succeeded",
                "artifacts": [
                    {
                        "path": partial_path,
                        "sha256": "a" * 64,
                        "description": "sandbox-generated analysis artifact",
                    },
                    {
                        "path": complete_path,
                        "sha256": "b" * 64,
                        "description": "sandbox-generated analysis artifact",
                    },
                ],
            }
        ]
    }
    reconciliation = {
        "comparisons": [
            {"python": {"artifact_sha256": "b" * 64}},
            {"python": {"artifact_sha256": "a" * 64}},
        ]
    }

    selected = _language_result(
        computation,
        {partial_path: partial, complete_path: complete},
        "python",
        reconciliation,
    )

    assert selected is complete


def test_language_result_fails_closed_for_unresolved_bound_digest():
    result_path = "computations/a/output/data/complete.json"
    complete = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference": 5.0,
        "welch_t_statistic": 10.897247358851683,
        "degrees_of_freedom": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "hedges_g_correction_J": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    computation = {
        "records": [
            {
                "language": "python",
                "status": "succeeded",
                "artifacts": [
                    {
                        "path": result_path,
                        "sha256": "a" * 64,
                        "description": "sandbox-generated analysis artifact",
                    }
                ],
            }
        ]
    }
    reconciliation = {"comparisons": [{"python": {"artifact_sha256": "c" * 64}}]}

    assert (
        _language_result(
            computation,
            {result_path: complete},
            "python",
            reconciliation,
        )
        is None
    )


def test_language_result_fails_closed_when_typed_reconciliation_omits_language():
    result_path = "computations/a/output/data/r_complete.json"
    complete = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference": 5.0,
        "welch_t_statistic": 10.897247358851683,
        "degrees_of_freedom": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "hedges_g_correction_J": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    computation = {
        "records": [
            {
                "language": "r",
                "status": "succeeded",
                "artifacts": [
                    {
                        "path": result_path,
                        "sha256": "b" * 64,
                        "description": "sandbox-generated analysis artifact",
                    }
                ],
            }
        ]
    }
    reconciliation = {
        "comparisons": [
            {
                "metric": "mean_difference",
                "python": {"artifact_sha256": "a" * 64},
                "absolute_difference": 0.0,
                "passed": True,
            }
        ]
    }

    assert (
        _language_result(
            computation,
            {result_path: complete},
            "r",
            reconciliation,
        )
        is None
    )


def test_known_effect_accepts_nested_language_artifact_field_names():
    value = {
        "primary": {
            "treatment_n": 20,
            "control_n": 20,
            "treatment_mean_change": 5.0,
            "control_mean_change": 0.0,
            "mean_difference": 5.0,
            "welch_t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
            "ci_95_lower": 4.071144254485707,
            "ci_95_upper": 5.928855745514293,
            "pooled_sd": 1.4509525002200232,
            "j_correction": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        }
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_deployed_python_and_r_field_names():
    value = {
        "primary": {
            "mean_difference": 5.0,
            "t_statistic": 10.897247358851683,
            "df_welch": 38.0,
            "p_value": 2.971749478841818e-13,
            "ci_lower_95": 4.071144254485707,
            "ci_upper_95": 5.928855745514293,
        },
        "effect_size": {
            "pooled_sd": 1.4509525002200232,
            "J_correction": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
        "descriptive": {
            "n_treatment": 20,
            "n_control": 20,
            "mean_change_treatment": 5.0,
            "mean_change_control": 0.0,
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_actual_nested_live_result_shape():
    value = {
        "study_design": {"n_treatment": 20, "n_control": 20},
        "group_summaries": {
            "treatment": {"mean_change": 5.0},
            "control": {"mean_change": 0.0},
        },
        "primary": {
            "point_estimate": 5.0,
            "welch_t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
            "ci_95_lower": 4.071144254485707,
            "ci_95_upper": 5.928855745514293,
            "pooled_sd": 1.4509525002200232,
            "hedges_g_correction_factor_J": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_v041_top_level_welch_result_shape():
    value = {
        "primary": {
            "mean_difference": 5.0,
            "treatment_mean_change": 5.0,
            "control_mean_change": 0.0,
            "ci_95_lower": 4.071144254485707,
            "ci_95_upper": 5.928855745514293,
        },
        "welch_t_test": {
            "t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
        },
        "effect_size": {
            "pooled_sd": 1.4509525002200232,
            "j_correction": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
        "group_summaries": {
            "treatment": {"n": 20, "mean_change": 5.0},
            "control": {"n": 20, "mean_change": 0.0},
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_current_deployed_language_artifact_shape():
    value = {
        "study_design": "two-group pre/post",
        "sample_sizes": {"n_treatment": 20, "n_control": 20},
        "group_descriptives": {
            "treatment": {"mean_change": 5.0},
            "control": {"mean_change": 0.0},
        },
        "primary": {
            "mean_difference": 5.0,
            "t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
            "ci_95_lower": 4.071144254485707,
            "ci_95_upper": 5.928855745514293,
            "pooled_sd": 1.4509525002200232,
            "j_correction": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_rejects_reference_echo_that_conflicts_with_results():
    reference = {
        "n_treatment": 20,
        "n_control": 20,
        "treatment_mean_change": 5.0,
        "control_mean_change": 0.0,
        "mean_difference": 5.0,
        "t_statistic": 10.897247358851683,
        "degrees_of_freedom": 38.0,
        "p_value": 2.971749478841818e-13,
        "ci_95_lower": 4.071144254485707,
        "ci_95_upper": 5.928855745514293,
        "pooled_sd": 1.4509525002200232,
        "j_correction": 0.9801324503311258,
        "hedges_g": 3.3775483697174717,
    }
    value = {
        "reference_check": reference,
        "results": {"primary": {"mean_difference": 99.0}},
    }

    assert _known_effect_matches_reference(value) is False


def test_known_effect_accepts_nested_live_v041_result_shape():
    value = {
        "study_design": {"n_treatment": 20, "n_control": 20},
        "descriptive_statistics": {
            "treatment": {"mean_change": 5.0, "sd_change": 1.4509525002200232},
            "control": {"mean_change": 0.0, "sd_change": 1.4509525002200232},
        },
        "primary": {
            "point_estimate": 5.0,
            "ci_95_percent": {
                "lower": 4.071144254485707,
                "upper": 5.928855745514293,
            },
            "welch_t_test": {
                "t_statistic": 10.897247358851683,
                "degrees_of_freedom": 38.0,
                "p_value_two_sided": 2.971749478841818e-13,
            },
            "hedges_g": {
                "value": 3.3775483697174717,
                "j_correction_factor": 0.9801324503311258,
            },
        },
    }

    assert _known_effect_matches_reference(value) is True
    assert _metric(value, "hedges_g") == 3.3775483697174717


def test_known_effect_accepts_live_effect_size_and_interval_array_shape():
    value = {
        "study_design": {"n_treatment": 20, "n_control": 20},
        "group_summaries": {
            "treatment": {"mean_change": 5.0},
            "control": {"mean_change": 0.0},
        },
        "primary": {
            "point_estimate": 5.0,
            "confidence_interval_95": [
                4.071144254485707,
                5.928855745514293,
            ],
            "t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
        },
        "effect_size": {
            "pooled_sd": 1.4509525002200232,
            "j_correction_factor": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_live_change_mean_group_summary_shape():
    value = {
        "study_design": {"n_treatment": 20, "n_control": 20},
        "group_summaries": {
            "treatment": {"change_mean": 5.0},
            "control": {"change_mean": 0.0},
        },
        "primary": {
            "point_estimate": 5.0,
            "welch_t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
            "ci_95_lower": 4.071144254485707,
            "ci_95_upper": 5.928855745514293,
            "pooled_sd": 1.4509525002200232,
            "hedges_g_correction_factor_J": 0.9801324503311258,
            "hedges_g": 3.3775483697174717,
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_known_effect_accepts_exact_live_groups_primary_shape():
    value = {
        "primary": {
            "point_estimate": 5.0,
            "ci_lower_95": 4.071144254485707,
            "ci_upper_95": 5.928855745514293,
            "t_statistic": 10.897247358851683,
            "degrees_of_freedom": 38.0,
            "p_value": 2.971749478841818e-13,
            "hedges_g": 3.3775483697174717,
            "j_correction": 0.9801324503311258,
            "pooled_sd": 1.4509525002200232,
        },
        "groups": {
            "control": {"n": 20, "mean_change": 0.0},
            "treatment": {"n": 20, "mean_change": 5.0},
        },
    }

    assert _known_effect_matches_reference(value) is True


def test_reconciliation_delta_accepts_typed_comparison_records():
    artifact = {
        "comparisons": [
            {
                "metric": "mean_difference",
                "absolute_difference": 0.0,
                "passed": True,
            },
            {
                "metric": "welch_t_statistic",
                "absolute_difference": 1.8e-14,
                "passed": True,
            },
        ],
        "reconciliation_passed": True,
    }

    assert _reconciliation_delta(artifact, "mean_diff", "mean_difference") == 0.0
    assert (
        _reconciliation_delta(artifact, "t_statistic", "welch_t_statistic") == 1.8e-14
    )


def test_cancelled_run_scores_cleanly_without_requesting_final_artifacts():
    class NoDownloads:
        @staticmethod
        def download_artifact(*args, **kwargs):
            raise AssertionError("cancelled runs must not request final artifacts")

    result = score(
        "pubmed-fulltext",
        {"enable_code": False},
        None,
        {"id": "run-1", "status": "cancelled", "artifacts": []},
        NoDownloads(),
        {
            "event_ids": [1, 2],
            "stream_actors": {"Qwen", "Gemma"},
            "downloaded_while_active": set(),
        },
    )

    assert result["passed"] is False
    assert result["checks"] == {
        "terminal_status": False,
        "live_events_observed": True,
        "qwen_output_streamed": True,
        "gemma_output_streamed": True,
    }


def test_known_effect_score_reads_nested_run_result_and_checks_fixture():
    python_path = "computations/a/exec-1/output/results/python_analysis.json"
    r_path = "computations/b/exec-1/output/results/r_analysis.json"
    reconcile_path = (
        "computations/b/exec-1/output/reconciliation/reconciliation_report.json"
    )
    figure_path = "computations/a/exec-1/output/figures/primary.png"
    table_path = "computations/a/exec-1/output/tables/effects.csv"
    generated = [
        {"path": python_path, "description": "sandbox-generated analysis artifact"},
        {"path": r_path, "description": "sandbox-generated analysis artifact"},
        {"path": reconcile_path, "description": "sandbox-generated analysis artifact"},
        {"path": figure_path, "description": "sandbox-generated analysis artifact"},
        {"path": table_path, "description": "sandbox-generated analysis artifact"},
    ]
    detail = {
        "id": "run-1",
        "parent_run_id": "run-parent",
        "status": "supported",
        "artifacts": [
            {"path": path}
            for path in (
                "environment.json",
                "input_manifest.json",
                "protocol.json",
                "report.md",
                figure_path,
                table_path,
                "gemma_display_audit.json",
            )
        ],
        "report": {
            "executive_summary": "The planted effect was recovered.",
            "introduction": "The fixture evaluates a known contrast.",
            "methods": ["Python and R cross-check"],
            "results": "Figure 1 and Table 1 show the recovered effect.",
            "discussion": "The two implementations agree.",
            "conclusions": "The fixture passes its tolerance.",
            "claims": [
                {
                    "claim_type": "computed",
                    "evidence_refs": ["python"],
                }
            ],
            "sources": [{"source_id": "python", "artifact_path": python_path}],
        },
        "display_manifest": {
            "displays": [
                {"display_id": "primary", "kind": "figure"},
                {"display_id": "effects", "kind": "table"},
            ]
        },
        "result": {
            "deterministic_validation": {"passed": True},
            "scientific_review": {"verdict": "pass"},
            "computation_evidence": {
                "records": [
                    {
                        "language": "python",
                        "status": "succeeded",
                        "artifacts": generated[:1],
                    },
                    {
                        "language": "r",
                        "status": "succeeded",
                        "artifacts": generated[1:3],
                    },
                ],
                "artifacts": generated,
            },
        },
    }
    client = _Artifacts(
        {
            python_path: {
                "n_treatment": 20,
                "n_control": 20,
                "treatment_mean_change": 5.0,
                "control_mean_change": 0.0,
                "mean_difference": 5.0,
                "welch_t_statistic": 10.897247358851683,
                "degrees_of_freedom": 38.0,
                "p_value": 2.971749478841818e-13,
                "ci_95_lower": 4.071144254485707,
                "ci_95_upper": 5.928855745514293,
                "pooled_sd": 1.4509525002200232,
                "hedges_g_correction_J": 0.9801324503311258,
                "hedges_g": 3.3775483697174717,
            },
            r_path: {
                "n_treatment": 20,
                "n_control": 20,
                "treatment_mean_change": 5.0,
                "control_mean_change": 0.0,
                "mean_difference_treatment_minus_control": 5.0,
                "t_statistic": 10.897247358851683,
                "welch_df": 38.0,
                "p_value": 2.971749478841818e-13,
                "ci_95_lower": 4.071144254485707,
                "ci_95_upper": 5.928855745514293,
                "pooled_sd": 1.4509525002200232,
                "j_correction": 0.9801324503311258,
                "hedges_g": 3.3775483697174717,
            },
            reconcile_path: {
                "primary_metrics": {
                    "mean_diff": {"absolute_difference": 0.0},
                    "welch_t_statistic": {"absolute_difference": 1e-14},
                    "hedges_g": {"absolute_difference": 1e-14},
                },
                "all_pass": True,
            },
            "gemma_display_audit.json": {
                "verdict": "pass",
                "review_mode": "raster_with_ocr_geometry_and_table_previews",
                "review_source": "gemma_multimodal_critic",
                "critic_model": "gemma-test",
                "visual_critic": "Gemma",
                "qwen_image_inputs": 0,
                "figure_text_inputs": [
                    {
                        "display_id": "primary",
                        "ocr_available": True,
                        "ocr_character_count": 80,
                        "ocr_text_sha256": "a" * 64,
                    }
                ],
                "table_previews": [{"display_id": "effects"}],
            },
        },
        parent_detail={
            "id": "run-parent",
            "parent_run_id": None,
            "artifacts": [
                {"path": python_path},
                {"path": r_path},
                {"path": reconcile_path},
            ],
        },
    )

    result = score(
        "known-effect",
        {"enable_code": True},
        None,
        detail,
        client,
    )

    assert result["passed"] is True
    assert result["checks"]["planted_effect_recovered"] is True


def test_pubmed_fulltext_score_requires_verified_local_markdown_and_pdf():
    markdown_path = "references/markdown/cox-2026-pmid42158852.md"
    pdf_path = "references/pdfs/cox-2026-pmid42158852.pdf"
    markdown = (
        b"# Ultra-sensitive detection of mutant KRAS\n\n"
        b"PMID 42158852. Mutant KRAS was measured in a cohort of 45 patients.\n"
    )
    pdf = b"%PDF-1.7\n" + b"scientific article" * 800
    report_markdown = (f"[Markdown]({markdown_path})\n[PDF]({pdf_path})\n").encode()
    detail = {
        "id": "run-1",
        "status": "supported",
        "artifacts": [
            {"path": path}
            for path in (
                "environment.json",
                "input_manifest.json",
                "protocol.json",
                "report.md",
                markdown_path,
                pdf_path,
            )
        ],
        "report": {
            "executive_summary": "This PubMed paper reports a prognostic association.",
            "introduction": "The cohort included 45 participants.",
            "methods": ["PubMed acquisition and local article search"],
            "results": "Standard-depth sequencing detected 11 cases; ultra-deep sequencing added 7. The OS HRs were 2.57 (95% CI 0.94-7.04) and 3.13 (95% CI 1.18-8.29).",
            "discussion": "The small cohort limits precision; association does not establish causality.",
            "conclusions": "The result is prognostic and does not establish treatment-predictive utility.",
            "claims": [
                {
                    "claim_type": "literature_supported",
                    "evidence_refs": ["pubmed-42158852"],
                }
            ],
            "sources": [
                {
                    "source_id": "pubmed-42158852",
                    "url": "https://pubmed.ncbi.nlm.nih.gov/42158852/",
                }
            ],
        },
        "reference_manifest": {
            "references": [
                {
                    "source_id": "pubmed-42158852",
                    "pmid": "42158852",
                    "pmcid": "PMC13180577",
                    "doi": "10.3389/fonc.2025.1657746",
                    "markdown": {
                        "path": markdown_path,
                        "sha256": hashlib.sha256(markdown).hexdigest(),
                    },
                    "pdf": {
                        "path": pdf_path,
                        "sha256": hashlib.sha256(pdf).hexdigest(),
                    },
                }
            ]
        },
        "result": {
            "deterministic_validation": {"passed": True},
            "scientific_review": {"verdict": "pass"},
            "retrieval_evidence": {
                "successful_calls": 3,
                "urls": ["https://pubmed.ncbi.nlm.nih.gov/42158852/"],
            },
        },
    }
    client = _ByteArtifacts(
        {markdown_path: markdown, pdf_path: pdf, "report.md": report_markdown}
    )

    result = score(
        "pubmed-fulltext",
        {"enable_code": False},
        None,
        detail,
        client,
    )

    assert result["passed"] is True
    assert result["checks"]["local_article_pdf_stored"] is True
