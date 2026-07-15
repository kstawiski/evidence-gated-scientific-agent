import hashlib
import json

from evals.run_deployed_eval import _language_result, score


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
