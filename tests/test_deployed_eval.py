import json

from evals.run_deployed_eval import score


class _Artifacts:
    def __init__(self, values):
        self.values = values

    def download_artifact(self, run_id, path):
        assert run_id == "run-1"
        return json.dumps(self.values[path]).encode()


def test_known_effect_score_reads_nested_run_result_and_checks_fixture():
    python_path = "computations/a/exec-1/output/results/python_analysis.json"
    r_path = "computations/b/exec-1/output/results/r_analysis.json"
    reconcile_path = (
        "computations/b/exec-1/output/reconciliation/reconciliation_report.json"
    )
    figure_path = "computations/a/exec-1/output/figures/primary.png"
    generated = [
        {"path": python_path, "description": "sandbox-generated analysis artifact"},
        {"path": r_path, "description": "sandbox-generated analysis artifact"},
        {"path": reconcile_path, "description": "sandbox-generated analysis artifact"},
        {"path": figure_path, "description": "sandbox-generated analysis artifact"},
    ]
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
                python_path,
                r_path,
                reconcile_path,
                figure_path,
            )
        ],
        "report": {
            "claims": [
                {
                    "claim_type": "computed",
                    "evidence_refs": ["python"],
                }
            ],
            "sources": [{"source_id": "python", "artifact_path": python_path}],
        },
        "result": {
            "deterministic_validation": {"passed": True},
            "scientific_review": {"verdict": "pass"},
            "computation_evidence": {
                "records": [
                    {"language": "python", "status": "succeeded", "artifacts": generated[:1]},
                    {"language": "r", "status": "succeeded", "artifacts": generated[1:3]},
                ],
                "artifacts": generated,
            },
        },
    }
    client = _Artifacts(
        {
            python_path: {
                "mean_difference": {"value": 5.0},
                "effect_size": {"hedges_g": 3.3775483697},
            },
            r_path: {
                "mean_difference": {"value": 5.0},
                "effect_size": {"hedges_g": 3.3775483697},
            },
            reconcile_path: {
                "primary_metrics": {
                    "mean_diff": {"absolute_difference": 0.0},
                    "welch_t_statistic": {"absolute_difference": 1e-14},
                },
                "all_pass": True,
            },
        }
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
