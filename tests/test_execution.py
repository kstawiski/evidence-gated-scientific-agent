from dataclasses import replace
from pathlib import Path

import pytest

from scientific_agent.config import SandboxSettings
from scientific_agent.execution import (
    AnalysisExecutor,
    RemoteAnalysisExecutor,
    _python_static_violations,
    _unavailable_prior_reference_violations,
    sandbox_preflight,
)


def _executor(tmp_path: Path, **overrides) -> AnalysisExecutor:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    settings = replace(SandboxSettings(), **overrides)
    return AnalysisExecutor(workspace, tmp_path / "computations", settings)


def test_python_static_preflight_rejects_invalid_errorbar_arguments():
    violations = _python_static_violations(
        "ax.errorbar(1, estimate, yerr=[lower, upper], linewidths=2)"
    )
    assert any("rejects linewidths" in item for item in violations)
    assert any("singleton asymmetric yerr" in item for item in violations)
    assert (
        _python_static_violations(
            "ax.errorbar([1, 2], values, yerr=[0.1, 0.2], linewidth=2)"
        )
        == []
    )
    assert (
        _python_static_violations(
            "ax.errorbar(1, estimate, yerr=[[lower], [upper]], elinewidth=2)"
        )
        == []
    )


def test_python_static_preflight_rejects_transposed_x_interval_by_dataflow():
    violations = _python_static_violations(
        "ax.errorbar([0], [md], xerr=[[md - ci_lo], [ci_hi - md]])"
    )
    assert any("effect interval is transposed" in item for item in violations)
    assert (
        _python_static_violations(
            """
y_pos = rng.uniform(-0.15, 0.15)
mean_diff = primary["point_estimate"]
effect_ax.errorbar(
    y_pos,
    mean_diff,
    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            "ax.errorbar([md], [0], xerr=[[md - ci_lo], [ci_hi - md]])"
        )
        == []
    )
    violations = _python_static_violations(
        """
y_pos = 0
mean_diff = primary["point_estimate"]
ci_low = primary["ci_lower_95"]
ci_high = primary["ci_upper_95"]
effect_ax.errorbar(
    y_pos,
    mean_diff,
    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
    )
    assert any("effect interval is transposed" in item for item in violations)
    violations = _python_static_violations(
        """
y_pos = 0
mean_diff = primary["mean_difference"]
ci_low = primary["ci_95_lower"]
ci_high = primary["ci_95_upper"]
effect_ax.errorbar(
    x=[y_pos],
    y=[mean_diff],
    xerr=[[mean_diff - ci_low], [ci_high - mean_diff]],
)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
    )
    assert any("effect interval is transposed" in item for item in violations)


def test_python_static_preflight_rejects_y_interval_on_effect_x_axis():
    violations = _python_static_violations(
        """
effect_ax.errorbar([0.0], [mean_difference], yerr=np.array([[lower], [upper]]))
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
    )

    assert any("effect interval is transposed" in item for item in violations)
    violations = _python_static_violations(
        """
x_pos = 0
effect_ax.errorbar(
    x_pos,
    mean_difference,
    yerr=np.array([[lower], [upper]]),
)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
    )
    assert any("effect interval is transposed" in item for item in violations)
    assert (
        _python_static_violations(
            """
raw_ax.errorbar([0.0], [group_mean], yerr=np.array([[se], [se]]))
raw_ax.set_xlabel("Group")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
effect_ax.errorbar([mean_difference], [0.0], xerr=np.array([[lower], [upper]]))
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )


def test_python_static_preflight_rejects_clipped_effect_null_reference():
    violations = _python_static_violations(
        """
effect_ax.axvline(x=0)
effect_ax.set_xlim(ci_lower - 1, ci_upper + 1)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
    )

    assert any("clip the intended zero/null reference" in item for item in violations)
    assert (
        _python_static_violations(
            """
effect_ax.axvline(x=0)
effect_ax.set_xlim(min(0, ci_lower - 1), max(0, ci_upper + 1))
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
effect_ax.axvline(x=0)
effect_ax.set_xlim(-1.5, 7.5)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
ZERO_INCLUSIVE_LOW = -2.0
ZERO_INCLUSIVE_HIGH = 8.0
effect_ax.axvline(x=0)
effect_ax.set_xlim(ZERO_INCLUSIVE_LOW, ZERO_INCLUSIVE_HIGH)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
effect_ax.set_xlim(min(mean_difference - 3, -1), max(mean_difference + 3, 1))
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
xlim_left = min(0, ci_lower - margin)
xlim_right = max(0, ci_upper + margin)
effect_ax.set_xlim(xlim_left, xlim_right)
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
xlim_left = ci_lower - margin
xlim_right = ci_upper + margin
xlim_left = min(xlim_left, 0.0)
xlim_right = max(xlim_right, 0.0)
effect_ax.set_xlim(xlim_left, xlim_right)
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
effect_span = max(abs(ci_lower), abs(ci_upper), abs(mean_difference), 0.5)
xlim_left = -effect_span * 1.3
xlim_right = effect_span * 1.3
effect_ax.set_xlim(xlim_left, xlim_right)
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
critical_values = [0.0, ci_lower, ci_upper, mean_difference]
lo = min(critical_values)
hi = max(critical_values)
span = hi - lo
padding = span * 0.2
effect_ax.set_xlim(lo - padding, hi + padding)
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
        == []
    )
    assert any(
        "clip the intended zero/null reference" in item
        for item in _python_static_violations(
            """
xlim_left = min(ci_lower - margin, 0.0)
xlim_right = max(ci_upper + margin, 0.0)
xlim_left = ci_lower
effect_ax.set_xlim(xlim_left, xlim_right)
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
    )
    assert any(
        "clip the intended zero/null reference" in item
        for item in _python_static_violations(
            """
effect_ax.set_xlim(ci_lower, max(0, ci_upper + 1))
effect_ax.axvline(x=0)
effect_ax.set_xlabel("Mean Difference (Treatment - Control)")
"""
        )
    )


def test_python_static_preflight_rejects_duplicate_group_jitter_centers():
    violations = _python_static_violations(
        """
jitter_control = rng.uniform(-0.15, 0.15, len(control))
jitter_treatment = rng.uniform(-0.15, 0.15, len(treatment))
ax.scatter(jitter_control, control, label="Control")
ax.scatter(jitter_treatment, treatment, label="Treatment")
ax.set_xticks([0, 1])
"""
    )

    assert any("share a jitter center" in item for item in violations)
    assert (
        _python_static_violations(
            """
jitter_control = rng.uniform(-0.15, 0.15, len(control))
jitter_treatment = rng.uniform(-0.15, 0.15, len(treatment))
ax.scatter(0 + jitter_control, control, label="Control")
ax.scatter(1 + jitter_treatment, treatment, label="Treatment")
ax.set_xticks([0, 1])
"""
        )
        == []
    )


def test_python_static_preflight_rejects_empty_effect_axis_ticks():
    violations = _python_static_violations(
        """
effect_ax.errorbar(mean_difference, 0, xerr=ci)
effect_ax.set_xlabel("Mean difference (treatment - control)")
effect_ax.set_xticks([])
"""
    )

    assert any("require visible numeric x ticks" in item for item in violations)
    assert (
        _python_static_violations(
            """
effect_ax.errorbar(mean_difference, 0, xerr=ci)
effect_ax.set_xlabel("Mean difference (treatment - control)")
effect_ax.set_xticks([0, 2, 4, 6])
"""
        )
        == []
    )


def test_python_static_preflight_rejects_empty_legend():
    violations = _python_static_violations(
        """
effect_ax.plot(mean_difference, 0, 'o')
effect_ax.legend(loc='lower right')
"""
    )

    assert any("legend() has no labeled artists" in item for item in violations)
    assert (
        _python_static_violations(
            """
effect_ax.plot(mean_difference, 0, 'o', label='Mean difference')
effect_ax.legend(loc='lower right')
"""
        )
        == []
    )
    assert (
        _python_static_violations(
            """
effect_ax.errorbar(mean_difference, 0, xerr=ci, label=f"{group} (n={n})")
effect_ax.legend(loc='lower right')
"""
        )
        == []
    )
    assert any(
        "legend() has no labeled artists" in item
        for item in _python_static_violations(
            """
effect_ax.plot(mean_difference, 0, 'o', label='_nolegend_')
effect_ax.legend(loc='lower right')
"""
        )
    )


def test_python_static_preflight_rejects_get_segments_on_errorbar_caplines():
    violations = _python_static_violations(
        """
container = effect_ax.errorbar([estimate], [0], xerr=[[lower], [upper]])
caps = container[1]
segments = caps.get_segments()
"""
    )

    assert any("caplines are a tuple" in item for item in violations)
    assert (
        _python_static_violations(
            """
container = effect_ax.errorbar([estimate], [0], xerr=[[lower], [upper]])
segments = container[2][0].get_segments()
"""
        )
        == []
    )


def test_python_static_preflight_rejects_line_methods_on_errorbar_container():
    violations = _python_static_violations(
        """
container = effect_ax.errorbar([estimate], [0], xerr=[[lower], [upper]])
estimate_x = container.get_xdata()[0]
"""
    )

    assert any("does not expose line coordinate methods" in item for item in violations)
    assert (
        _python_static_violations(
            """
container = effect_ax.errorbar([estimate], [0], xerr=[[lower], [upper]])
estimate_x = container[0].get_xdata()[0]
"""
        )
        == []
    )


def test_python_static_preflight_rejects_secondary_scientific_axes():
    for code in (
        "effect_ax = raw_ax.twinx()",
        "effect_ax = raw_ax.twiny()",
        "effect_ax = raw_ax.secondary_xaxis('top')",
        "effect_ax = raw_ax.secondary_yaxis('right')",
    ):
        violations = _python_static_violations(code)
        assert len(violations) == 1
        assert "separate labeled panel" in violations[0]

    assert _python_static_violations("fig, axes = plt.subplots(1, 2)") == []


def test_python_static_preflight_rejects_duplicate_category_ticks():
    violations = _python_static_violations("ax.set_xticks([0, 0])")
    assert any("unique tick positions" in item for item in violations)
    assert _python_static_violations("ax.set_xticks([0, 1])") == []


def test_output_inspection_rejects_nonrectangular_reader_table(tmp_path):
    executor = _executor(tmp_path)
    output = tmp_path / "output"
    tables = output / "tables"
    tables.mkdir(parents=True)
    table = tables / "summary.csv"
    table.write_text(
        "metric,control,treatment,contrast\nt_statistic,,,,10.897\n",
        encoding="utf-8",
    )

    _, violations = executor._inspect_outputs(output)

    assert any("table rows must be rectangular" in item for item in violations)

    table.write_text(
        "metric,control,treatment,contrast\nt_statistic,,,10.897\n",
        encoding="utf-8",
    )
    _, violations = executor._inspect_outputs(output)
    assert violations == []


def test_output_inspection_rejects_nonfinite_or_malformed_json(tmp_path):
    executor = _executor(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    result = output / "results.json"

    result.write_text('{"median": Infinity}\n', encoding="utf-8")
    _, violations = executor._inspect_outputs(output)
    assert any("strict JSON" in item and "Infinity" in item for item in violations)

    result.write_text('{"median": null}\n', encoding="utf-8")
    _, violations = executor._inspect_outputs(output)
    assert violations == []

    notebook = output / "analysis.ipynb"
    notebook.write_text('{"cells": [], "score": NaN}\n', encoding="utf-8")
    _, violations = executor._inspect_outputs(output)
    assert any("strict JSON" in item and "NaN" in item for item in violations)
    notebook.unlink()

    jsonl = output / "rows.jsonl"
    jsonl.write_text('{"estimate": 1}\n{"estimate": Infinity}\n', encoding="utf-8")
    _, violations = executor._inspect_outputs(output)
    assert any("strict JSON Lines" in item and "line 2" in item for item in violations)


def test_prior_reference_preflight_requires_current_successful_execution():
    code = "open('/prior/exec-002/output/results.json').read()"

    denied = _unavailable_prior_reference_violations(code, {"exec-001"})
    allowed = _unavailable_prior_reference_violations(code, {"exec-002"})

    assert len(denied) == 1
    assert "/history/attempt-N/exec-ID/output" in denied[0]
    assert allowed == []


def test_environment_snapshot_resolves_immutable_generation_and_copies_lock(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    environments = tmp_path / "environments" / str(__import__("uuid").uuid4())
    generation = environments / ".generations" / "python-one"
    (generation / "packages").mkdir(parents=True)
    (generation / "lock.json").write_text('{"installed":[]}\n', encoding="utf-8")
    (environments / "python").symlink_to(".generations/python-one")
    executor = AnalysisExecutor(
        workspace,
        tmp_path / "computations",
        SandboxSettings(),
        environment_dir=environments,
    )
    call_dir = executor.root / "snapshot"
    call_dir.mkdir()
    packages, locks, artifacts = executor._snapshot_environment("python", call_dir)
    assert packages == (generation / "packages").resolve()
    assert set(locks) == {"python"}
    assert artifacts[0].path.endswith("environment-python-lock.json")


def test_remote_preflight_uses_managed_worker_paths(tmp_path, monkeypatch):
    calls = []
    released = []

    def execute(self, language, code, timeout_seconds=120):
        del code, timeout_seconds
        calls.append((language, self.workspace, self.root))
        return {"status": "succeeded"}

    monkeypatch.setenv("SCIENTIFIC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(RemoteAnalysisExecutor, "execute", execute)
    monkeypatch.setattr(
        RemoteAnalysisExecutor,
        "close",
        lambda self: released.append((self.workspace, self.root)),
    )
    settings = replace(
        SandboxSettings(),
        worker_url="http://sandbox:8090",
        worker_token="x" * 32,
        bwrap=tmp_path / "missing-bwrap",
        python=tmp_path / "missing-python",
        rscript=tmp_path / "missing-rscript",
    )
    result = sandbox_preflight(settings)

    assert result["missing_required"] == []
    assert result["probes"] == {"python": "succeeded", "r": "succeeded"}
    assert [item[0] for item in calls] == ["python", "r"]
    assert all(item[1].parts[-1] == "files" for item in calls)
    assert all("runs" in item[2].parts for item in calls)
    assert len(released) == 1
    assert not any((tmp_path / "workspaces").iterdir())


def test_remote_executor_releases_matching_worker_state(tmp_path, monkeypatch):
    workspace = tmp_path / "workspaces" / "workspace-id" / "files"
    root = tmp_path / "workspaces" / "workspace-id" / "runs" / "run-1" / "computations"
    workspace.mkdir(parents=True)
    root.mkdir(parents=True)
    calls = []

    class Response:
        is_error = False

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("scientific_agent.execution.httpx.post", post)
    executor = RemoteAnalysisExecutor(
        workspace,
        root,
        replace(
            SandboxSettings(),
            worker_url="http://sandbox:8090",
            worker_token="x" * 32,
        ),
    )
    executor.close()

    assert calls[0][0] == "http://sandbox:8090/release"
    assert calls[0][1]["json"] == {
        "workspace": str(workspace),
        "computation_root": str(root),
    }


def test_remote_executor_sends_controller_attempt_budget(tmp_path, monkeypatch):
    workspace = tmp_path / "workspaces" / "workspace-id" / "files"
    root = tmp_path / "workspaces" / "workspace-id" / "runs" / "run-1" / "computations"
    workspace.mkdir(parents=True)
    root.mkdir(parents=True)
    calls = []

    class Response:
        is_error = False

        @staticmethod
        def json():
            return {
                "result": {"status": "succeeded"},
                "evidence": {"successful_calls": 0, "records": [], "artifacts": []},
            }

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Response()

    monkeypatch.setattr("scientific_agent.execution.httpx.post", post)
    executor = RemoteAnalysisExecutor(
        workspace,
        root,
        replace(
            SandboxSettings(),
            worker_url="http://sandbox:8090",
            worker_token="x" * 32,
            max_calls_per_attempt=8,
        ),
    )

    executor.execute("python", "print(1)", 10)

    assert calls[0][1]["json"]["max_calls_per_attempt"] == 8


def test_preflight_probes_advertised_python_and_r_packages(tmp_path, monkeypatch):
    seen: list[tuple[str, str]] = []

    class FakeExecutor:
        def __init__(self, *_args, **_kwargs):
            pass

        def execute(self, language, code, **_kwargs):
            seen.append((language, code))
            return {"status": "succeeded"}

        def close(self):
            pass

    monkeypatch.setattr("scientific_agent.execution.AnalysisExecutor", FakeExecutor)
    paths = {}
    for name in ("bwrap", "prlimit", "python", "rscript"):
        path = tmp_path / name
        path.touch()
        paths[name] = path
    for name in ("python_prefix", "python_packages", "r_library"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    settings = replace(SandboxSettings(), **paths)
    result = sandbox_preflight(settings, tmp_path)
    assert result["probes"] == {"python": "succeeded", "r": "succeeded"}
    assert "import matplotlib,numpy,pandas,scipy,sklearn,statsmodels" in seen[0][1]
    assert "ggplot2" in seen[1][1] and "data.table" in seen[1][1]


@pytest.mark.live
def test_python_and_r_execute_with_recorded_outputs_and_confinement(
    tmp_path, monkeypatch
):
    executor = _executor(tmp_path)
    (executor.workspace / "values.csv").write_text(
        "group,value\nA,1\nA,3\nB,5\nB,7\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SCIENTIFIC_AGENT_TEST_SECRET", "must-not-enter-sandbox")

    python_result = executor.execute(
        "python",
        """
import json
import os
import pathlib
import socket
import pandas as pd
from scipy import stats

data = pd.read_csv('/workspace/values.csv')
means = data.groupby('group', as_index=False)['value'].mean()
means.to_csv('/output/python_means.csv', index=False)
try:
    pathlib.Path('/workspace/blocked.txt').write_text('blocked')
    workspace_read_only = False
except OSError:
    workspace_read_only = True
network_errno = socket.socket().connect_ex(('1.1.1.1', 53))
summary = {
    'workspace_read_only': workspace_read_only,
    'network_blocked': network_errno != 0,
    'passwd_hidden': not pathlib.Path('/etc/passwd').exists(),
    'secret_scrubbed': 'SCIENTIFIC_AGENT_TEST_SECRET' not in os.environ,
    'welch_pvalue': float(stats.ttest_ind(
        data.loc[data.group == 'A', 'value'],
        data.loc[data.group == 'B', 'value'],
        equal_var=False,
    ).pvalue),
}
pathlib.Path('/output/python_summary.json').write_text(json.dumps(summary))
""",
        timeout_seconds=30,
    )
    assert python_result["status"] == "succeeded", python_result["stderr"]
    assert any(
        path.endswith("python_summary.json")
        for path in python_result["output_previews"]
    )
    summary_path = next(
        Path(item["path"])
        for item in python_result["artifacts"]
        if item["path"].endswith("python_summary.json")
    )
    summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
    assert all(
        summary[name]
        for name in (
            "workspace_read_only",
            "network_blocked",
            "passwd_hidden",
            "secret_scrubbed",
        )
    )

    r_result = executor.execute(
        "r",
        """
data <- read.csv('/workspace/values.csv')
means <- aggregate(value ~ group, data=data, FUN=mean)
python_means <- read.csv('/prior/exec-001/output/python_means.csv')
stopifnot(all.equal(means$value, python_means$value))
write.csv(means, '/output/r_means.csv', row.names=FALSE)
""",
        timeout_seconds=30,
    )
    assert r_result["status"] == "succeeded", r_result["stderr"]
    evidence = executor.evidence()
    assert evidence.successful_calls == 2
    assert {Path(item.path).name for item in evidence.artifacts} == {
        "python_means.csv",
        "python_summary.json",
        "r_means.csv",
    }
    assert not (executor.workspace / "blocked.txt").exists()


@pytest.mark.live
def test_sandbox_rejects_timeout_symlink_and_excess_calls(tmp_path):
    timed = _executor(tmp_path / "timed", max_wall_seconds=1)
    timeout_result = timed.execute("python", "import time; time.sleep(30)", 1)
    assert timeout_result["status"] == "timed_out"

    linked = _executor(tmp_path / "linked")
    link_result = linked.execute(
        "python",
        "import os; os.symlink('/etc/passwd', '/output/leak')",
        10,
    )
    assert link_result["status"] == "policy_denied"
    assert any("non-regular output" in item for item in link_result["violations"])
    assert linked.evidence().successful_calls == 0

    budgeted = _executor(tmp_path / "budgeted", max_calls_per_attempt=1)
    assert (
        budgeted.execute("python", "open('/output/ok.txt', 'w').write('ok')", 10)[
            "status"
        ]
        == "succeeded"
    )
    denied = budgeted.execute("python", "print('second')", 10)
    assert denied["status"] == "policy_denied"
    assert "analysis call budget exhausted" in denied["violations"]
    assert denied["calls_used"] == 2
    assert denied["calls_remaining"] == 0
    assert denied["stop_required"] is True


@pytest.mark.live
def test_repair_attempt_can_read_prior_attempt_history(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    history = tmp_path / "computations"
    first = AnalysisExecutor(workspace, history / "attempt-0", SandboxSettings())
    created = first.execute(
        "python",
        "open('/output/reference.json', 'w').write('{\"estimate\": 5}')",
        10,
    )
    assert created["status"] == "succeeded"

    repaired = AnalysisExecutor(workspace, history / "attempt-1", SandboxSettings())
    compared = repaired.execute(
        "python",
        """
import json
value = json.load(open('/history/attempt-0/exec-001/output/reference.json'))
open('/output/checked.txt', 'w').write(str(value['estimate']))
""",
        10,
    )
    assert compared["status"] == "succeeded", compared["stderr"]
    assert Path(compared["artifacts"][-1]["path"]).read_text() == "5"


@pytest.mark.live
def test_failed_partial_outputs_are_auditable_but_not_reusable(tmp_path):
    executor = _executor(tmp_path)
    failed = executor.execute(
        "python",
        "open('/output/partial.json', 'w').write('{}'); raise RuntimeError('stop')",
        10,
    )
    assert failed["status"] == "failed"
    assert failed["output_previews"] == {}
    rejected = [
        item
        for item in failed["artifacts"]
        if item["description"] == "rejected sandbox output (not evidence)"
    ]
    assert len(rejected) == 1
    assert "rejected_output" in rejected[0]["path"]
    assert not (executor.root / "exec-001" / "output" / "partial.json").exists()

    checked = executor.execute(
        "python",
        """
from pathlib import Path
visible = Path('/prior/exec-001/output/partial.json').exists()
Path('/output/visibility.txt').write_text(str(visible))
""",
        10,
    )
    assert checked["status"] == "succeeded"
    assert Path(checked["artifacts"][-1]["path"]).read_text() == "False"
