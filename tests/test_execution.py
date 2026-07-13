from dataclasses import replace
from pathlib import Path

import pytest

from scientific_agent.config import SandboxSettings
from scientific_agent.execution import AnalysisExecutor


def _executor(tmp_path: Path, **overrides) -> AnalysisExecutor:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    settings = replace(SandboxSettings(), **overrides)
    return AnalysisExecutor(workspace, tmp_path / "computations", settings)


@pytest.mark.live
def test_python_and_r_execute_with_recorded_outputs_and_confinement(tmp_path, monkeypatch):
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
    assert budgeted.execute(
        "python", "open('/output/ok.txt', 'w').write('ok')", 10
    )["status"] == "succeeded"
    denied = budgeted.execute("python", "print('second')", 10)
    assert denied["status"] == "policy_denied"
    assert "analysis call budget exhausted" in denied["violations"]
