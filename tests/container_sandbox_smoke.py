"""Manual Compose smoke probe for the internal Python/R sandbox worker."""

from pathlib import Path

from scientific_agent.config import SandboxSettings
from scientific_agent.execution import RemoteAnalysisExecutor


workspace = Path("/data/workspaces/00000000-0000-4000-8000-000000000001/files")
root = Path(
    "/data/workspaces/00000000-0000-4000-8000-000000000001/"
    "runs/run-1/computations/attempt-0"
)
workspace.mkdir(parents=True, exist_ok=True)
root.mkdir(parents=True, exist_ok=True)
executor = RemoteAnalysisExecutor(workspace, root, SandboxSettings())
try:
    python_result = executor.execute(
        "python",
        """
import pandas as pd
pd.DataFrame({'x': [1, 2, 3]}).describe().to_csv('/output/python.csv')
""",
        30,
    )
except Exception as exc:
    print(f"python worker call failed: {exc}")
    response = getattr(exc, "response", None)
    if response is not None:
        print(response.text)
    raise
r_result = executor.execute(
    "r",
    "d <- read.csv('/prior/exec-001/output/python.csv'); "
    "write.csv(d, '/output/r.csv', row.names=FALSE)",
    30,
)
evidence = executor.evidence()
assert python_result["status"] == "succeeded", python_result["stderr"]
assert r_result["status"] == "succeeded", r_result["stderr"]
assert evidence.successful_calls == 2
assert len(evidence.artifacts) == 2
print("python=succeeded r=succeeded calls=2 artifacts=2")
