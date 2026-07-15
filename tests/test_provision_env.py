import os
import stat
import subprocess
import sys
from pathlib import Path


def test_provision_without_web_auth_omits_web_password(tmp_path):
    output = tmp_path / ".env"
    script = Path(__file__).parents[1] / "deploy" / "provision_env.py"
    environment = {
        **os.environ,
        "WEB_AUTH_ENABLED": "false",
        "SCIENTIFIC_AGENT_PUBLIC_URL": "http://evidence-bench.internal:8070",
        "QWEN_BASE_URL": "http://qwen.internal/v1",
        "GEMMA_BASE_URL": "http://gemma.internal/v1",
        "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_BYTES": "42949672960",
        "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_BYTES": "107374182400",
        "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_ENTRIES": "500000",
        "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_ENTRIES": "2500000",
    }
    environment.pop("WEB_PASSWORD", None)

    subprocess.run(
        [sys.executable, str(script), "--output", str(output)],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )

    values = output.read_text(encoding="utf-8")
    assert "WEB_AUTH_ENABLED=false\n" in values
    assert "WEB_PASSWORD=" not in values
    assert "A2A_TOKEN=" in values
    assert "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_BYTES=42949672960\n" in values
    assert "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_BYTES=107374182400\n" in values
    assert "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_ENTRIES=500000\n" in values
    assert "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_ENTRIES=2500000\n" in values
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
