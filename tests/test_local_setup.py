import os
import re
import stat
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).parents[1]
SETUP = ROOT / "scripts" / "local_setup.sh"
RUNNER = ROOT / "scripts" / "local_run.sh"


def recommendation(ram_gb: int, vram_gb: int, *extra: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "bash",
            str(SETUP),
            "--recommend-only",
            "--ram-gb",
            str(ram_gb),
            "--vram-gb",
            str(vram_gb),
            *extra,
        ],
        cwd=ROOT,
        env=os.environ,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(
        line.split("=", 1)
        for line in result.stdout.splitlines()
        if re.fullmatch(r"[A-Z_]+=[^\n]+", line)
    )


@pytest.mark.parametrize(
    ("ram_gb", "vram_gb", "profile", "qwen", "gemma"),
    [
        (16, 0, "compact", "qwen3:4b", "gemma3:4b"),
        (40, 0, "balanced", "qwen3:14b", "gemma4:12b"),
        (64, 0, "performance", "qwen3.6:27b", "gemma4:26b"),
        (64, 12, "performance", "qwen3.6:27b", "gemma4:26b"),
        (96, 0, "workstation", "qwen3.6:35b", "gemma4:31b"),
        (96, 12, "workstation", "qwen3.6:35b", "gemma4:31b"),
        (24, 12, "balanced", "qwen3:14b", "gemma4:12b"),
        (32, 24, "performance", "qwen3.6:27b", "gemma4:26b"),
        (48, 32, "workstation", "qwen3.6:35b", "gemma4:31b"),
    ],
)
def test_local_model_recommendations(
    ram_gb: int,
    vram_gb: int,
    profile: str,
    qwen: str,
    gemma: str,
):
    selected = recommendation(ram_gb, vram_gb)

    assert selected["PROFILE"] == profile
    assert selected["QWEN_SOURCE_MODEL"] == qwen
    assert selected["GEMMA_SOURCE_MODEL"] == gemma
    assert selected["CONTEXT_TOKENS"] == "32768"


def test_local_model_profile_and_exact_models_can_be_overridden():
    selected = recommendation(
        96,
        48,
        "--profile",
        "balanced",
        "--qwen-model",
        "qwen3:8b",
        "--gemma-model",
        "gemma3:12b",
        "--context",
        "16384",
    )

    assert selected["PROFILE"] == "balanced"
    assert selected["QWEN_SOURCE_MODEL"] == "qwen3:8b"
    assert selected["GEMMA_SOURCE_MODEL"] == "gemma3:12b"
    assert selected["CONTEXT_TOKENS"] == "16384"
    assert selected["MODEL_DOWNLOAD_GB"] == "custom"


@pytest.mark.parametrize(
    "arguments",
    [
        ("--profile", "oversized"),
        ("--context", "4096"),
        ("--ram-gb", "0"),
        ("--vram-gb", "-1"),
        ("--qwen-model", "bad model"),
        ("--gemma-model", "bad;model"),
    ],
)
def test_local_setup_rejects_invalid_selection_input(arguments: tuple[str, str]):
    result = subprocess.run(
        ["bash", str(SETUP), "--recommend-only", *arguments],
        cwd=ROOT,
        env=os.environ,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Error:" in result.stderr


def test_local_scripts_are_executable_and_runner_help_needs_no_docker():
    assert stat.S_IMODE(SETUP.stat().st_mode) & 0o111
    assert stat.S_IMODE(RUNNER.stat().st_mode) & 0o111

    result = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        cwd=ROOT,
        env=os.environ,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "start|stop|restart|status|logs|preflight|update" in result.stdout


def test_local_compose_overlay_uses_versioned_public_images():
    project_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    overlay_text = (ROOT / "compose.local.yaml").read_text(encoding="utf-8")
    overlay = yaml.safe_load(overlay_text)
    expected_services = {
        "evidence-bench",
        "browser",
        "browser-egress",
        "browser-cdp-gateway",
        "browser-ui-gateway",
        "sandbox-gateway",
        "environment-gateway",
        "sandbox-worker",
        "environment-worker",
    }

    assert set(overlay["services"]) == expected_services
    assert set(re.findall(r"\$\{EVIDENCE_BENCH_VERSION:-([^}]+)\}", overlay_text)) == {
        project_version
    }
    for service in overlay["services"].values():
        assert service["image"].startswith("ghcr.io/kstawiski/")
        assert service["pull_policy"] == "always"


def test_native_docker_can_override_the_ollama_host_gateway():
    compose = (ROOT / "compose.yaml").read_text(encoding="utf-8")

    assert "host.docker.internal:${OLLAMA_HOST_GATEWAY:-host-gateway}" in compose
    assert (
        'set_env_value OLLAMA_HOST_GATEWAY "$ollama_host_gateway"'
        in SETUP.read_text(encoding="utf-8")
    )


def test_local_tutorial_is_linked_from_product_surfaces():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    site = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
    tutorial = (ROOT / "docs" / "LOCAL_SETUP.md").read_text(encoding="utf-8")

    assert "docs/LOCAL_SETUP.md" in readme
    assert "docs/LOCAL_SETUP.md" in site
    assert "./scripts/local_setup.sh" in tutorial
    assert "./scripts/local_run.sh" in tutorial
    assert "macOS" in tutorial and "Linux" in tutorial and "WSL2" in tutorial
