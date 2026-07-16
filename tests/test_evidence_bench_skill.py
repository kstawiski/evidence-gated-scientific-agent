import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


SKILL_ROOT = Path("skills/evidence-bench")
SCRIPT = SKILL_ROOT / "scripts/evidence_bench.py"


def _client_module():
    spec = importlib.util.spec_from_file_location("evidence_bench_skill_client", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skill_metadata_and_agent_interface_are_installable():
    content = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
    _, frontmatter, _ = content.split("---", 2)
    metadata = yaml.safe_load(frontmatter)
    interface = yaml.safe_load(
        (SKILL_ROOT / "agents/openai.yaml").read_text(encoding="utf-8")
    )["interface"]

    assert metadata["name"] == "evidence-bench"
    assert "10.20.102.122" in metadata["description"]
    assert "$evidence-bench" in interface["default_prompt"]
    assert SCRIPT.is_file()


def test_skill_client_defaults_to_the_lab_service_and_all_mcps():
    module = _client_module()
    parser = module.build_parser()
    args = parser.parse_args(["run", "--objective", "test objective"])

    assert args.base_url == "http://10.20.102.122"
    assert module.DEFAULT_MCPS == (
        "context7",
        "brave-search",
        "chrome-devtools",
    )
    assert args.no_code is False


def test_skill_client_rejects_credentials_in_service_url():
    module = _client_module()

    with pytest.raises(module.ClientError, match="credentials"):
        module.Client("http://user:password@10.20.102.122")


def test_skill_client_rejects_symlink_upload(tmp_path):
    module = _client_module()
    source = tmp_path / "source.txt"
    source.write_text("bounded input", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(source)

    with pytest.raises(module.ClientError, match="non-symlink"):
        module.Client("http://127.0.0.1:9").upload("workspace", link)


def test_skill_status_command_uses_machine_readable_stdout(monkeypatch, capsys):
    module = _client_module()

    monkeypatch.setattr(
        module.Client,
        "json_request",
        lambda self, method, path, payload=None: {
            "id": "run-id",
            "status": "supported",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "status", "--run-id", "run-id"],
    )

    assert module.main() == 0
    assert '"status": "supported"' in capsys.readouterr().out
