import os

import pytest

from scientific_agent.config import load_mcp_secrets


def test_secret_loader_parses_only_allow_list(tmp_path):
    path = tmp_path / "mcp.env"
    path.write_text(
        "export CONTEXT7_API_KEY='context-key'\n"
        "BRAVE_SEARCH_API_KEY=brave-key\n"
        "UNRELATED_SECRET=must-not-load\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    values = load_mcp_secrets(path)
    assert values == {
        "CONTEXT7_API_KEY": "context-key",
        "BRAVE_SEARCH_API_KEY": "brave-key",
        "BRAVE_API_KEY": "brave-key",
    }


def test_secret_loader_rejects_permissive_mode(tmp_path):
    path = tmp_path / "mcp.env"
    path.write_text("CONTEXT7_API_KEY=x\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        load_mcp_secrets(path)


def test_secret_loader_rejects_symlink(tmp_path):
    target = tmp_path / "target"
    target.write_text("CONTEXT7_API_KEY=x\n", encoding="utf-8")
    target.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(PermissionError):
        load_mcp_secrets(link)
