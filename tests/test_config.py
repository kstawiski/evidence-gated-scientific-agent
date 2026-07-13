import os

import pytest

from scientific_agent.config import load_mcp_secrets


def test_secret_loader_parses_only_allow_list(tmp_path, monkeypatch):
    for name in ("CONTEXT7_API_KEY", "BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
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


def test_environment_secret_overrides_file(tmp_path, monkeypatch):
    path = tmp_path / "mcp.env"
    path.write_text("CONTEXT7_API_KEY=file-key\n", encoding="utf-8")
    path.chmod(0o600)
    monkeypatch.setenv("CONTEXT7_API_KEY", "environment-key")
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    assert load_mcp_secrets(path)["CONTEXT7_API_KEY"] == "environment-key"


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
