import pytest

import scientific_agent.mcp as mcp_module
from scientific_agent.cli import _parser
from scientific_agent.config import DEFAULT_MCP_SERVERS, Settings, load_mcp_secrets
from scientific_agent.mcp import build_mcp_toolsets


def test_all_research_mcps_are_enabled_by_default():
    assert DEFAULT_MCP_SERVERS == (
        "context7",
        "brave-search",
        "chrome-devtools",
    )
    assert Settings().mcp_servers == DEFAULT_MCP_SERVERS


def test_explicit_empty_mcp_selection_does_not_load_defaults(monkeypatch):
    def unexpected_secret_load():
        raise AssertionError("an empty MCP selection must not load secrets")

    monkeypatch.setattr(mcp_module, "load_mcp_secrets", unexpected_secret_load)

    assert build_mcp_toolsets(Settings(), ()) == []


def test_cli_defaults_to_all_mcps_and_preserves_explicit_empty_opt_out():
    parser = _parser()

    assert parser.parse_args(["run", "task"]).mcp == ",".join(DEFAULT_MCP_SERVERS)
    assert parser.parse_args(["run", "--mcp", "", "task"]).mcp == ""


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


@pytest.mark.parametrize("value", ["-1", "9"])
def test_repair_round_budget_is_bounded(monkeypatch, value):
    monkeypatch.setenv("MAX_REPAIR_ROUNDS", value)

    with pytest.raises(ValueError, match="between 0 and 8"):
        Settings()


def test_zero_repair_rounds_is_an_explicit_fail_closed_budget(monkeypatch):
    monkeypatch.setenv("MAX_REPAIR_ROUNDS", "0")

    assert Settings().max_repair_rounds == 0


def test_zero_model_token_ceiling_preserves_proxy_maximum_thinking(monkeypatch):
    monkeypatch.setenv("QWEN_MAX_TOKENS", "0")
    monkeypatch.setenv("QWEN_ENABLE_THINKING", "inherit")
    monkeypatch.setenv("QWEN_NATIVE_JSON_SCHEMA", "false")

    endpoint = Settings().qwen

    assert endpoint.max_tokens is None
    assert endpoint.enable_thinking is None
    assert endpoint.native_json_schema is False
    assert endpoint.request_timeout_seconds == 21600
    assert endpoint.capacity_wait_seconds == 21600


@pytest.mark.parametrize("name", ["QWEN_MAX_TOKENS", "GEMMA_MAX_TOKENS"])
def test_negative_model_token_ceiling_is_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "-1")

    with pytest.raises(ValueError, match="zero or a positive integer"):
        Settings()


def test_invalid_model_boolean_is_rejected(monkeypatch):
    monkeypatch.setenv("GEMMA_ENABLE_THINKING", "sometimes")

    with pytest.raises(ValueError, match="true, false, or inherit"):
        Settings()


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("SCIENTIFIC_AGENT_MAX_RESEARCH_MODEL_TURNS", "0", "model_turns"),
        ("SCIENTIFIC_AGENT_MAX_RESEARCH_TOOL_CALLS", "257", "tool_calls"),
        ("SCIENTIFIC_AGENT_MAX_REPEATED_TOOL_RESULTS", "0", "repeated_tool_results"),
    ],
)
def test_research_budgets_are_bounded(monkeypatch, name, value, message):
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        Settings()
