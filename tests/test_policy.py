import json
from pathlib import Path

import scientific_agent.policy as policy_module
from scientific_agent.policy import ToolPolicy, default_allowed_tools
from scientific_agent.provenance import EventLedger


def policy(tmp_path: Path, *, preserve_evidence: bool = False) -> ToolPolicy:
    return ToolPolicy(
        EventLedger(tmp_path / "events.jsonl"),
        default_allowed_tools(True),
        evidence_dir=tmp_path / "evidence" if preserve_evidence else None,
    )


def test_disallowed_tool_is_denied(tmp_path):
    allowed, reason = policy(tmp_path).evaluate("run_bash", {"command": "id"})
    assert not allowed
    assert "allow-listed" in reason


def test_private_and_non_http_urls_are_denied(tmp_path):
    gate = policy(tmp_path)
    assert not gate.evaluate("new_page", {"url": "http://127.0.0.1:8000"})[0]
    assert not gate.evaluate("new_page", {"url": "file:///etc/passwd"})[0]
    assert not gate.evaluate("new_page", {"url": "https://100.64.0.1/"})[0]


def test_public_literal_ip_is_allowed(tmp_path):
    assert policy(tmp_path).evaluate("new_page", {"url": "https://1.1.1.1/"})[0]


def test_chrome_tools_are_opt_in(tmp_path):
    gate = ToolPolicy(
        EventLedger(tmp_path / "events.jsonl"), default_allowed_tools(False)
    )
    assert not gate.evaluate("new_page", {"url": "https://1.1.1.1/"})[0]


def test_chrome_callback_forces_isolated_context_and_orders_page_tools(tmp_path):
    gate = policy(tmp_path)

    class Tool:
        name = "take_snapshot"

    denied = gate.before_tool(tool=Tool(), args={}, tool_context=None)
    assert denied["error"] == "POLICY_DENIED"

    Tool.name = "new_page"
    arguments = {"url": "https://1.1.1.1/"}
    assert gate.before_tool(tool=Tool(), args=arguments, tool_context=None) is None
    assert arguments["isolatedContext"].startswith("scientific-agent-")


def test_retrieval_callback_records_tools_and_urls_without_response_body(tmp_path):
    gate = policy(tmp_path, preserve_evidence=True)

    class Tool:
        name = "query-docs"

    gate.after_tool(
        tool=Tool(),
        args={"query": "graphs"},
        tool_context=None,
        tool_response={"content": "Documentation: https://adk.dev/graphs/."},
    )
    evidence = gate.retrieval_evidence()
    assert evidence.successful_calls == 1
    assert evidence.tools == ["query-docs"]
    assert evidence.urls == ["https://adk.dev/graphs/"]
    assert evidence.retrieval_dates
    assert len(evidence.artifacts) == 1
    assert Path(evidence.artifacts[0]).is_file()


def test_analysis_tools_are_opt_in_and_code_is_hashed_in_ledger(tmp_path):
    default_gate = policy(tmp_path)
    assert not default_gate.evaluate("run_python_analysis", {"code": "print(1)"})[0]

    ledger_path = tmp_path / "code-events.jsonl"
    gate = ToolPolicy(
        EventLedger(ledger_path),
        default_allowed_tools(include_chrome=False, enable_code=True),
    )

    class Tool:
        name = "run_python_analysis"

    code = "print('sensitive-input-name')"
    assert gate.before_tool(Tool(), {"code": code}, None) is None
    event = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    assert event["decision"] == "allow"
    assert event["arguments"]["code"]["bytes"] == len(code)
    assert code not in ledger_path.read_text(encoding="utf-8")


def test_package_installation_tools_are_separately_opt_in(tmp_path):
    default_gate = policy(tmp_path)
    assert not default_gate.evaluate(
        "install_python_packages", {"packages": ["polars"]}
    )[0]
    gate = ToolPolicy(
        EventLedger(tmp_path / "package-events.jsonl"),
        default_allowed_tools(
            include_chrome=False,
            enable_code=True,
            enable_packages=True,
        ),
    )
    allowed, reason = gate.evaluate(
        "install_r_packages",
        {"packages": ["BiocGenerics"], "repository": "bioconductor"},
    )
    assert allowed
    assert "canonical package registry" in reason


def test_cancelled_tool_result_is_observed_as_failed(tmp_path):
    observed = []
    gate = ToolPolicy(
        EventLedger(tmp_path / "events.jsonl"),
        default_allowed_tools(
            include_chrome=False,
            enable_code=True,
            enable_packages=True,
        ),
        observer=lambda *event: observed.append(event),
    )

    class Tool:
        name = "install_python_packages"

    gate.after_tool(
        Tool(),
        {"packages": ["polars"]},
        None,
        {"status": "cancelled", "installed": []},
    )

    assert observed == [("tool_result", "install_python_packages", "failed")]


def test_oversized_snapshot_is_preserved_and_compacted_for_model(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_module, "MAX_MODEL_TOOL_RESULT_BYTES", 4_096)
    monkeypatch.setattr(policy_module, "MAX_MODEL_TOOL_RESULTS_TOTAL_BYTES", 8_192)
    gate = policy(tmp_path, preserve_evidence=True)

    class Tool:
        name = "take_snapshot"

    original = {"content": "scientific page text " * 1_000}
    observed = gate.after_tool(Tool(), {}, None, original)

    assert observed["result_compacted"] is True
    assert "scientific page text" in observed["result_preview"]
    artifact = Path(observed["full_result_artifact"])
    assert json.loads(artifact.read_text(encoding="utf-8")) == original
    assert len(json.dumps(observed).encode("utf-8")) <= 4_096


def test_screenshot_binary_is_never_injected_as_text_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_module, "MAX_MODEL_TOOL_RESULT_BYTES", 1_024)
    gate = policy(tmp_path, preserve_evidence=True)

    class Tool:
        name = "take_screenshot"

    observed = gate.after_tool(Tool(), {}, None, {"image_base64": "aGVsbG8=" * 500})

    assert observed["result_compacted"] is True
    assert "result_preview" not in observed
    assert Path(observed["full_result_artifact"]).is_file()


def test_tiny_screenshot_is_also_preserved_and_never_injected(tmp_path):
    gate = policy(tmp_path, preserve_evidence=True)

    class Tool:
        name = "take_screenshot"

    observed = gate.after_tool(Tool(), {}, None, {"image_base64": "aGVsbG8="})

    assert observed["result_compacted"] is True
    assert "result_preview" not in observed
    assert Path(observed["full_result_artifact"]).is_file()


def test_tool_ledger_redacts_secret_keys_and_url_credentials(tmp_path):
    ledger_path = tmp_path / "secret-events.jsonl"
    gate = ToolPolicy(
        EventLedger(ledger_path),
        default_allowed_tools(include_chrome=True),
    )

    class Tool:
        name = "new_page"

    arguments = {
        "url": "https://user:pass@1.1.1.1/article?token=hidden#fragment",
        "metadata": {"api_key": "hidden-key", "label": "safe"},
    }
    assert gate.before_tool(Tool(), arguments, None) is None
    logged = ledger_path.read_text(encoding="utf-8")

    assert "user:pass" not in logged
    assert "?token" not in logged
    assert "#fragment" not in logged
    assert "hidden-key" not in logged
    event = json.loads(logged.splitlines()[0])
    assert event["arguments"]["url"] == "https://1.1.1.1/article"
    assert event["arguments"]["metadata"]["api_key"] == "[REDACTED]"


def test_cumulative_model_observation_budget_compacts_later_results(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(policy_module, "MAX_MODEL_TOOL_RESULT_BYTES", 4_096)
    monkeypatch.setattr(policy_module, "MAX_MODEL_TOOL_RESULTS_TOTAL_BYTES", 1_000)
    gate = policy(tmp_path, preserve_evidence=True)

    class Tool:
        name = "take_snapshot"

    first = gate.after_tool(Tool(), {}, None, {"content": "x" * 700})
    second = gate.after_tool(Tool(), {}, None, {"content": "y" * 700})

    assert first is None
    assert second["result_compacted"] is True
    assert "result_preview" not in second
    assert Path(second["full_result_artifact"]).is_file()
