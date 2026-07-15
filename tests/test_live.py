import pytest
from google.adk import Agent

from scientific_agent.config import Settings
from scientific_agent.mcp import build_mcp_toolsets, close_mcp_toolsets
from scientific_agent.models import qwen_model
from scientific_agent.policy import ToolPolicy, default_allowed_tools
from scientific_agent.preflight import run_preflight
from scientific_agent.provenance import EventLedger
from scientific_agent.runtime import run_text, run_typed
from scientific_agent.schemas import PlanningResult
from scientific_agent.workflow import build_planning_workflow


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_model_and_mcp_preflight():
    result = await run_preflight(
        Settings(),
        mcp_names=("context7", "brave-search", "chrome-devtools"),
    )
    assert not any(item["missing_required"] for item in result["mcp"].values())


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_planning_graph():
    result = await run_typed(
        build_planning_workflow(Settings()),
        "Produce a short evidence-backed plan for checking whether a CSV has duplicate sample IDs.",
        PlanningResult,
    )
    assert result.master_plan.plan.steps
    assert result.audit.verdict in {
        "pass",
        "pass_with_nonblocking_comments",
        "fail",
        "inconclusive",
    }


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_adk_context7_tool_is_called(tmp_path):
    settings = Settings()
    toolsets = build_mcp_toolsets(settings, ("context7",))
    policy = ToolPolicy(
        EventLedger(tmp_path / "tool-events.jsonl"),
        default_allowed_tools(include_chrome=False),
        evidence_dir=tmp_path / "evidence",
    )
    agent = Agent(
        name="context7_tool_probe",
        model=qwen_model(settings, temperature=0.2, timeout=180),
        instruction=(
            "You must call Context7 to retrieve Google ADK graph workflow "
            "documentation before answering. Return one concise summary."
        ),
        tools=toolsets,
        before_tool_callback=policy.before_tool,
        after_tool_callback=policy.after_tool,
        mode="chat",
        include_contents="none",
    )
    try:
        summary = await run_text(agent, "Find one ADK graph workflow fact.")
    finally:
        await close_mcp_toolsets(toolsets)
    assert summary
    evidence = policy.retrieval_evidence()
    assert evidence.successful_calls >= 1
    assert evidence.artifacts
