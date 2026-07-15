"""Pinned, least-privilege MCP toolsets for ADK agents."""

from __future__ import annotations

import os

from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

from .config import PROJECT_ROOT, Settings, load_mcp_secrets


MCP_TOOL_FILTERS = {
    "context7": ["resolve-library-id", "query-docs"],
    "brave-search": ["brave_web_search", "brave_llm_context"],
    "chrome-devtools": [
        "new_page",
        "navigate_page",
        "take_snapshot",
        "take_screenshot",
        "wait_for",
    ],
}


def _bin(name: str) -> str:
    path = PROJECT_ROOT / "node_modules" / ".bin" / name
    if not path.is_file():
        raise FileNotFoundError(
            f"pinned MCP executable is missing: {path}; run npm ci --ignore-scripts"
        )
    return str(path)


def _base_env() -> dict[str, str]:
    keep = ("PATH", "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME")
    return {name: os.environ[name] for name in keep if name in os.environ}


def build_mcp_toolsets(
    settings: Settings, names: tuple[str, ...] | None = None
) -> list[McpToolset]:
    selected = settings.mcp_servers if names is None else names
    unknown = set(selected) - set(MCP_TOOL_FILTERS)
    if unknown:
        raise ValueError(f"unknown MCP servers: {', '.join(sorted(unknown))}")
    if not selected:
        return []
    secrets = load_mcp_secrets()
    toolsets: list[McpToolset] = []
    for name in selected:
        env = _base_env()
        if name == "context7":
            key = secrets.get("CONTEXT7_API_KEY", "")
            if not key:
                raise RuntimeError("CONTEXT7_API_KEY is required for Context7")
            env["CONTEXT7_API_KEY"] = key
            command, args = _bin("context7-mcp"), []
        elif name == "brave-search":
            key = secrets.get("BRAVE_API_KEY", "")
            if not key:
                raise RuntimeError("BRAVE_API_KEY is required for Brave Search")
            env["BRAVE_API_KEY"] = key
            command, args = _bin("brave-search-mcp-server"), ["--transport", "stdio"]
        else:
            env["NODE_TLS_REJECT_UNAUTHORIZED"] = "0"
            command = _bin("chrome-devtools-mcp")
            args = [
                "--browserUrl",
                settings.chrome_browser_url,
                "--acceptInsecureCerts",
                "--no-usageStatistics",
                "--no-performance-crux",
            ]
        toolsets.append(
            McpToolset(
                connection_params=StdioConnectionParams(
                    server_params=StdioServerParameters(
                        command=command, args=args, env=env
                    ),
                    timeout=30,
                ),
                tool_filter=MCP_TOOL_FILTERS[name],
            )
        )
    return toolsets


async def close_mcp_toolsets(toolsets: list[McpToolset]) -> None:
    for toolset in toolsets:
        try:
            await toolset.close()
        except Exception:
            pass
