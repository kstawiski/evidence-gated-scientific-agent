"""Endpoint and MCP capability preflight without exposing secrets."""

from __future__ import annotations

import asyncio
import json
import urllib.request
from pathlib import Path

from .config import PROJECT_ROOT, Settings, load_mcp_secrets
from .execution import sandbox_preflight
from .mcp import MCP_TOOL_FILTERS, build_mcp_toolsets, close_mcp_toolsets


def _models(base_url: str) -> list[str]:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
        payload = json.load(response)
    return [item["id"] for item in payload.get("data", [])]


async def run_preflight(
    settings: Settings,
    mcp_names: tuple[str, ...] | None = None,
    include_code: bool = False,
) -> dict:
    selected = mcp_names if mcp_names is not None else settings.mcp_servers
    result: dict = {
        "qwen": {"models": _models(settings.qwen.base_url)},
        "gemma": {"models": _models(settings.gemma.base_url)},
        "secrets": {name: True for name in load_mcp_secrets()},
        "mcp": {},
    }
    if settings.qwen.model not in result["qwen"]["models"]:
        raise RuntimeError(f"Qwen model not advertised: {settings.qwen.model}")
    if settings.gemma.model not in result["gemma"]["models"]:
        raise RuntimeError(f"Gemma model not advertised: {settings.gemma.model}")
    if include_code:
        result["sandbox"] = sandbox_preflight(settings.sandbox, settings.workspace)
        if result["sandbox"]["missing_required"]:
            raise RuntimeError(
                "sandbox runtime paths are missing: "
                + ", ".join(result["sandbox"]["missing_required"])
            )
        failed_probes = [
            name
            for name, status in result["sandbox"]["probes"].items()
            if status != "succeeded"
        ]
        if failed_probes:
            raise RuntimeError(
                "sandbox runtime probes failed: " + ", ".join(failed_probes)
            )
    for name in selected:
        attempts = 0
        while True:
            attempts += 1
            toolset = build_mcp_toolsets(settings, (name,))[0]
            try:
                tools = await toolset.get_tools()
                break
            except ConnectionError:
                if attempts >= 2:
                    raise
                await asyncio.sleep(0.25)
            finally:
                await close_mcp_toolsets([toolset])
        names = sorted(tool.name for tool in tools)
        missing = sorted(set(MCP_TOOL_FILTERS[name]) - set(names))
        result["mcp"][name] = {
            "tools": names,
            "missing_required": missing,
            "startup_attempts": attempts,
        }
        if missing:
            raise RuntimeError(f"MCP {name} missing required tools: {', '.join(missing)}")
    return result


def preflight(
    settings: Settings,
    mcp_names: tuple[str, ...] | None = None,
    include_code: bool = False,
) -> dict:
    return asyncio.run(
        run_preflight(
            settings,
            mcp_names=mcp_names,
            include_code=include_code,
        )
    )
