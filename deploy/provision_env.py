#!/usr/bin/env python3
"""Create an owner-only deployment environment without printing credentials."""

from __future__ import annotations

import argparse
import os
import secrets
import shlex
import stat
import tempfile
from pathlib import Path


PUBLIC_KEYS = (
    "WEB_USERNAME",
    "WEB_AUTH_ENABLED",
    "WEB_BIND_ADDRESS",
    "WEB_PUBLISHED_PORT",
    "SCIENTIFIC_AGENT_PUBLIC_URL",
    "EVIDENCE_BENCH_DATA_PATH",
    "EVIDENCE_BENCH_ENVIRONMENTS_PATH",
    "EVIDENCE_BENCH_BROWSER_PATH",
    "BROWSER_BIND_ADDRESS",
    "BROWSER_NOVNC_PORT",
    "BROWSER_PUBLIC_URL",
    "BROWSER_GEOMETRY",
    "BROWSER_RUNTIME_USER",
    "BROWSER_DOWNLOADS_MODE",
    "QWEN_BASE_URL",
    "QWEN_MODEL",
    "QWEN_API_KEY",
    "QWEN_MAX_TOKENS",
    "QWEN_ENABLE_THINKING",
    "QWEN_NATIVE_JSON_SCHEMA",
    "QWEN_REQUEST_TIMEOUT_SECONDS",
    "GEMMA_BASE_URL",
    "GEMMA_MODEL",
    "GEMMA_API_KEY",
    "GEMMA_MAX_TOKENS",
    "GEMMA_ENABLE_THINKING",
    "GEMMA_NATIVE_JSON_SCHEMA",
    "GEMMA_REQUEST_TIMEOUT_SECONDS",
    "CHROME_DEVTOOLS_BROWSER_URL",
    "SCIENTIFIC_AGENT_NCBI_EMAIL",
    "SCIENTIFIC_AGENT_NCBI_TOOL",
    "SCIENTIFIC_AGENT_NCBI_API_KEY",
    "WEB_MAX_WORKERS",
    "MAX_REPAIR_ROUNDS",
    "SCIENTIFIC_AGENT_MAX_RESEARCH_MODEL_TURNS",
    "SCIENTIFIC_AGENT_MAX_RESEARCH_TOOL_CALLS",
    "SCIENTIFIC_AGENT_MAX_REPEATED_TOOL_RESULTS",
    "SCIENTIFIC_AGENT_MAX_WALL_SECONDS",
    "SCIENTIFIC_AGENT_MAX_MEMORY_BYTES",
    "SCIENTIFIC_AGENT_MAX_PROCESSES",
    "SCIENTIFIC_AGENT_MAX_FILE_BYTES",
    "SCIENTIFIC_AGENT_MAX_OUTPUT_BYTES",
    "SCIENTIFIC_AGENT_MAX_CODE_BYTES",
    "SCIENTIFIC_AGENT_MAX_CODE_CALLS",
    "SCIENTIFIC_AGENT_MAX_PACKAGES_PER_CALL",
    "SCIENTIFIC_AGENT_PACKAGE_TIMEOUT_SECONDS",
    "SCIENTIFIC_AGENT_MAX_ENVIRONMENT_BYTES",
    "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_BYTES",
    "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_BYTES",
    "SCIENTIFIC_AGENT_MAX_ENVIRONMENT_ENTRIES",
    "SCIENTIFIC_AGENT_MAX_WORKSPACE_ENVIRONMENT_ENTRIES",
    "SCIENTIFIC_AGENT_MAX_TOTAL_ENVIRONMENT_ENTRIES",
)
SECRET_KEYS = (
    "WEB_PASSWORD",
    "A2A_TOKEN",
    "SANDBOX_WORKER_TOKEN",
    "PACKAGE_WORKER_TOKEN",
)
MCP_KEYS = ("CONTEXT7_API_KEY", "BRAVE_API_KEY")


def parse_env(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        return {}
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        name, raw_value = line.split("=", 1)
        parsed = shlex.split(raw_value, posix=True)
        if len(parsed) == 1:
            values[name.strip()] = parsed[0]
    return values


def quote(value: str) -> str:
    return shlex.quote(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mcp-env-file", type=Path)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    existing = parse_env(output)
    mcp = (
        parse_env(args.mcp_env_file.expanduser().resolve()) if args.mcp_env_file else {}
    )

    required = ("SCIENTIFIC_AGENT_PUBLIC_URL", "QWEN_BASE_URL", "GEMMA_BASE_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"missing deployment setting(s): {', '.join(missing)}")

    auth_enabled = os.environ.get(
        "WEB_AUTH_ENABLED", existing.get("WEB_AUTH_ENABLED", "true")
    ).lower() not in {"0", "false", "no", "off"}
    values = {
        "A2A_ENABLED": "true",
        **{
            name: existing.get(name) or secrets.token_urlsafe(36)
            for name in SECRET_KEYS
            if name != "WEB_PASSWORD"
        },
    }
    if auth_enabled:
        values["WEB_USERNAME"] = os.environ.get(
            "WEB_USERNAME", existing.get("WEB_USERNAME", "scientist")
        )
        values["WEB_PASSWORD"] = existing.get("WEB_PASSWORD") or secrets.token_urlsafe(
            36
        )
    for name in PUBLIC_KEYS:
        if os.environ.get(name):
            values[name] = os.environ[name]
    for name in MCP_KEYS:
        if mcp.get(name):
            values[name] = mcp[name]

    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{output.name}-", dir=output.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for name in values:
                handle.write(f"{name}={quote(values[name])}\n")
        temporary.replace(output)
        output.chmod(0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"wrote owner-only deployment configuration to {output}")


if __name__ == "__main__":
    main()
