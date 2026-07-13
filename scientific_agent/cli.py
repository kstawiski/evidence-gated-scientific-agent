"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from .config import Settings
from .orchestrator import run
from .preflight import preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scientific-agent")
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("preflight", help="check model endpoints and MCP schemas")
    check.add_argument(
        "--mcp",
        default="context7,brave-search",
        help="comma-separated MCP servers to validate; pass an empty value for none",
    )
    check.add_argument(
        "--enable-code",
        action="store_true",
        help="also validate the configured Python/R sandbox runtime",
    )
    execute = sub.add_parser("run", help="run one evidence-gated scientific task")
    execute.add_argument(
        "objective",
        nargs="?",
        help="task text, '-' to read private stdin, or omit when using --prompt-file",
    )
    execute.add_argument(
        "--prompt-file",
        type=Path,
        help="read the task from an owner-only regular file (keeps it out of argv)",
    )
    execute.add_argument(
        "--mcp",
        default="context7,brave-search",
        help="comma-separated MCP servers; chrome-devtools is opt-in",
    )
    execute.add_argument(
        "--enable-code",
        action="store_true",
        help="authorize offline, resource-bounded Python and R analysis tools",
    )
    execute.add_argument(
        "--mode",
        choices=("simple", "full"),
        default="simple",
        help="simple uses one lean Qwen plan and one final Gemma audit; full uses dual planning",
    )
    return parser


def _read_objective(objective: str | None, prompt_file: Path | None) -> str:
    if prompt_file is not None and objective is not None:
        raise ValueError("use either objective/stdin or --prompt-file, not both")
    if prompt_file is not None:
        info = prompt_file.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise PermissionError("prompt file must be a regular non-symlink")
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077:
            raise PermissionError("prompt file must be current-user-owned and private")
        text = prompt_file.read_text(encoding="utf-8")
    elif objective == "-":
        text = sys.stdin.read()
    elif objective is not None:
        text = objective
    else:
        raise ValueError("task required: pass '-', --prompt-file, or an objective")
    text = text.strip()
    if not text:
        raise ValueError("task must not be empty")
    return text


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = Settings()
    try:
        if args.command == "preflight":
            names = tuple(name.strip() for name in args.mcp.split(",") if name.strip())
            print(
                json.dumps(
                    preflight(
                        settings,
                        mcp_names=names,
                        include_code=args.enable_code,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        objective = _read_objective(args.objective, args.prompt_file)
        names = tuple(name.strip() for name in args.mcp.split(",") if name.strip())
        include_chrome = "chrome-devtools" in names
        result = run(
            objective,
            settings,
            mcp_names=names,
            include_chrome=include_chrome,
            enable_code=args.enable_code,
            simple_mode=args.mode == "simple",
        )
        print(result.model_dump_json(indent=2))
        return 0 if result.status in {"supported", "supported_with_comments"} else 3
    except Exception as exc:
        print(f"scientific-agent: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
