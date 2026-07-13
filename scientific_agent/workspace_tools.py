"""Read-only, path-confined workspace tools for the Qwen executor."""

from __future__ import annotations

import re
from pathlib import Path


MAX_READ_BYTES = 2 * 1024 * 1024
MAX_SEARCH_HITS = 200


def build_workspace_tools(root: Path):
    workspace = root.resolve()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        resolved = candidate.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return resolved

    def list_workspace(path: str = ".") -> dict:
        """List files and directories inside the assigned workspace.

        Args:
            path: Relative directory path within the assigned workspace.
        """
        try:
            target = resolve(path)
            entries = [
                item.name + ("/" if item.is_dir() else "")
                for item in sorted(target.iterdir(), key=lambda item: item.name)
            ]
            return {"path": str(target.relative_to(workspace)), "entries": entries[:1000]}
        except Exception as exc:  # surface a structured, non-fatal tool error
            return {"error": str(exc)}

    def read_text_file(path: str) -> dict:
        """Read a bounded UTF-8 text file inside the assigned workspace.

        Args:
            path: Relative file path within the assigned workspace.
        """
        try:
            target = resolve(path)
            size = target.stat().st_size
            if size > MAX_READ_BYTES:
                return {"error": f"file exceeds {MAX_READ_BYTES} byte read limit", "bytes": size}
            return {
                "path": str(target.relative_to(workspace)),
                "content": target.read_text(encoding="utf-8", errors="replace"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def search_workspace(pattern: str, path: str = ".") -> dict:
        """Regex-search bounded text files inside the assigned workspace.

        Args:
            pattern: Python regular expression.
            path: Relative file or directory path within the workspace.
        """
        try:
            target = resolve(path)
            rx = re.compile(pattern)
            files = [target] if target.is_file() else [p for p in target.rglob("*") if p.is_file()]
            hits: list[str] = []
            for file in files:
                if file.stat().st_size > MAX_READ_BYTES:
                    continue
                for line_no, line in enumerate(
                    file.read_text(encoding="utf-8", errors="replace").splitlines(), 1
                ):
                    if rx.search(line):
                        hits.append(f"{file.relative_to(workspace)}:{line_no}:{line}")
                        if len(hits) >= MAX_SEARCH_HITS:
                            return {"matches": hits, "truncated": True}
            return {"matches": hits, "truncated": False}
        except Exception as exc:
            return {"error": str(exc)}

    return [list_workspace, read_text_file, search_workspace]
