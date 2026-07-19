"""Read-only, path-confined workspace tools for the Qwen executor."""

from __future__ import annotations

import hashlib
import re
import stat
from collections.abc import Callable, Iterable
from pathlib import Path

from .schemas import ArtifactRef


MAX_READ_BYTES = 2 * 1024 * 1024
MAX_SEARCH_HITS = 200
GENERATED_TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".tsv", ".txt"}


def build_workspace_tools(
    root: Path,
    registered_artifacts: Callable[[], Iterable[ArtifactRef]] | None = None,
):
    workspace = root.resolve()

    def resolve(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() and candidate.parts[:2] == ("/", "workspace"):
            candidate = Path(*candidate.parts[2:])
        if not candidate.is_absolute():
            candidate = workspace / candidate
        current = candidate
        while current != workspace and workspace in current.parents:
            if current.is_symlink():
                raise ValueError(f"symlinks are not allowed in workspace paths: {path}")
            current = current.parent
        resolved = candidate.resolve()
        if resolved != workspace and workspace not in resolved.parents:
            raise ValueError(f"path escapes workspace: {path}")
        return resolved

    def resolve_registered_artifact(path: str) -> tuple[Path, str]:
        if registered_artifacts is None:
            raise ValueError(f"path escapes workspace: {path}")
        for artifact in registered_artifacts():
            if (
                path != artifact.path
                or artifact.description != "sandbox-generated analysis artifact"
                or not artifact.sha256
                or not re.fullmatch(r"[0-9a-fA-F]{64}", artifact.sha256)
            ):
                continue
            candidate = Path(path)
            if (
                not candidate.is_absolute()
                or candidate.suffix.casefold() not in GENERATED_TEXT_SUFFIXES
            ):
                break
            resolved = candidate.resolve(strict=True)
            if resolved != candidate:
                raise ValueError(
                    f"registered computation artifact path contains a symlink: {path}"
                )
            current = candidate
            while True:
                if stat.S_ISLNK(current.lstat().st_mode):
                    raise ValueError(
                        "registered computation artifact path contains a symlink: "
                        f"{path}"
                    )
                if current == current.parent:
                    break
                current = current.parent
            if not stat.S_ISREG(candidate.lstat().st_mode):
                raise ValueError(
                    f"registered computation artifact is not a regular file: {path}"
                )
            return candidate, artifact.sha256.casefold()
        raise ValueError(
            f"absolute path is not an exact registered successful text artifact: {path}"
        )

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
            return {
                "path": str(target.relative_to(workspace)),
                "entries": entries[:1000],
            }
        except Exception as exc:  # surface a structured, non-fatal tool error
            return {"error": str(exc)}

    def read_text_file(path: str) -> dict:
        """Read bounded UTF-8 workspace or registered computation text.

        Args:
            path: A workspace-relative path, or the exact absolute path of a
                controller-registered successful generated text artifact.
        """
        try:
            candidate = Path(path)
            registered_hash = None
            if candidate.is_absolute() and candidate.parts[:2] != ("/", "workspace"):
                absolute = candidate.resolve()
                if absolute == workspace or workspace in absolute.parents:
                    target = resolve(path)
                else:
                    target, registered_hash = resolve_registered_artifact(path)
            else:
                target = resolve(path)
            size = target.stat().st_size
            if size > MAX_READ_BYTES:
                return {
                    "error": f"file exceeds {MAX_READ_BYTES} byte read limit",
                    "bytes": size,
                }
            data = target.read_bytes()
            if registered_hash is not None:
                observed_hash = hashlib.sha256(data).hexdigest()
                if observed_hash != registered_hash:
                    return {
                        "error": "registered computation artifact hash mismatch",
                        "path": str(target),
                        "expected_sha256": registered_hash,
                        "observed_sha256": observed_hash,
                    }
                return {
                    "path": str(target),
                    "content": data.decode("utf-8", errors="replace"),
                    "source": "registered_computation_artifact",
                    "sha256": observed_hash,
                }
            return {
                "path": str(target.relative_to(workspace)),
                "content": data.decode("utf-8", errors="replace"),
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
            files = [target] if target.is_file() else target.rglob("*")
            hits: list[str] = []
            for file in files:
                if file.is_symlink():
                    continue
                confined = file.resolve()
                if (
                    confined != workspace and workspace not in confined.parents
                ) or not confined.is_file():
                    continue
                if confined.stat().st_size > MAX_READ_BYTES:
                    continue
                for line_no, line in enumerate(
                    confined.read_text(encoding="utf-8", errors="replace").splitlines(),
                    1,
                ):
                    if rx.search(line):
                        hits.append(
                            f"{confined.relative_to(workspace)}:{line_no}:{line}"
                        )
                        if len(hits) >= MAX_SEARCH_HITS:
                            return {"matches": hits, "truncated": True}
            return {"matches": hits, "truncated": False}
        except Exception as exc:
            return {"error": str(exc)}

    return [list_workspace, read_text_file, search_workspace]
