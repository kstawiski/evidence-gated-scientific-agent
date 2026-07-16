"""Deterministic, credential-free integration downloads for the Web UI."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SKILL_FILES = (
    "SKILL.md",
    "agents/openai.yaml",
    "scripts/evidence_bench.py",
)
_A2A_FILES = ("README.md", "a2a_client.py")


@dataclass(frozen=True)
class IntegrationArchive:
    """One reproducible archive exposed through a fixed download route."""

    filename: str
    content: bytes
    sha256: str


def _source_dir(name: str) -> Path:
    """Find a force-included wheel resource or its source-tree counterpart."""

    packaged = Path(__file__).with_name("release_assets") / name
    source = Path(__file__).resolve().parents[2] / (
        "skills/evidence-bench" if name == "evidence-bench" else "integrations/a2a"
    )
    for candidate in (packaged, source):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(f"required integration asset is missing: {name}")


def _read_allowed(root: Path, names: Iterable[str]) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    resolved_root = root.resolve(strict=True)
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("integration asset allowlist contains an unsafe path")
        source = root / relative
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(f"integration asset is not a regular file: {name}")
        resolved = source.resolve(strict=True)
        if not resolved.is_relative_to(resolved_root):
            raise RuntimeError(f"integration asset escapes its root: {name}")
        files[name] = source.read_bytes()
    return files


def _checksums(files: dict[str, bytes]) -> bytes:
    lines = [
        f"{hashlib.sha256(data).hexdigest()}  {name}"
        for name, data in sorted(files.items())
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _archive(
    filename: str, root_name: str, files: dict[str, bytes]
) -> IntegrationArchive:
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(f"{root_name}/{name}", date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            mode = 0o755 if name.endswith(".py") else 0o644
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, data)
    content = buffer.getvalue()
    return IntegrationArchive(
        filename=filename,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def build_skill_archive(version: str) -> IntegrationArchive:
    """Build the installable Codex/Claude skill from an explicit file allowlist."""

    files = _read_allowed(_source_dir("evidence-bench"), _SKILL_FILES)
    files["SHA256SUMS"] = _checksums(files)
    return _archive(
        f"evidence-bench-skill-v{version}.zip",
        "evidence-bench",
        files,
    )


def build_a2a_archive(
    version: str, public_url: str, enabled: bool
) -> IntegrationArchive:
    """Build a deployment-specific starter without embedding the bearer token."""

    files = _read_allowed(_source_dir("a2a"), _A2A_FILES)
    base_url = public_url.rstrip("/")
    connection = {
        "a2a_enabled": enabled,
        "a2a_url": f"{base_url}/a2a",
        "agent_card_url": f"{base_url}/.well-known/agent-card.json",
        "protocol_binding": "JSONRPC",
        "protocol_version": "1.0",
        "service_url": base_url,
        "service_version": version,
    }
    files["connection.json"] = (
        json.dumps(connection, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    files["SHA256SUMS"] = _checksums(files)
    return _archive(
        f"evidence-bench-a2a-v{version}.zip",
        "evidence-bench-a2a",
        files,
    )
