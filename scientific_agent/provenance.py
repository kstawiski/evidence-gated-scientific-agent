"""Append-only event and artifact provenance helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class EventLedger:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(mode=0o600, exist_ok=True)
        os.chmod(self.path, 0o600)

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"timestamp": utc_now(), "event_type": event_type, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(record) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_manifest(run_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "manifest.json"):
        files.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {"created_at": utc_now(), "files": files}
    write_json(run_dir / "manifest.json", manifest)
    return manifest


def build_input_manifest(workspace: Path) -> dict[str, Any]:
    """Describe immutable run inputs without copying potentially sensitive data."""

    root = workspace.resolve()
    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if path.is_symlink() or not path.is_file():
            continue
        files.append(
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {"created_at": utc_now(), "files": files}


def build_environment_snapshot(*, application_version: str) -> dict[str, Any]:
    """Capture reproducibility-relevant controller/runtime identity."""

    lockfiles = []
    for path in (Path("/app/uv.lock"), Path("/var/lib/dpkg/status")):
        if path.is_file():
            lockfiles.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "created_at": utc_now(),
        "application": {"name": "Evidence Bench", "version": application_version},
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "dependency_lockfiles": lockfiles,
    }
