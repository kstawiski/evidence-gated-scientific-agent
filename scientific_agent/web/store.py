"""SQLite metadata and path-confined workspace storage."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, Iterator

from ..provenance import utc_now


WORKSPACE_NAME = re.compile(r"^[^\x00-\x1f]{1,80}$")
RUN_STATES = {
    "queued",
    "running",
    "supported",
    "supported_with_comments",
    "contradicted",
    "inconclusive",
    "requires_more_evidence",
    "requires_human_decision",
    "failed",
    "interrupted",
}
ACTIVE_RUN_STATES = {"queued", "running"}


class WorkspaceStore:
    def __init__(self, database_path: Path, workspaces_dir: Path):
        self.database_path = database_path.resolve()
        self.workspaces_dir = workspaces_dir.resolve()
        self.database_path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.workspaces_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        os.chmod(self.database_path.parent, 0o700)
        os.chmod(self.workspaces_dir, 0o700)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    external_key TEXT UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
                    objective TEXT NOT NULL,
                    enable_code INTEGER NOT NULL,
                    mcp_servers TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    provenance_dir TEXT,
                    error_type TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_workspace_created
                    ON runs(workspace_id, created_at DESC);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_one_active
                    ON runs(workspace_id)
                    WHERE status IN ('queued', 'running');
                """
            )
            connection.execute(
                """
                UPDATE runs
                SET status='interrupted', phase='stopped',
                    message='Service restarted before this run completed',
                    finished_at=?
                WHERE status IN ('queued', 'running')
                """,
                (utc_now(),),
            )

    def _workspace_root(self, workspace_id: str) -> Path:
        try:
            uuid.UUID(workspace_id)
        except ValueError as exc:
            raise KeyError("workspace not found") from exc
        root = (self.workspaces_dir / workspace_id).resolve()
        if root.parent != self.workspaces_dir:
            raise KeyError("workspace not found")
        return root

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        return dict(row) if row is not None else None

    def create_workspace(self, name: str, external_key: str | None = None) -> dict:
        name = name.strip()
        if not WORKSPACE_NAME.fullmatch(name):
            raise ValueError("workspace name must be 1-80 printable characters")
        workspace_id = str(uuid.uuid4())
        now = utc_now()
        root = self._workspace_root(workspace_id)
        (root / "files").mkdir(parents=True, mode=0o700)
        (root / "runs").mkdir(mode=0o700)
        try:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO workspaces VALUES (?, ?, ?, ?, ?)",
                    (workspace_id, name, external_key, now, now),
                )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise
        return self.get_workspace(workspace_id)

    def get_or_create_external_workspace(self, external_key: str, name: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM workspaces WHERE external_key=?", (external_key,)
            ).fetchone()
        if row is not None:
            return dict(row)
        try:
            return self.create_workspace(name, external_key=external_key)
        except sqlite3.IntegrityError:
            with self._connection() as connection:
                row = connection.execute(
                    "SELECT * FROM workspaces WHERE external_key=?", (external_key,)
                ).fetchone()
            if row is None:
                raise
            return dict(row)

    def list_workspaces(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT w.*,
                       (SELECT COUNT(*) FROM runs r WHERE r.workspace_id=w.id) AS run_count,
                       (SELECT COUNT(*) FROM runs r WHERE r.workspace_id=w.id AND r.status IN ('queued','running')) AS active_runs
                FROM workspaces w ORDER BY w.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_workspace(self, workspace_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM workspaces WHERE id=?", (workspace_id,)
            ).fetchone()
        if row is None:
            raise KeyError("workspace not found")
        return dict(row)

    def paths(self, workspace_id: str) -> tuple[Path, Path]:
        self.get_workspace(workspace_id)
        root = self._workspace_root(workspace_id)
        return root / "files", root / "runs"

    def delete_workspace(self, workspace_id: str) -> None:
        self.get_workspace(workspace_id)
        if self.has_active_run(workspace_id):
            raise RuntimeError("cannot delete a workspace with an active run")
        root = self._workspace_root(workspace_id)
        with self._connection() as connection:
            connection.execute("DELETE FROM workspaces WHERE id=?", (workspace_id,))
        shutil.rmtree(root)

    @staticmethod
    def _safe_filename(filename: str) -> str:
        if "/" in filename or "\\" in filename:
            raise ValueError("filenames must not contain a path")
        name = filename.strip()
        if not name or name in {".", ".."} or "\x00" in name or len(name) > 180:
            raise ValueError("invalid filename")
        if any(ord(character) < 32 for character in name):
            raise ValueError("invalid filename")
        return name

    def save_file(
        self,
        workspace_id: str,
        filename: str,
        source: BinaryIO,
        max_bytes: int,
        *,
        overwrite: bool = False,
    ) -> dict:
        files_dir, _ = self.paths(workspace_id)
        name = self._safe_filename(filename)
        destination = files_dir / name
        if destination.exists() and not overwrite:
            raise FileExistsError(name)
        temporary = files_dir / f".upload-{uuid.uuid4().hex}"
        written = 0
        try:
            with temporary.open("xb") as handle:
                while chunk := source.read(1024 * 1024):
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"file exceeds {max_bytes} byte upload limit")
                    handle.write(chunk)
            temporary.chmod(0o600)
            temporary.replace(destination)
            destination.chmod(0o600)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        with self._connection() as connection:
            connection.execute(
                "UPDATE workspaces SET updated_at=? WHERE id=?",
                (utc_now(), workspace_id),
            )
        return {"name": name, "bytes": written}

    def list_files(self, workspace_id: str) -> list[dict]:
        files_dir, _ = self.paths(workspace_id)
        return [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "modified_at": path.stat().st_mtime,
            }
            for path in sorted(files_dir.iterdir(), key=lambda item: item.name.lower())
            if path.is_file() and not path.is_symlink() and not path.name.startswith(".upload-")
        ]

    def file_path(self, workspace_id: str, filename: str) -> Path:
        files_dir, _ = self.paths(workspace_id)
        path = (files_dir / self._safe_filename(filename)).resolve()
        if path.parent != files_dir or not path.is_file() or path.is_symlink():
            raise KeyError("file not found")
        return path

    def delete_file(self, workspace_id: str, filename: str) -> None:
        if self.has_active_run(workspace_id):
            raise RuntimeError("cannot delete files while a run is active")
        self.file_path(workspace_id, filename).unlink()

    def has_active_run(self, workspace_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs WHERE workspace_id=? AND status IN ('queued','running') LIMIT 1",
                (workspace_id,),
            ).fetchone()
        return row is not None

    def create_run(
        self,
        workspace_id: str,
        objective: str,
        enable_code: bool,
        mcp_servers: tuple[str, ...],
    ) -> dict:
        self.get_workspace(workspace_id)
        objective = objective.strip()
        if not 3 <= len(objective) <= 20_000:
            raise ValueError("objective must contain 3-20,000 characters")
        if self.has_active_run(workspace_id):
            raise RuntimeError("this workspace already has an active run")
        run_id = str(uuid.uuid4())
        now = utc_now()
        try:
            with self._connection() as connection:
                connection.execute(
                    """
                    INSERT INTO runs
                    (id, workspace_id, objective, enable_code, mcp_servers, status,
                     phase, message, created_at)
                    VALUES (?, ?, ?, ?, ?, 'queued', 'queued', 'Waiting for an execution slot', ?)
                    """,
                    (
                        run_id,
                        workspace_id,
                        objective,
                        int(enable_code),
                        json.dumps(list(mcp_servers)),
                        now,
                    ),
                )
                connection.execute(
                    "UPDATE workspaces SET updated_at=? WHERE id=?",
                    (now, workspace_id),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("this workspace already has an active run") from exc
        return self.get_run(run_id)

    def update_run(self, run_id: str, **updates) -> None:
        allowed = {
            "status",
            "phase",
            "message",
            "started_at",
            "finished_at",
            "provenance_dir",
            "error_type",
        }
        if not updates or set(updates) - allowed:
            raise ValueError("invalid run update")
        if "status" in updates and updates["status"] not in RUN_STATES:
            raise ValueError("invalid run status")
        assignments = ", ".join(f"{name}=?" for name in updates)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE runs SET {assignments} WHERE id=?",  # noqa: S608 - fixed allow-list
                (*updates.values(), run_id),
            )
        if cursor.rowcount != 1:
            raise KeyError("run not found")

    def get_run(self, run_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError("run not found")
        value = dict(row)
        value["enable_code"] = bool(value["enable_code"])
        value["mcp_servers"] = json.loads(value["mcp_servers"])
        return value

    def list_runs(self, workspace_id: str) -> list[dict]:
        self.get_workspace(workspace_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM runs WHERE workspace_id=? ORDER BY created_at DESC",
                (workspace_id,),
            ).fetchall()
        values = []
        for row in rows:
            value = dict(row)
            value["enable_code"] = bool(value["enable_code"])
            value["mcp_servers"] = json.loads(value["mcp_servers"])
            values.append(value)
        return values

    def run_artifact(self, run_id: str, relative_path: str) -> Path:
        run = self.get_run(run_id)
        provenance = run.get("provenance_dir")
        if not provenance:
            raise KeyError("artifact not found")
        root = Path(provenance).resolve()
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise KeyError("artifact not found")
        if not candidate.is_file() or candidate.is_symlink():
            raise KeyError("artifact not found")
        return candidate
