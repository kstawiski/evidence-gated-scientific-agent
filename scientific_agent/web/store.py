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
from typing import AsyncIterable, BinaryIO, Iterator

from ..provenance import utc_now


WORKSPACE_NAME = re.compile(r"^[^\x00-\x1f]{1,80}$")
RUN_STATES = {
    "queued",
    "running",
    "cancel_requested",
    "cancelled",
    "supported",
    "supported_with_comments",
    "contradicted",
    "inconclusive",
    "requires_more_evidence",
    "requires_human_decision",
    "failed",
    "interrupted",
}
ACTIVE_RUN_STATES = {"queued", "running", "cancel_requested"}


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
        except Exception:
            connection.rollback()
            raise
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
                    error_type TEXT,
                    parent_run_id TEXT REFERENCES runs(id),
                    run_kind TEXT NOT NULL DEFAULT 'analysis',
                    cancel_requested_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_workspace_created
                    ON runs(workspace_id, created_at DESC);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(runs)").fetchall()
            }
            migrations = {
                "parent_run_id": "ALTER TABLE runs ADD COLUMN parent_run_id TEXT",
                "run_kind": (
                    "ALTER TABLE runs ADD COLUMN run_kind TEXT NOT NULL "
                    "DEFAULT 'analysis'"
                ),
                "cancel_requested_at": (
                    "ALTER TABLE runs ADD COLUMN cancel_requested_at TEXT"
                ),
                "knowledge_snapshot": (
                    "ALTER TABLE runs ADD COLUMN knowledge_snapshot TEXT NOT NULL "
                    "DEFAULT '{}'"
                ),
                "requested_outputs": (
                    "ALTER TABLE runs ADD COLUMN requested_outputs TEXT NOT NULL "
                    "DEFAULT '[]'"
                ),
            }
            for column, statement in migrations.items():
                if column not in columns:
                    connection.execute(statement)
            connection.executescript(
                """
                DROP INDEX IF EXISTS idx_runs_one_active;
                CREATE UNIQUE INDEX idx_runs_one_active
                    ON runs(workspace_id)
                    WHERE status IN ('queued', 'running', 'cancel_requested');
                CREATE INDEX IF NOT EXISTS idx_runs_parent
                    ON runs(parent_run_id, created_at);
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    phase TEXT NOT NULL,
                    message TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    artifact_path TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_run_events_run_id
                    ON run_events(run_id, id);
                CREATE TABLE IF NOT EXISTS report_discussion (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence_refs TEXT NOT NULL DEFAULT '[]',
                    unresolved_uncertainties TEXT NOT NULL DEFAULT '[]',
                    suggested_revision_prompt TEXT,
                    status TEXT NOT NULL,
                    model TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_report_discussion_run_id
                    ON report_discussion(run_id, id);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_report_discussion_generating
                    ON report_discussion(run_id)
                    WHERE status='generating';
                """
            )
            interrupted = connection.execute(
                """
                SELECT id FROM runs
                WHERE status IN ('queued', 'running', 'cancel_requested')
                """
            ).fetchall()
            stopped_at = utc_now()
            connection.execute(
                """
                UPDATE runs
                SET status='interrupted', phase='stopped',
                    message='Service restarted before this run completed',
                    finished_at=?
                WHERE status IN ('queued', 'running', 'cancel_requested')
                """,
                (stopped_at,),
            )
            for row in interrupted:
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'service_restart', 'interrupted', 'stopped',
                            'Service restarted before this run completed',
                            'Controller', ?)
                    """,
                    (row["id"], stopped_at),
                )
            connection.execute(
                """
                UPDATE report_discussion
                SET status='failed', content='Gemma response was interrupted by a service restart',
                    updated_at=?
                WHERE status='generating'
                """,
                (stopped_at,),
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
                       (SELECT COUNT(*) FROM runs r WHERE r.workspace_id=w.id AND r.status IN ('queued','running','cancel_requested')) AS active_runs
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

    def delete_workspace(self, workspace_id: str, before_delete=None) -> None:
        self.get_workspace(workspace_id)
        root = self._workspace_root(workspace_id)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM runs WHERE workspace_id=?
                AND status IN ('queued','running','cancel_requested') LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
            if active is not None:
                raise RuntimeError("cannot delete a workspace with an active run")
            if before_delete is not None:
                before_delete()
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
        temporary = files_dir / f".upload-{uuid.uuid4().hex}"
        written = 0
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                active = connection.execute(
                    """
                    SELECT 1 FROM runs
                    WHERE workspace_id=?
                      AND status IN ('queued','running','cancel_requested')
                    LIMIT 1
                    """,
                    (workspace_id,),
                ).fetchone()
                if active is not None:
                    raise RuntimeError("cannot upload files while a run is active")
                if destination.exists() and not overwrite:
                    raise FileExistsError(name)
                with temporary.open("xb") as handle:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > max_bytes:
                            raise ValueError(
                                f"file exceeds {max_bytes} byte upload limit"
                            )
                        handle.write(chunk)
                temporary.chmod(0o600)
                temporary.replace(destination)
                destination.chmod(0o600)
                connection.execute(
                    "UPDATE workspaces SET updated_at=? WHERE id=?",
                    (utc_now(), workspace_id),
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return {"name": name, "bytes": written}

    async def save_streamed_file(
        self,
        workspace_id: str,
        filename: str,
        chunks: AsyncIterable[bytes],
        max_bytes: int,
        *,
        overwrite: bool = False,
    ) -> dict:
        """Persist a raw request body without multipart temp-file duplication."""

        files_dir, _ = self.paths(workspace_id)
        name = self._safe_filename(filename)
        destination = files_dir / name
        temporary = files_dir / f".upload-{uuid.uuid4().hex}"
        written = 0
        try:
            # Stream before taking SQLite's service-wide write lock. The short
            # final transaction below serializes the active-run check and atomic
            # rename with create_run(), so a run that starts during a long upload
            # wins and the incomplete input is discarded.
            with temporary.open("xb") as handle:
                async for chunk in chunks:
                    written += len(chunk)
                    if written > max_bytes:
                        raise ValueError(f"file exceeds {max_bytes} byte upload limit")
                    handle.write(chunk)
            temporary.chmod(0o600)
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                active = connection.execute(
                    """
                    SELECT 1 FROM runs
                    WHERE workspace_id=?
                      AND status IN ('queued','running','cancel_requested')
                    LIMIT 1
                    """,
                    (workspace_id,),
                ).fetchone()
                if active is not None:
                    raise RuntimeError("cannot upload files while a run is active")
                if destination.exists() and not overwrite:
                    raise FileExistsError(name)
                temporary.replace(destination)
                destination.chmod(0o600)
                connection.execute(
                    "UPDATE workspaces SET updated_at=? WHERE id=?",
                    (utc_now(), workspace_id),
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
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
            if path.is_file()
            and not path.is_symlink()
            and not path.name.startswith(".upload-")
        ]

    def file_path(self, workspace_id: str, filename: str) -> Path:
        files_dir, _ = self.paths(workspace_id)
        path = (files_dir / self._safe_filename(filename)).resolve()
        if path.parent != files_dir or not path.is_file() or path.is_symlink():
            raise KeyError("file not found")
        return path

    def delete_file(self, workspace_id: str, filename: str) -> None:
        path = self.file_path(workspace_id, filename)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM runs WHERE workspace_id=?
                AND status IN ('queued','running','cancel_requested') LIMIT 1
                """,
                (workspace_id,),
            ).fetchone()
            if active is not None:
                raise RuntimeError("cannot delete files while a run is active")
            path.unlink()
            connection.execute(
                "UPDATE workspaces SET updated_at=? WHERE id=?",
                (utc_now(), workspace_id),
            )

    def has_active_run(self, workspace_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM runs WHERE workspace_id=? AND status IN ('queued','running','cancel_requested') LIMIT 1",
                (workspace_id,),
            ).fetchone()
        return row is not None

    def create_run(
        self,
        workspace_id: str,
        objective: str,
        enable_code: bool,
        mcp_servers: tuple[str, ...],
        *,
        parent_run_id: str | None = None,
        run_kind: str = "analysis",
        knowledge_snapshot: dict | None = None,
        requested_outputs: tuple[str, ...] = (),
    ) -> dict:
        self.get_workspace(workspace_id)
        objective = objective.strip()
        if not 3 <= len(objective) <= 20_000:
            raise ValueError("objective must contain 3-20,000 characters")
        if self.has_active_run(workspace_id):
            raise RuntimeError("this workspace already has an active run")
        run_id = str(uuid.uuid4())
        now = utc_now()
        if run_kind not in {"analysis", "revision"}:
            raise ValueError("invalid run kind")
        try:
            with self._connection() as connection:
                if parent_run_id is not None:
                    parent = connection.execute(
                        "SELECT * FROM runs WHERE id=?", (parent_run_id,)
                    ).fetchone()
                    if parent is None:
                        raise KeyError("parent run not found")
                    if parent["workspace_id"] != workspace_id:
                        raise ValueError("parent run belongs to another workspace")
                    if parent["status"] in ACTIVE_RUN_STATES:
                        raise RuntimeError("parent run is not complete")
                connection.execute(
                    """
                    INSERT INTO runs
                    (id, workspace_id, objective, enable_code, mcp_servers, status,
                     phase, message, created_at, parent_run_id, run_kind,
                    knowledge_snapshot, requested_outputs)
                    VALUES (?, ?, ?, ?, ?, 'queued', 'queued',
                            'Waiting for an execution slot', ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        workspace_id,
                        objective,
                        int(enable_code),
                        json.dumps(list(mcp_servers)),
                        now,
                        parent_run_id,
                        run_kind,
                        json.dumps(knowledge_snapshot or {}, sort_keys=True),
                        json.dumps(list(requested_outputs)),
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'run_queued', 'queued', 'queued',
                            'Waiting for an execution slot', 'Controller', ?)
                    """,
                    (run_id, now),
                )
                connection.execute(
                    "UPDATE workspaces SET updated_at=? WHERE id=?",
                    (now, workspace_id),
                )
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("this workspace already has an active run") from exc
        return self.get_run(run_id)

    @staticmethod
    def _discussion_row(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["evidence_refs"] = json.loads(value["evidence_refs"])
        value["unresolved_uncertainties"] = json.loads(
            value["unresolved_uncertainties"]
        )
        return value

    def list_discussion(self, run_id: str) -> list[dict]:
        self.get_run(run_id)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM report_discussion WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [self._discussion_row(row) for row in rows]

    def start_discussion(self, run_id: str, message: str, model: str) -> int:
        self.get_run(run_id)
        message = message.strip()
        if not 3 <= len(message) <= 20_000:
            raise ValueError("discussion message must contain 3-20,000 characters")
        now = utc_now()
        try:
            with self._connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute(
                    "SELECT 1 FROM report_discussion WHERE run_id=? AND status='generating'",
                    (run_id,),
                ).fetchone():
                    raise RuntimeError("Gemma is already answering this report")
                connection.execute(
                    """
                    INSERT INTO report_discussion
                    (run_id, role, content, status, created_at, updated_at)
                    VALUES (?, 'user', ?, 'complete', ?, ?)
                    """,
                    (run_id, message, now, now),
                )
                cursor = connection.execute(
                    """
                    INSERT INTO report_discussion
                    (run_id, role, content, status, model, created_at, updated_at)
                    VALUES (?, 'assistant', '', 'generating', ?, ?, ?)
                    """,
                    (run_id, model, now, now),
                )
                return int(cursor.lastrowid)
        except sqlite3.IntegrityError as exc:
            raise RuntimeError("Gemma is already answering this report") from exc

    def finish_discussion(
        self,
        message_id: int,
        *,
        content: str,
        evidence_refs: list[str],
        unresolved_uncertainties: list[str],
        suggested_revision_prompt: str | None,
    ) -> dict:
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE report_discussion
                SET content=?, evidence_refs=?, unresolved_uncertainties=?,
                    suggested_revision_prompt=?, status='complete', updated_at=?
                WHERE id=? AND status='generating'
                """,
                (
                    content,
                    json.dumps(evidence_refs),
                    json.dumps(unresolved_uncertainties),
                    suggested_revision_prompt,
                    utc_now(),
                    message_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError("discussion response not found")
            row = connection.execute(
                "SELECT * FROM report_discussion WHERE id=?", (message_id,)
            ).fetchone()
        assert row is not None
        return self._discussion_row(row)

    def fail_discussion(self, message_id: int) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE report_discussion
                SET content='Gemma could not complete this response', status='failed',
                    updated_at=?
                WHERE id=? AND status='generating'
                """,
                (utc_now(), message_id),
            )

    @staticmethod
    def _event_text(value: str, *, limit: int) -> str:
        cleaned = " ".join(value.replace("\x00", "").split())
        if not cleaned:
            raise ValueError("event text must not be empty")
        return cleaned[:limit]

    def update_run(
        self,
        run_id: str,
        *,
        event_type: str = "run_updated",
        actor: str = "Controller",
        artifact_path: str | None = None,
        **updates,
    ) -> bool:
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
            row = connection.execute(
                "SELECT status, phase, message FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            assert row is not None
            connection.execute(
                """
                INSERT INTO run_events
                (run_id, event_type, status, phase, message, actor,
                 artifact_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._event_text(event_type, limit=80),
                    row["status"],
                    row["phase"],
                    self._event_text(row["message"], limit=2000),
                    self._event_text(actor, limit=80),
                    artifact_path,
                    utc_now(),
                ),
            )
        return True

    def start_run(self, run_id: str) -> bool:
        now = utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status='running', phase='planning',
                    message='Independent plans are being prepared', started_at=?
                WHERE id=? AND status='queued'
                """,
                (now, run_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'run_started', 'running', 'planning',
                            'Independent plans are being prepared', 'Controller', ?)
                    """,
                    (run_id, now),
                )
            return bool(cursor.rowcount)

    def append_event(
        self,
        run_id: str,
        event_type: str,
        actor: str,
        phase: str,
        message: str,
        artifact_path: str | None = None,
    ) -> dict:
        with self._connection() as connection:
            run = connection.execute(
                "SELECT status FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if run is None:
                raise KeyError("run not found")
            cursor = connection.execute(
                """
                INSERT INTO run_events
                (run_id, event_type, status, phase, message, actor,
                 artifact_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._event_text(event_type, limit=80),
                    run["status"],
                    self._event_text(phase, limit=80),
                    self._event_text(message, limit=2000),
                    self._event_text(actor, limit=80),
                    artifact_path,
                    utc_now(),
                ),
            )
            event_id = cursor.lastrowid
            row = connection.execute(
                "SELECT * FROM run_events WHERE id=?", (event_id,)
            ).fetchone()
        assert row is not None
        return dict(row)

    def list_events(self, run_id: str, after_id: int = 0) -> list[dict]:
        self.get_run(run_id)
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM run_events
                WHERE run_id=? AND id>? ORDER BY id ASC LIMIT 500
                """,
                (run_id, max(0, after_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    def request_cancel(self, run_id: str) -> dict:
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError("run not found")
            if row["status"] not in ACTIVE_RUN_STATES:
                return self._decode_run(row)
            if row["status"] != "cancel_requested":
                now = utc_now()
                connection.execute(
                    """
                    UPDATE runs
                    SET status='cancel_requested', phase='canceling',
                        message='Cancellation requested; stopping safely',
                        cancel_requested_at=?
                    WHERE id=?
                    """,
                    (now, run_id),
                )
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'cancel_requested', 'cancel_requested', 'canceling',
                            'Cancellation requested; stopping safely', 'User', ?)
                    """,
                    (run_id, now),
                )
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        assert row is not None
        return self._decode_run(row)

    def finish_run(self, run_id: str, **updates) -> bool:
        status = updates.get("status")
        if status not in RUN_STATES - ACTIVE_RUN_STATES:
            raise ValueError("finish_run requires a terminal status")
        allowed = {
            "status",
            "phase",
            "message",
            "finished_at",
            "provenance_dir",
            "error_type",
        }
        if set(updates) - allowed:
            raise ValueError("invalid run finish update")
        assignments = ", ".join(f"{name}=?" for name in updates)
        with self._connection() as connection:
            cursor = connection.execute(
                f"UPDATE runs SET {assignments} WHERE id=? AND status!='cancel_requested'",  # noqa: S608
                (*updates.values(), run_id),
            )
            if cursor.rowcount:
                row = connection.execute(
                    "SELECT status, phase, message FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                assert row is not None
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'run_finished', ?, ?, ?, 'Controller', ?)
                    """,
                    (run_id, row["status"], row["phase"], row["message"], utc_now()),
                )
            return bool(cursor.rowcount)

    def mark_cancelled(
        self,
        run_id: str,
        *,
        provenance_dir: str | None = None,
    ) -> bool:
        now = utc_now()
        with self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status='cancelled', phase='cancelled',
                    message='Run cancelled; partial artifacts were preserved',
                    finished_at=?, provenance_dir=COALESCE(?, provenance_dir)
                WHERE id=? AND status='cancel_requested'
                """,
                (now, provenance_dir, run_id),
            )
            if cursor.rowcount:
                connection.execute(
                    """
                    INSERT INTO run_events
                    (run_id, event_type, status, phase, message, actor, created_at)
                    VALUES (?, 'run_cancelled', 'cancelled', 'cancelled',
                            'Run cancelled; partial artifacts were preserved',
                            'Controller', ?)
                    """,
                    (run_id, now),
                )
            return bool(cursor.rowcount)

    def get_run(self, run_id: str) -> dict:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError("run not found")
        return self._decode_run(row)

    @staticmethod
    def _decode_run(row: sqlite3.Row) -> dict:
        value = dict(row)
        value["enable_code"] = bool(value["enable_code"])
        value["mcp_servers"] = json.loads(value["mcp_servers"])
        value["knowledge_snapshot"] = json.loads(
            value.get("knowledge_snapshot") or "{}"
        )
        value["requested_outputs"] = json.loads(value.get("requested_outputs") or "[]")
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
            values.append(self._decode_run(row))
        return values

    def run_artifact(self, run_id: str, relative_path: str) -> Path:
        root = self.run_root(run_id)
        candidate = (root / relative_path).resolve()
        if candidate != root and root not in candidate.parents:
            raise KeyError("artifact not found")
        if not candidate.is_file() or candidate.is_symlink():
            raise KeyError("artifact not found")
        return candidate

    def run_root(self, run_id: str) -> Path:
        run = self.get_run(run_id)
        provenance = run.get("provenance_dir")
        if not provenance:
            raise KeyError("run provenance not found")
        root = Path(provenance).resolve()
        _, runs_dir = self.paths(run["workspace_id"])
        if root.parent != runs_dir.resolve() or not root.is_dir():
            raise KeyError("run provenance not found")
        return root
