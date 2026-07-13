"""Background execution service shared by the web API and A2A adapter."""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Awaitable, Callable

from ..config import Settings
from ..orchestrator import run_scientific_task
from ..provenance import utc_now
from ..schemas import RunResult
from .store import ACTIVE_RUN_STATES, WorkspaceStore


Runner = Callable[..., Awaitable[RunResult]]


class TaskService:
    def __init__(
        self,
        store: WorkspaceStore,
        base_settings: Settings,
        max_workers: int,
        runner: Runner = run_scientific_task,
    ):
        self.store = store
        self.base_settings = base_settings
        self.runner = runner
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="evidence-bench"
        )

    def submit(
        self,
        workspace_id: str,
        objective: str,
        enable_code: bool,
        mcp_servers: tuple[str, ...],
    ) -> dict:
        run = self.store.create_run(
            workspace_id, objective, enable_code, mcp_servers
        )
        self.executor.submit(lambda: asyncio.run(self.execute(run["id"])))
        return run

    async def execute(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        if run["status"] not in ACTIVE_RUN_STATES:
            return run
        files_dir, runs_dir = self.store.paths(run["workspace_id"])
        settings = replace(
            self.base_settings,
            workspace=files_dir,
            runs_dir=runs_dir,
        )

        def progress(phase: str, message: str) -> None:
            self.store.update_run(run_id, phase=phase, message=message)

        self.store.update_run(
            run_id,
            status="running",
            phase="planning",
            message="Independent plans are being prepared",
            started_at=utc_now(),
        )
        try:
            result = await self.runner(
                run["objective"],
                settings,
                mcp_names=tuple(run["mcp_servers"]),
                include_chrome="chrome-devtools" in run["mcp_servers"],
                enable_code=run["enable_code"],
                progress=progress,
            )
        except Exception as exc:
            self.store.update_run(
                run_id,
                status="failed",
                phase="failed",
                message="The run stopped before producing a validated result",
                finished_at=utc_now(),
                error_type=type(exc).__name__,
            )
            return self.store.get_run(run_id)
        self.store.update_run(
            run_id,
            status=result.status,
            phase="complete",
            message="Validated result is ready",
            finished_at=utc_now(),
            provenance_dir=result.provenance_dir,
        )
        return self.store.get_run(run_id)

    def detail(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        provenance = run.get("provenance_dir")
        if not provenance:
            return {**run, "result": None, "report": None, "artifacts": []}
        root = Path(provenance)
        result_path = root / "run_result.json"
        report_path = root / "scientific_report.json"
        manifest_path = root / "manifest.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else None
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {"files": []}
        return {
            **run,
            "result": result,
            "report": report,
            "artifacts": manifest.get("files", []),
        }

    async def wait(self, run_id: str, poll_seconds: float = 0.5) -> dict:
        while True:
            run = self.store.get_run(run_id)
            if run["status"] not in ACTIVE_RUN_STATES:
                return self.detail(run_id)
            await asyncio.sleep(poll_seconds)

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
