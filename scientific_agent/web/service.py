"""Background execution service shared by the web API and A2A adapter."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

from ..config import Settings
from ..knowledge import KnowledgeLibrary
from ..orchestrator import run_scientific_task
from ..provenance import build_manifest, utc_now, write_json
from ..reporting import describe_available_displays
from ..schemas import ComputationEvidence, RunResult, ScientificReport
from .store import ACTIVE_RUN_STATES, WorkspaceStore

if TYPE_CHECKING:
    from ..knowledge_indexing import KnowledgeIndexService


logger = logging.getLogger(__name__)
Runner = Callable[..., Awaitable[RunResult]]
ACTORS = {
    "input-intake": "Controller + Gemma",
    "planning": "Qwen + Gemma",
    "plan-review": "Qwen + Gemma",
    "research": "Qwen",
    "reporting": "Qwen",
    "validation": "Controller",
    "scientific-review": "Gemma",
    "repair": "Qwen",
    "finalizing": "Controller",
    "complete": "Controller",
    "stopped": "Controller",
}
WEB_HIDDEN_ROOT_FILES = {"tool_call_log.jsonl"}
WEB_HIDDEN_DIRECTORIES = {"evidence"}


def web_visible_artifact(relative_path: str) -> bool:
    """Expose produced artifacts except raw tool arguments and response evidence."""

    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        return False
    if len(path.parts) == 1 and path.name in WEB_HIDDEN_ROOT_FILES:
        return False
    return path.parts[0] not in WEB_HIDDEN_DIRECTORIES


class TaskService:
    def __init__(
        self,
        store: WorkspaceStore,
        base_settings: Settings,
        max_workers: int,
        runner: Runner = run_scientific_task,
        knowledge_library: KnowledgeLibrary | None = None,
        knowledge_index_service: "KnowledgeIndexService | None" = None,
        public_url: str = "",
    ):
        self.store = store
        self.base_settings = base_settings
        self.runner = runner
        self.knowledge_library = knowledge_library
        self.knowledge_index_service = knowledge_index_service
        self.public_url = public_url.rstrip("/")
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="evidence-bench"
        )
        self._lock = threading.Lock()
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._async_tasks: dict[
            str, tuple[asyncio.AbstractEventLoop, asyncio.Task]
        ] = {}

    def submit(
        self,
        workspace_id: str,
        objective: str,
        enable_code: bool,
        mcp_servers: tuple[str, ...],
        *,
        parent_run_id: str | None = None,
        run_kind: str = "analysis",
        knowledge_document_ids: list[str] | None = None,
        knowledge_snapshot: dict | None = None,
        requested_outputs: tuple[str, ...] = (),
    ) -> dict:
        if knowledge_snapshot is None and self.knowledge_library is not None:
            knowledge_snapshot = self.knowledge_library.snapshot(knowledge_document_ids)
        run = self.store.create_run(
            workspace_id,
            objective,
            enable_code,
            mcp_servers,
            parent_run_id=parent_run_id,
            run_kind=run_kind,
            knowledge_snapshot=knowledge_snapshot,
            requested_outputs=requested_outputs,
        )
        cancellation = threading.Event()
        with self._lock:
            self._cancel_events[run["id"]] = cancellation
            future = self.executor.submit(
                lambda: asyncio.run(self.execute(run["id"], cancellation))
            )
            self._futures[run["id"]] = future
        future.add_done_callback(lambda _: self._forget(run["id"]))
        return run

    def _forget(self, run_id: str) -> None:
        with self._lock:
            self._futures.pop(run_id, None)
            self._cancel_events.pop(run_id, None)
            self._async_tasks.pop(run_id, None)

    def submit_follow_up(
        self,
        parent_run_id: str,
        request: str,
        *,
        enable_code: bool | None = None,
    ) -> dict:
        parent = self.store.get_run(parent_run_id)
        request = request.strip()
        if not 3 <= len(request) <= 20_000:
            raise ValueError("follow-up request must contain 3-20,000 characters")
        provenance = parent.get("provenance_dir")
        if (
            not provenance
            or not (Path(provenance) / "scientific_report.json").is_file()
        ):
            raise RuntimeError("the parent run has no scientific report to revise")
        return self.submit(
            parent["workspace_id"],
            request,
            parent["enable_code"] if enable_code is None else enable_code,
            tuple(parent["mcp_servers"]),
            parent_run_id=parent_run_id,
            run_kind="revision",
            knowledge_snapshot=parent.get("knowledge_snapshot") or {},
            requested_outputs=tuple(parent.get("requested_outputs") or ()),
        )

    async def execute(
        self,
        run_id: str,
        cancellation: threading.Event | None = None,
    ) -> dict:
        cancellation = cancellation or threading.Event()
        run = self.store.get_run(run_id)
        if run["status"] not in ACTIVE_RUN_STATES:
            return run
        if not self.store.start_run(run_id):
            if self.store.get_run(run_id)["status"] == "cancel_requested":
                self.store.mark_cancelled(run_id)
            return self.store.get_run(run_id)

        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        if task is not None:
            with self._lock:
                self._async_tasks[run_id] = (loop, task)

        files_dir, runs_dir = self.store.paths(run["workspace_id"])
        settings = replace(
            self.base_settings,
            workspace=files_dir,
            runs_dir=runs_dir,
            knowledge_root=(
                self.knowledge_library.root if self.knowledge_library else None
            ),
            knowledge_deployment_id=(
                self.knowledge_library.deployment_id
                if self.knowledge_library
                else self.base_settings.knowledge_deployment_id
            ),
            knowledge_snapshot=run.get("knowledge_snapshot") or None,
            knowledge_citation_base_url=(
                f"{self.public_url}/api/runs/{run_id}/knowledge/passages"
                if self.public_url and run.get("knowledge_snapshot")
                else ""
            ),
        )
        provenance_holder: dict[str, Path] = {}

        def provenance_ready(path: Path) -> None:
            resolved = path.resolve()
            if resolved.parent != runs_dir.resolve():
                raise ValueError("run provenance escaped the workspace")
            provenance_holder["path"] = resolved
            self.store.update_run(
                run_id,
                provenance_dir=str(resolved),
                event_type="provenance_ready",
                actor="Controller",
            )

        def progress(phase: str, message: str) -> None:
            current = self.store.get_run(run_id)
            if current["status"] != "running" or cancellation.is_set():
                return
            self.store.update_run(
                run_id,
                phase=phase,
                message=message,
                event_type="phase_changed",
                actor=ACTORS.get(phase, "Controller"),
            )

        def activity(
            event_type: str,
            actor: str,
            phase: str,
            message: str,
            artifact_path: str | None,
        ) -> None:
            relative: str | None = None
            if artifact_path is not None:
                root = provenance_holder.get("path")
                candidate = Path(artifact_path).resolve()
                if root is None or (
                    candidate != root and root not in candidate.parents
                ):
                    return
                relative = candidate.relative_to(root).as_posix()
            self.store.append_event(
                run_id,
                event_type,
                actor,
                phase,
                message,
                relative,
            )

        runner_kwargs = {
            "mcp_names": tuple(run["mcp_servers"]),
            "include_chrome": "chrome-devtools" in run["mcp_servers"],
            "enable_code": run["enable_code"],
            "progress": progress,
            "on_provenance_ready": provenance_ready,
            "activity": activity,
            "cancel_event": cancellation,
            # The shared lab service uses the asymmetric evidence gate by default:
            # one Qwen plan, a fixed five-criterion Gemma plan audit, and the
            # independent final report/display audits. The CLI retains explicit
            # full dual-plan mode for exceptional investigations.
            "simple_mode": True,
            "requested_outputs": tuple(run.get("requested_outputs") or ()),
        }
        if run.get("parent_run_id"):
            parent = self.store.get_run(run["parent_run_id"])
            runner_kwargs.update(
                {
                    "parent_provenance_dir": Path(parent["provenance_dir"]),
                    "revision_request": run["objective"],
                }
            )

        existing_run_dirs = {
            path.resolve() for path in runs_dir.iterdir() if path.is_dir()
        }
        try:
            result = await self.runner(run["objective"], settings, **runner_kwargs)
        except asyncio.CancelledError:
            cancellation.set()
            partial = self._partial_root(
                runs_dir, existing_run_dirs, provenance_holder.get("path")
            )
            self._preserve_cancelled(partial)
            self.store.mark_cancelled(
                run_id, provenance_dir=str(partial) if partial is not None else None
            )
            return self.store.get_run(run_id)
        except Exception as exc:
            if (
                cancellation.is_set()
                or self.store.get_run(run_id)["status"] == "cancel_requested"
            ):
                partial = self._partial_root(
                    runs_dir, existing_run_dirs, provenance_holder.get("path")
                )
                self._preserve_cancelled(partial)
                self.store.mark_cancelled(
                    run_id, provenance_dir=str(partial) if partial is not None else None
                )
                return self.store.get_run(run_id)
            partial = self._partial_root(
                runs_dir, existing_run_dirs, provenance_holder.get("path")
            )
            provenance_dir = None
            if partial is not None:
                try:
                    write_json(
                        partial / "run_failure.json",
                        {"error_type": type(exc).__name__, "failed_at": utc_now()},
                    )
                    build_manifest(partial)
                    provenance_dir = str(partial)
                except Exception:
                    provenance_dir = None
            self.store.finish_run(
                run_id,
                status="failed",
                phase="failed",
                message="The run stopped before producing a validated result",
                finished_at=utc_now(),
                error_type=type(exc).__name__,
                provenance_dir=provenance_dir,
            )
            return self.store.get_run(run_id)
        finally:
            with self._lock:
                self._async_tasks.pop(run_id, None)

        completion_messages = {
            "supported": "Validated result is ready",
            "supported_with_comments": "Validated result is ready with nonblocking comments",
            "contradicted": "Completed: the evidence contradicts a material claim",
            "inconclusive": "Completed: the available evidence is inconclusive",
            "requires_more_evidence": "Completed with unresolved evidence requirements",
            "requires_human_decision": "Completed and awaiting a human scientific decision",
        }
        if self.knowledge_library is not None and not cancellation.is_set():
            try:
                self.store.update_run(
                    run_id,
                    phase="finalizing",
                    message="Registering controller-verified literature",
                    event_type="phase_changed",
                    actor="Controller",
                )
                imported = self.knowledge_library.import_verified_run_articles(
                    Path(result.provenance_dir),
                    workspace_id=run["workspace_id"],
                    run_id=run_id,
                    semantic_pending=self.knowledge_index_service is not None,
                )
                new_documents = [
                    item for item in imported if not item.get("deduplicated")
                ]
                if imported:
                    self.store.append_event(
                        run_id,
                        "knowledge_import",
                        "Controller",
                        "finalizing",
                        (
                            f"Recorded {len(imported)} verified article acquisition(s); "
                            f"{len(new_documents)} created a new knowledge generation"
                        ),
                        None,
                    )
                if self.knowledge_index_service is not None:
                    for document in new_documents:
                        self.knowledge_index_service.enqueue(
                            document["id"], "verified_run_article"
                        )
            except Exception:
                # Knowledge promotion is a post-analysis convenience. It must never
                # change or invalidate the completed scientific record.
                logger.exception("verified article knowledge import failed")
        committed = self.store.finish_run(
            run_id,
            status=result.status,
            phase="complete",
            message=completion_messages.get(result.status, "Run completed"),
            finished_at=utc_now(),
            provenance_dir=result.provenance_dir,
        )
        if not committed:
            cancellation.set()
            partial = Path(result.provenance_dir)
            self._preserve_cancelled(partial)
            self.store.mark_cancelled(run_id, provenance_dir=str(partial))
        return self.store.get_run(run_id)

    @staticmethod
    def _partial_root(
        runs_dir: Path,
        existing: set[Path],
        known: Path | None,
    ) -> Path | None:
        if known is not None and known.is_dir():
            return known
        candidates = [
            path.resolve()
            for path in runs_dir.iterdir()
            if path.is_dir() and path.resolve() not in existing
        ]
        return (
            max(candidates, key=lambda path: path.stat().st_mtime_ns)
            if candidates
            else None
        )

    @staticmethod
    def _preserve_cancelled(partial: Path | None) -> None:
        if partial is None:
            return
        try:
            write_json(
                partial / "run_cancelled.json",
                {
                    "cancelled_at": utc_now(),
                    "status": "cancelled",
                    "note": "Partial artifacts are incomplete and are not validated evidence.",
                },
            )
            build_manifest(partial)
        except Exception:
            pass

    def cancel(self, run_id: str) -> dict:
        run = self.store.request_cancel(run_id)
        if run["status"] != "cancel_requested":
            return run
        with self._lock:
            cancellation = self._cancel_events.get(run_id)
            future = self._futures.get(run_id)
            async_task = self._async_tasks.get(run_id)
        if cancellation is not None:
            cancellation.set()
        if future is not None and future.cancel():
            self.store.mark_cancelled(run_id)
            return self.store.get_run(run_id)
        if async_task is not None:
            loop, task = async_task
            loop.call_soon_threadsafe(task.cancel)
        return self.store.get_run(run_id)

    def detail(self, run_id: str) -> dict:
        run = self.store.get_run(run_id)
        provenance = run.get("provenance_dir")
        if not provenance:
            return {
                **run,
                "result": None,
                "report": None,
                "display_manifest": None,
                "reference_manifest": None,
                "artifacts": [],
            }
        root = Path(provenance)
        result_path = root / "run_result.json"
        report_path = root / "scientific_report.json"
        manifest_path = root / "manifest.json"
        displays_path = root / "display_manifest.json"
        references_path = root / "reference_manifest.json"
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.is_file()
            else None
        )
        report = (
            json.loads(report_path.read_text(encoding="utf-8"))
            if report_path.is_file()
            else None
        )
        display_manifest = (
            json.loads(displays_path.read_text(encoding="utf-8"))
            if displays_path.is_file()
            else None
        )
        if display_manifest is None and report is not None:
            computation_path = root / "computation_evidence.json"
            try:
                computation = ComputationEvidence.model_validate_json(
                    computation_path.read_text(encoding="utf-8")
                )
                report_model = ScientificReport.model_validate(report)
                deterministic_passed = bool(
                    result
                    and result.get("deterministic_validation", {}).get("passed") is True
                )
                review_passed = bool(
                    result
                    and result.get("scientific_review", {}).get("verdict") == "pass"
                )
                display_manifest = describe_available_displays(
                    report_model,
                    computation,
                    validated=deterministic_passed and review_passed,
                    quality_status=run["status"],
                )
            except (OSError, ValueError, json.JSONDecodeError):
                logger.warning(
                    "Could not describe historical displays for run %s", run_id
                )
        reference_manifest = (
            json.loads(references_path.read_text(encoding="utf-8"))
            if references_path.is_file()
            else None
        )
        if manifest_path.is_file() and run["status"] not in ACTIVE_RUN_STATES:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifacts = [
                artifact
                for artifact in manifest.get("files", [])
                if isinstance(artifact, dict)
                and isinstance(artifact.get("path"), str)
                and web_visible_artifact(artifact["path"])
            ]
        else:
            artifacts = self._live_artifacts(root)
        return {
            **run,
            "result": result,
            "report": report,
            "display_manifest": display_manifest,
            "reference_manifest": reference_manifest,
            "artifacts": artifacts,
        }

    @staticmethod
    def _live_artifacts(root: Path) -> list[dict]:
        if not root.is_dir():
            return []
        artifacts = []
        for path in sorted(root.rglob("*")):
            try:
                is_file = path.is_file()
                is_symlink = path.is_symlink()
                size = path.stat().st_size if is_file and not is_symlink else None
            except OSError:
                # Sandbox workers tighten permissions while atomically staging an
                # execution record. A concurrent Web/UI poll must remain available;
                # the temporarily inaccessible artifact will appear on the next poll.
                continue
            if not is_file or is_symlink:
                continue
            relative = path.relative_to(root).as_posix()
            if not web_visible_artifact(relative):
                continue
            artifacts.append(
                {
                    "path": relative,
                    "bytes": size,
                    "sha256": None,
                    "live": True,
                }
            )
        return artifacts[:2000]

    async def wait(self, run_id: str, poll_seconds: float = 0.5) -> dict:
        while True:
            run = self.store.get_run(run_id)
            if run["status"] not in ACTIVE_RUN_STATES:
                return self.detail(run_id)
            await asyncio.sleep(poll_seconds)

    def close(self) -> None:
        with self._lock:
            cancellations = list(self._cancel_events.values())
        for cancellation in cancellations:
            cancellation.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
