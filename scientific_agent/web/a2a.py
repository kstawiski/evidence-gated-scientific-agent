"""A2A 1.0 adapter for the standalone scientific agent service."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from a2a.helpers import (
    new_data_part,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    StringList,
)
from a2a.utils.errors import TaskNotCancelableError
from google.protobuf.json_format import MessageToDict

from .. import __version__
from .service import TaskService
from .store import WorkspaceStore


ALLOWED_MCP_SERVERS = {"context7", "brave-search", "chrome-devtools"}
SUCCESS_STATES = {
    "supported",
    "supported_with_comments",
    "contradicted",
    "inconclusive",
    "requires_more_evidence",
    "requires_human_decision",
}


def build_agent_card(public_url: str) -> AgentCard:
    """Describe the stable, generic A2A surface without fleet-specific details."""

    skill = AgentSkill(
        id="evidence-gated-scientific-analysis",
        name="Evidence-gated scientific analysis",
        description=(
            "Plan, research, execute sandboxed Python or R when authorized, "
            "validate claims, and return a provenance-backed scientific report."
        ),
        tags=["science", "analysis", "python", "r", "provenance"],
        examples=[
            "Analyze the attached CSV for group differences and report effect sizes.",
            "Review current evidence for this research question and audit every claim.",
        ],
        input_modes=["text/plain", "application/octet-stream"],
        output_modes=["text/markdown", "application/json"],
        security_requirements=[SecurityRequirement(schemes={"bearer": StringList()})],
    )
    return AgentCard(
        name="Evidence Bench",
        description=(
            "An evidence-gated dual-model scientific agent: Qwen executes, Gemma "
            "criticizes, and deterministic checks arbitrate objective claims."
        ),
        version=__version__,
        supported_interfaces=[
            AgentInterface(
                url=f"{public_url.rstrip('/')}/a2a",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain", "application/octet-stream"],
        default_output_modes=["text/markdown", "application/json"],
        security_schemes={
            "bearer": SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(
                    scheme="bearer",
                    bearer_format="opaque",
                    description="Set Authorization: Bearer <A2A_TOKEN>.",
                )
            )
        },
        skills=[skill],
    )


def _message_metadata(context: RequestContext) -> dict[str, Any]:
    metadata = dict(context.metadata)
    if context.message is not None and context.message.HasField("metadata"):
        metadata.update(MessageToDict(context.message.metadata))
    return metadata


def _run_options(
    context: RequestContext,
    default_mcp_servers: tuple[str, ...],
) -> tuple[bool, tuple[str, ...]]:
    metadata = _message_metadata(context)
    enable_code = metadata.get("enable_code", False)
    if not isinstance(enable_code, bool):
        raise ValueError("metadata.enable_code must be a boolean")
    requested = metadata.get("mcp_servers", list(default_mcp_servers))
    if not isinstance(requested, list) or any(
        not isinstance(item, str) for item in requested
    ):
        raise ValueError("metadata.mcp_servers must be a list of strings")
    unknown = set(requested) - ALLOWED_MCP_SERVERS
    if unknown:
        raise ValueError(f"unsupported MCP servers: {', '.join(sorted(unknown))}")
    return enable_code, tuple(dict.fromkeys(requested))


class EvidenceBenchExecutor(AgentExecutor):
    """Map an A2A context to one persistent, path-confined web workspace."""

    def __init__(
        self,
        store: WorkspaceStore,
        service: TaskService,
        max_upload_bytes: int,
        default_mcp_servers: tuple[str, ...] = (),
    ):
        self.store = store
        self.service = service
        self.max_upload_bytes = max_upload_bytes
        self.default_mcp_servers = default_mcp_servers
        self._task_runs: dict[str, str] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        if context.message is None or not context.task_id or not context.context_id:
            raise ValueError("A2A message, task ID, and context ID are required")
        task = context.current_task
        if task is None:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, task.context_id)

        objective = context.get_user_input().strip()
        if not objective:
            await updater.reject(
                new_text_message("A scientific objective is required.")
            )
            return
        try:
            enable_code, mcp_servers = _run_options(context, self.default_mcp_servers)
            workspace = self.store.get_or_create_external_workspace(
                context.context_id,
                f"A2A {context.context_id[:12]}",
            )
            for part in context.message.parts:
                if part.WhichOneof("content") != "raw":
                    if part.WhichOneof("content") == "url":
                        raise ValueError(
                            "URL file parts are not fetched; send raw bytes"
                        )
                    continue
                if not part.filename:
                    raise ValueError("raw file parts require a filename")
                self.store.save_file(
                    workspace["id"],
                    part.filename,
                    io.BytesIO(part.raw),
                    self.max_upload_bytes,
                )
            await updater.start_work(
                new_text_message("Planning and evidence collection started.")
            )
            run = self.service.submit(
                workspace["id"], objective, enable_code, mcp_servers
            )
            self._task_runs[task.id] = run["id"]
            try:
                detail = await self.service.wait(run["id"])
            finally:
                self._task_runs.pop(task.id, None)
        except (ValueError, FileExistsError, RuntimeError) as exc:
            await updater.reject(new_text_message(str(exc)))
            return

        summary = {
            "run_id": detail["id"],
            "workspace_id": detail["workspace_id"],
            "status": detail["status"],
            "phase": detail["phase"],
            "artifacts": detail["artifacts"],
        }
        await updater.add_artifact(
            [new_data_part(summary, media_type="application/json")],
            name="run-summary.json",
        )
        provenance = detail.get("provenance_dir")
        report_path = Path(provenance) / "report.md" if provenance else None
        if report_path is not None and report_path.is_file():
            await updater.add_artifact(
                [
                    new_text_part(
                        report_path.read_text(encoding="utf-8"),
                        media_type="text/markdown",
                    )
                ],
                name="report.md",
            )
        message = new_text_message(
            json.dumps(summary, sort_keys=True), media_type="application/json"
        )
        if detail["status"] == "cancelled":
            await updater.cancel(message)
        elif detail["status"] in SUCCESS_STATES:
            await updater.complete(message)
        else:
            await updater.failed(message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        if not context.task_id or not context.context_id:
            raise TaskNotCancelableError(message="A2A task context is required.")
        run_id = self._task_runs.get(context.task_id)
        if run_id is None:
            raise TaskNotCancelableError(
                message="The scientific run is not active or is already terminal."
            )
        self.service.cancel(run_id)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel(new_text_message("Scientific run cancellation requested."))
