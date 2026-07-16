"""FastAPI application for Evidence Bench."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import mimetypes
import secrets
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, load_mcp_secrets
from ..discussion import discuss_report
from ..environment import cleanup_workspace_environment
from ..knowledge import (
    DOCUMENT_ID,
    KNOWLEDGE_VISUAL_ID,
    KnowledgeLibrary,
    PASSAGE_ID,
    SOURCE_TYPES,
)
from ..knowledge_indexing import KnowledgeIndexService, KnowledgeSemanticIndexer
from ..orchestrator import run_scientific_task
from ..provenance import sha256_file
from ..reporting import FIGURE_MEDIA_TYPES
from .a2a import ALLOWED_MCP_SERVERS, EvidenceBenchExecutor, build_agent_card
from .integrations import IntegrationArchive, build_a2a_archive, build_skill_archive
from .model_status import (
    ModelStatusMonitor,
    ModelStatusTarget,
    validate_status_base_url,
)
from .service import Runner, TaskService, web_visible_artifact
from .settings import WebSettings
from .store import ACTIVE_RUN_STATES, WorkspaceStore


STATIC_DIR = Path(__file__).with_name("static")
MAX_TEXT_PREVIEW_BYTES = 512 * 1024
PREVIEW_TRUNCATION_MARKER = "\n\n--- PREVIEW TRUNCATED; MIDDLE OMITTED ---\n\n"
logger = logging.getLogger(__name__)


def _decode_truncated_utf8(head: bytes, tail: bytes) -> str:
    """Decode independently cut UTF-8 ranges without accepting interior corruption."""

    try:
        head_text = head.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        if exc.end != len(head) or exc.reason != "unexpected end of data":
            raise
        head_text = head[: exc.start].decode("utf-8-sig")
    tail_start = 0
    while tail_start < len(tail) and tail[tail_start] & 0b1100_0000 == 0b1000_0000:
        tail_start += 1
    tail_text = tail[tail_start:].decode("utf-8")
    return head_text + PREVIEW_TRUNCATION_MARKER + tail_text


def _text_preview(path: Path, display_path: str) -> dict:
    """Return a bounded UTF-8 preview for a path already confined by the store."""

    size = path.stat().st_size
    with path.open("rb") as handle:
        if size <= MAX_TEXT_PREVIEW_BYTES:
            data = handle.read(MAX_TEXT_PREVIEW_BYTES + 1)
            truncated = False
            head = tail = b""
        else:
            head_bytes = MAX_TEXT_PREVIEW_BYTES * 3 // 4
            tail_bytes = MAX_TEXT_PREVIEW_BYTES - head_bytes
            head = handle.read(head_bytes)
            handle.seek(-tail_bytes, 2)
            tail = handle.read(tail_bytes)
            data = head + PREVIEW_TRUNCATION_MARKER.encode("utf-8") + tail
            truncated = True
    if b"\x00" in data:
        raise ValueError("file is not UTF-8 text")
    try:
        content = (
            _decode_truncated_utf8(head, tail)
            if truncated
            else data.decode("utf-8-sig")
        )
    except UnicodeDecodeError as exc:
        raise ValueError("file is not UTF-8 text") from exc
    return {
        "path": display_path,
        "content": content,
        "bytes": size,
        "preview_bytes": len(data),
        "truncated": truncated,
    }


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class RunCreate(BaseModel):
    objective: str = Field(min_length=3, max_length=20_000)
    enable_code: bool = True
    mcp_servers: list[str] | None = Field(default=None, max_length=3)
    knowledge_document_ids: list[str] | None = Field(default=None, max_length=10_000)
    requested_outputs: list[
        Literal["pptx_presentation", "analysis_notebook", "data_bundle"]
    ] = Field(default_factory=list, max_length=3)


class FollowUpCreate(BaseModel):
    request: str = Field(min_length=3, max_length=20_000)
    enable_code: bool | None = None


class DiscussionCreate(BaseModel):
    message: str = Field(min_length=3, max_length=20_000)


class KnowledgeToggle(BaseModel):
    enabled: bool
    etag: int = Field(ge=1)


class KnowledgeEdit(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=4_000)
    tags: list[str] | None = Field(default=None, max_length=32)
    source_type: str | None = None
    canonical_url: str | None = Field(default=None, max_length=2_000)
    etag: int = Field(ge=1)


class KnowledgeSearch(BaseModel):
    query: str = Field(min_length=2, max_length=1_000)
    document_ids: list[str] | None = Field(default=None, max_length=10_000)
    limit: int = Field(default=8, ge=1, le=20)


def _json_error(status: int, message: str, *, challenge: str | None = None):
    headers = {"WWW-Authenticate": challenge} if challenge else None
    return JSONResponse({"detail": message}, status_code=status, headers=headers)


class AuthenticationMiddleware:
    """Keep the Agent Card public while protecting UI/API and A2A execution."""

    def __init__(self, app, settings: WebSettings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path in {"/healthz", "/.well-known/agent-card.json"}:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode(
            "latin-1", errors="ignore"
        )
        if path == "/a2a" or path.startswith("/a2a/"):
            expected = f"Bearer {self.settings.a2a_token}"
            if not secrets.compare_digest(authorization, expected):
                response = _json_error(
                    401,
                    "valid A2A bearer token required",
                    challenge="Bearer",
                )
                await response(scope, receive, send)
                return
        elif self.settings.auth_enabled and not self._valid_basic_auth(authorization):
            response = _json_error(
                401,
                "authentication required",
                challenge='Basic realm="Evidence Bench", charset="UTF-8"',
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _valid_basic_auth(self, authorization: str) -> bool:
        if not authorization.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(authorization[6:], validate=True).decode("utf-8")
            username, password = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError, binascii.Error):
            return False
        return secrets.compare_digest(
            username, self.settings.username
        ) and secrets.compare_digest(password, self.settings.password)


def _validate_mcp_servers(names: list[str]) -> tuple[str, ...]:
    unknown = set(names) - ALLOWED_MCP_SERVERS
    if unknown:
        raise ValueError(f"unsupported MCP servers: {', '.join(sorted(unknown))}")
    return tuple(dict.fromkeys(names))


def _integration_response(archive: IntegrationArchive) -> Response:
    """Serve one fixed archive with an independently verifiable digest."""

    return Response(
        content=archive.content,
        media_type="application/zip",
        headers={
            "Cache-Control": "private, max-age=3600",
            "Content-Disposition": f'attachment; filename="{archive.filename}"',
            "ETag": f'"sha256:{archive.sha256}"',
            "X-Checksum-SHA256": archive.sha256,
        },
    )


def create_app(
    web_settings: WebSettings | None = None,
    agent_settings: Settings | None = None,
    runner: Runner = run_scientific_task,
    discussion_runner=discuss_report,
    model_status_monitor: ModelStatusMonitor | None = None,
    knowledge_semantic_indexer: KnowledgeSemanticIndexer | None = None,
) -> FastAPI:
    web = web_settings or WebSettings()
    scientific_settings = agent_settings or Settings()
    web.validate()
    web.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    try:
        configured_mcp_secrets = load_mcp_secrets()
    except (OSError, ValueError):
        configured_mcp_secrets = {}
    mcp_availability = {
        "context7": bool(configured_mcp_secrets.get("CONTEXT7_API_KEY")),
        "brave-search": bool(configured_mcp_secrets.get("BRAVE_API_KEY")),
        "chrome-devtools": bool(scientific_settings.chrome_browser_url),
    }
    default_mcp_servers = tuple(
        name for name in scientific_settings.mcp_servers if mcp_availability.get(name)
    )
    store = WorkspaceStore(web.database_path, web.workspaces_dir)
    knowledge = KnowledgeLibrary(web.knowledge_dir, web.deployment_id, web.public_url)
    semantic_indexer = knowledge_semantic_indexer or KnowledgeSemanticIndexer(
        scientific_settings
    )
    knowledge_index_service = KnowledgeIndexService(
        knowledge,
        semantic_indexer,
        scientific_work_active=store.has_any_active_run,
    )
    service = TaskService(
        store,
        scientific_settings,
        max_workers=web.max_workers,
        runner=runner,
        knowledge_library=knowledge,
        knowledge_index_service=knowledge_index_service,
        public_url=web.public_url,
    )
    skill_archive = build_skill_archive(__version__)
    a2a_archive = build_a2a_archive(
        __version__, web.public_url, enabled=web.a2a_enabled
    )
    status_monitor = model_status_monitor or ModelStatusMonitor(
        (
            ModelStatusTarget(
                role="executor",
                model=scientific_settings.qwen.model,
                provider="vllm",
                base_url=validate_status_base_url(
                    web.qwen_status_base_url, "QWEN_STATUS_BASE_URL"
                ),
            ),
            ModelStatusTarget(
                role="critic",
                model=scientific_settings.gemma.model,
                provider="llama_cpp",
                base_url=validate_status_base_url(
                    web.gemma_status_base_url, "GEMMA_STATUS_BASE_URL"
                ),
            ),
        ),
        timeout_seconds=web.model_status_timeout_seconds,
        cache_seconds=web.model_status_cache_seconds,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await knowledge_index_service.start()
        try:
            yield
        finally:
            await knowledge_index_service.close()
            service.close()

    app = FastAPI(
        title="Evidence Bench API",
        description="Workspace and A2A API for evidence-gated scientific tasks.",
        version=__version__,
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(AuthenticationMiddleware, settings=web)
    app.state.store = store
    app.state.service = service
    app.state.knowledge = knowledge
    app.state.knowledge_index_service = knowledge_index_service
    app.state.web_settings = web

    @app.middleware("http")
    async def security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; object-src 'none'; "
            f"base-uri 'none'; frame-src 'self' {' '.join(web.browser_frame_sources)}; "
            "frame-ancestors 'none'"
        )
        return response

    @app.exception_handler(KeyError)
    async def key_error_handler(_, exc: KeyError):
        return _json_error(404, str(exc.args[0] if exc.args else "not found"))

    @app.exception_handler(FileExistsError)
    async def exists_handler(_, exc: FileExistsError):
        return _json_error(409, f"file already exists: {exc.args[0]}")

    @app.exception_handler(RuntimeError)
    async def runtime_handler(_, exc: RuntimeError):
        return _json_error(409, str(exc))

    @app.exception_handler(ValueError)
    async def value_handler(_, exc: ValueError):
        status = 413 if "upload limit" in str(exc) else 400
        return _json_error(status, str(exc))

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/api/config")
    async def public_config() -> dict:
        return {
            "version": __version__,
            "models": {
                "executor": scientific_settings.qwen.model,
                "critic": scientific_settings.gemma.model,
            },
            "mcp": mcp_availability,
            "default_mcp_servers": list(default_mcp_servers),
            "a2a": web.a2a_enabled,
            "browser_auth": web.auth_enabled,
            "browser": {
                "enabled": bool(scientific_settings.chrome_browser_url),
                "public_url": web.browser_public_url,
                "novnc_port": web.browser_novnc_port,
            },
            "max_upload_bytes": web.max_upload_bytes,
            "knowledge": knowledge.stats(),
        }

    @app.get("/api/model-status")
    async def model_status() -> dict:
        return await asyncio.to_thread(status_monitor.snapshot)

    @app.get("/api/integrations")
    async def integration_catalog() -> dict:
        return {
            "version": __version__,
            "a2a_enabled": web.a2a_enabled,
            "agent_card_url": "/.well-known/agent-card.json",
            "downloads": [
                {
                    "id": "skill",
                    "label": "Codex / Claude skill",
                    "url": "/api/integrations/skill",
                    "filename": skill_archive.filename,
                    "bytes": len(skill_archive.content),
                    "sha256": skill_archive.sha256,
                    "setup": (
                        "Extract the evidence-bench directory into your agent's "
                        "skills directory, then invoke $evidence-bench."
                    ),
                },
                {
                    "id": "a2a",
                    "label": "A2A 1.0 starter",
                    "url": "/api/integrations/a2a",
                    "filename": a2a_archive.filename,
                    "bytes": len(a2a_archive.content),
                    "sha256": a2a_archive.sha256,
                    "setup": (
                        "Extract the archive, read connection.json, set A2A_TOKEN "
                        "from the lab administrator, and run a2a_client.py."
                    ),
                },
            ],
        }

    @app.get("/api/integrations/skill")
    async def download_skill_integration() -> Response:
        return _integration_response(skill_archive)

    @app.get("/api/integrations/a2a")
    async def download_a2a_integration() -> Response:
        return _integration_response(a2a_archive)

    @app.get("/api/knowledge")
    async def knowledge_catalog(include_retired: bool = Query(default=False)) -> dict:
        return {
            "stats": knowledge.stats(),
            "documents": knowledge.list_documents(include_retired=include_retired),
            "jobs": knowledge.list_index_jobs(limit=100),
        }

    def public_visual(document_id: str, asset: dict) -> dict:
        return {
            "id": asset["id"],
            "sha256": asset["sha256"],
            "source_label": asset["source_label"],
            "preview_url": (
                f"/api/knowledge/{quote(document_id, safe='')}/visuals/"
                f"{quote(asset['id'], safe='')}/preview"
            ),
        }

    def document_index_bundle(document: dict, operation: str) -> dict:
        if document.get("published") and document.get("semantic_status") == "ready":
            return {"document": document, "job": None, "visuals": []}
        if document.get("published"):
            # Never attach a new semantic index to a generation that may already
            # be pinned by a run snapshot. Publish an immutable successor only
            # after its background index passes every precondition.
            document = knowledge.reindex(
                document["id"], document["etag"], semantic_pending=True
            )
        job = knowledge_index_service.enqueue(
            document["id"], operation, document.get("supersedes_id")
        )
        return {
            "document": document,
            "job": job,
            "visuals": [
                public_visual(document["id"], item)
                for item in knowledge.visual_assets(document["id"])
            ],
        }

    @app.post("/api/knowledge", status_code=202)
    async def upload_knowledge(
        upload: UploadFile = File(...),
        title: str = Form(default="", max_length=300),
        description: str = Form(default="", max_length=4_000),
        tags: str = Form(default=""),
        source_type: str = Form(default="other"),
        canonical_url: str = Form(default=""),
    ) -> dict:
        if upload.filename is None:
            raise ValueError("uploaded knowledge requires a filename")
        parsed_tags = [item.strip() for item in tags.split(",") if item.strip()]
        document = await asyncio.to_thread(
            knowledge.ingest,
            upload.filename,
            upload.file,
            web.max_upload_bytes,
            title=title.strip() or Path(upload.filename).stem or upload.filename,
            description=description,
            tags=parsed_tags,
            source_type=source_type,
            canonical_url=canonical_url.strip() or None,
            semantic_pending=True,
        )
        return await asyncio.to_thread(document_index_bundle, document, "upload")

    @app.get("/api/knowledge/jobs")
    async def knowledge_jobs(
        status: str | None = Query(default=None, max_length=40),
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> list[dict]:
        return knowledge.list_index_jobs(status=status, limit=limit)

    @app.get("/api/knowledge/jobs/{job_id}")
    async def knowledge_job(job_id: str) -> dict:
        return knowledge.get_index_job(job_id)

    @app.get("/api/knowledge/jobs/{job_id}/events")
    async def knowledge_job_events(
        job_id: str, after_id: int = Query(default=0, ge=0)
    ) -> list[dict]:
        knowledge.get_index_job(job_id)
        return knowledge.list_index_events(job_id, after_id=after_id)

    @app.post("/api/knowledge/jobs/{job_id}/cancel")
    async def cancel_knowledge_job(job_id: str) -> dict:
        return knowledge_index_service.request_cancel(job_id)

    @app.post("/api/knowledge/jobs/{job_id}/retry")
    async def retry_knowledge_job(job_id: str) -> dict:
        return knowledge_index_service.retry(job_id)

    @app.post("/api/knowledge/search")
    async def test_knowledge_search(request: KnowledgeSearch) -> dict:
        snapshot = knowledge.snapshot(request.document_ids)
        return await asyncio.to_thread(
            knowledge.search, request.query, snapshot, request.limit
        )

    @app.post("/api/knowledge/search/visuals")
    async def search_knowledge_visuals(request: KnowledgeSearch) -> dict:
        snapshot = knowledge.snapshot(request.document_ids)
        result = await asyncio.to_thread(
            knowledge.search_visuals, request.query, snapshot, request.limit
        )
        return {
            **result,
            "visuals": [
                {
                    **{key: value for key, value in item.items() if key != "path"},
                    **public_visual(item["document_id"], {"id": item["visual_id"], **item}),
                }
                for item in result["visuals"]
            ],
        }

    @app.post("/api/knowledge/reindex-all", status_code=202)
    async def reindex_all_knowledge() -> dict:
        documents = await asyncio.to_thread(
            knowledge.reindex_all, enabled_only=True, semantic_pending=True
        )
        items = [
            await asyncio.to_thread(document_index_bundle, document, "reindex")
            for document in documents
        ]
        return {
            "documents": [item["document"] for item in items],
            "jobs": [item["job"] for item in items],
            "items": items,
            "count": len(items),
        }

    @app.get("/api/knowledge/{document_id}")
    async def get_knowledge(document_id: str) -> dict:
        return knowledge.get_document(document_id)

    @app.patch("/api/knowledge/{document_id}")
    async def edit_knowledge(document_id: str, request: KnowledgeEdit) -> dict:
        if request.source_type is not None and request.source_type not in SOURCE_TYPES:
            raise ValueError("invalid knowledge source type")
        document = await asyncio.to_thread(
            knowledge.retire_and_clone,
            document_id,
            title=request.title,
            description=request.description,
            tags=request.tags,
            source_type=request.source_type,
            canonical_url=request.canonical_url,
            etag=request.etag,
            semantic_pending=True,
        )
        return await asyncio.to_thread(
            document_index_bundle, document, "metadata_revision"
        )

    @app.patch("/api/knowledge/{document_id}/enabled")
    async def toggle_knowledge(document_id: str, request: KnowledgeToggle) -> dict:
        return knowledge.update_enabled(document_id, request.enabled, request.etag)

    @app.delete("/api/knowledge/{document_id}", status_code=204)
    async def delete_knowledge(
        document_id: str, etag: int = Query(..., ge=1)
    ) -> Response:
        knowledge.delete(document_id, etag)
        return Response(status_code=204)

    @app.post("/api/knowledge/{document_id}/reindex", status_code=202)
    async def reindex_knowledge(document_id: str, etag: int = Query(..., ge=1)) -> dict:
        document = await asyncio.to_thread(
            knowledge.reindex, document_id, etag, semantic_pending=True
        )
        return await asyncio.to_thread(document_index_bundle, document, "reindex")

    @app.get("/api/knowledge/{document_id}/download")
    async def download_knowledge(document_id: str) -> FileResponse:
        document = knowledge.get_document(document_id)
        path = knowledge.source_path(document_id)
        return FileResponse(path, filename=document["filename"])

    @app.get("/api/knowledge/{document_id}/preview")
    async def preview_knowledge(document_id: str) -> dict:
        path = knowledge.extracted_path(document_id)
        return _text_preview(path, f"knowledge/{document_id}/extracted.txt")

    @app.get("/api/knowledge/{document_id}/chunks")
    async def knowledge_chunks(
        document_id: str, limit: int = Query(default=200, ge=1, le=1_000)
    ) -> list[dict]:
        return knowledge.chunks(document_id, limit)

    @app.get("/api/knowledge/{document_id}/acquisitions")
    async def knowledge_acquisitions(document_id: str) -> list[dict]:
        return knowledge.acquisition_history(document_id)

    @app.get("/api/knowledge/{document_id}/visuals")
    async def knowledge_visuals(document_id: str) -> list[dict]:
        return [
            public_visual(document_id, item)
            for item in knowledge.visual_assets(document_id)
        ]

    @app.get("/api/knowledge/{document_id}/visuals/{visual_id}/preview")
    async def preview_knowledge_visual(document_id: str, visual_id: str) -> FileResponse:
        assets = knowledge.visual_assets(document_id, visual_id=visual_id)
        if not assets:
            raise KeyError("knowledge visual not found")
        path = Path(assets[0]["path"])
        return FileResponse(
            path,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            headers={"Content-Disposition": "inline"},
        )

    @app.get("/api/workspaces")
    async def list_workspaces() -> list[dict]:
        return store.list_workspaces()

    @app.post("/api/workspaces", status_code=201)
    async def create_workspace(request: WorkspaceCreate) -> dict:
        return store.create_workspace(request.name)

    @app.get("/api/workspaces/{workspace_id}")
    async def get_workspace(workspace_id: str) -> dict:
        return {
            **store.get_workspace(workspace_id),
            "files": store.list_files(workspace_id),
            "runs": store.list_runs(workspace_id),
        }

    @app.delete("/api/workspaces/{workspace_id}", status_code=204)
    async def delete_workspace(workspace_id: str) -> Response:
        store.delete_workspace(
            workspace_id,
            before_delete=lambda: cleanup_workspace_environment(
                scientific_settings.environment,
                workspace_id,
            ),
        )
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/files", status_code=201)
    async def upload_file(
        workspace_id: str,
        upload: UploadFile = File(...),
        overwrite: bool = Query(default=False),
    ) -> dict:
        if upload.filename is None:
            raise ValueError("uploaded file requires a filename")
        return store.save_file(
            workspace_id,
            upload.filename,
            upload.file,
            web.max_upload_bytes,
            overwrite=overwrite,
        )

    @app.put("/api/workspaces/{workspace_id}/files/{filename}", status_code=201)
    async def upload_file_streamed(
        workspace_id: str,
        filename: str,
        request: Request,
        overwrite: bool = Query(default=False),
    ) -> dict:
        content_length = request.headers.get("content-length")
        if content_length is not None:
            try:
                announced = int(content_length)
            except ValueError as exc:
                raise ValueError("invalid Content-Length") from exc
            if announced < 0 or announced > web.max_upload_bytes:
                raise ValueError(
                    f"file exceeds {web.max_upload_bytes} byte upload limit"
                )
        return await store.save_streamed_file(
            workspace_id,
            filename,
            request.stream(),
            web.max_upload_bytes,
            overwrite=overwrite,
        )

    @app.get("/api/workspaces/{workspace_id}/files/{filename}")
    async def download_file(workspace_id: str, filename: str) -> FileResponse:
        path = store.file_path(workspace_id, filename)
        return FileResponse(path, filename=path.name)

    @app.get("/api/workspaces/{workspace_id}/file-preview")
    async def preview_workspace_file(
        workspace_id: str, filename: str = Query(...)
    ) -> dict:
        path = store.file_path(workspace_id, filename)
        return _text_preview(path, filename)

    @app.delete("/api/workspaces/{workspace_id}/files/{filename}", status_code=204)
    async def delete_file(workspace_id: str, filename: str) -> Response:
        store.delete_file(workspace_id, filename)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/runs", status_code=202)
    async def create_run(workspace_id: str, request: RunCreate) -> dict:
        if request.requested_outputs and not request.enable_code:
            raise ValueError(
                "requested PPTX/notebook/data artifacts require Python/R execution"
            )
        selected = _validate_mcp_servers(
            request.mcp_servers
            if request.mcp_servers is not None
            else list(default_mcp_servers)
        )
        return service.submit(
            workspace_id,
            request.objective,
            request.enable_code,
            selected,
            knowledge_document_ids=request.knowledge_document_ids,
            requested_outputs=tuple(dict.fromkeys(request.requested_outputs)),
        )

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return service.detail(run_id)

    @app.get("/api/runs/{run_id}/knowledge/passages/{passage_id}")
    async def run_knowledge_passage(run_id: str, passage_id: str) -> FileResponse:
        if not PASSAGE_ID.fullmatch(passage_id):
            raise KeyError("knowledge passage not found")
        root = store.run_root(run_id)
        path = (root / "knowledge" / "passages" / f"{passage_id}.md").resolve()
        expected_parent = (root / "knowledge" / "passages").resolve()
        evidence_path = root / "retrieval_evidence.json"
        if (
            path.parent != expected_parent
            or not path.is_file()
            or path.is_symlink()
            or not evidence_path.is_file()
        ):
            raise KeyError("knowledge passage not found")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        record = next(
            (
                item
                for item in evidence.get("knowledge_passages", [])
                if item.get("passage_id") == passage_id
            ),
            None,
        )
        if record is None or sha256_file(path) != record.get("artifact_sha256"):
            raise KeyError("knowledge passage integrity check failed")
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'inline; filename="{passage_id}.md"',
                "Cache-Control": "private, immutable, max-age=31536000",
            },
        )

    @app.get("/api/runs/{run_id}/knowledge/visuals/{knowledge_visual_id}")
    async def run_knowledge_visual(
        run_id: str, knowledge_visual_id: str
    ) -> FileResponse:
        if not KNOWLEDGE_VISUAL_ID.fullmatch(knowledge_visual_id):
            raise KeyError("knowledge visual not found")
        root = store.run_root(run_id).resolve()
        evidence_path = root / "retrieval_evidence.json"
        if not evidence_path.is_file() or evidence_path.is_symlink():
            raise KeyError("knowledge visual not found")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        record = next(
            (
                item
                for item in evidence.get("knowledge_visuals", [])
                if item.get("knowledge_visual_id") == knowledge_visual_id
            ),
            None,
        )
        if not isinstance(record, dict):
            raise KeyError("knowledge visual not found")
        raw_path = record.get("artifact_path")
        if not isinstance(raw_path, str):
            raise KeyError("knowledge visual integrity check failed")
        unresolved = Path(raw_path)
        expected_parent = (root / "knowledge" / "visuals").resolve()
        try:
            path = unresolved.resolve()
        except (OSError, RuntimeError) as exc:
            raise KeyError("knowledge visual integrity check failed") from exc
        if (
            not unresolved.is_absolute()
            or unresolved.is_symlink()
            or not unresolved.is_file()
            or path.parent != expected_parent
            or not path.name.startswith(f"{knowledge_visual_id}.")
            or path.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp"}
            or record.get("visual_sha256") != record.get("artifact_sha256")
            or sha256_file(path) != record.get("artifact_sha256")
        ):
            raise KeyError("knowledge visual integrity check failed")
        return FileResponse(
            path,
            media_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            headers={
                "Content-Disposition": (
                    f"inline; filename*=UTF-8''{quote(path.name)}"
                ),
                "Cache-Control": "private, immutable, max-age=31536000",
            },
        )

    def registered_run_knowledge_document(
        run_id: str, document_id: str, kind: str
    ) -> tuple[Path, dict]:
        if not DOCUMENT_ID.fullmatch(document_id):
            raise KeyError("knowledge document not found")
        root = store.run_root(run_id).resolve()
        evidence_path = root / "retrieval_evidence.json"
        if not evidence_path.is_file():
            raise KeyError("knowledge document not found")
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        record = next(
            (
                item
                for item in evidence.get("knowledge_passages", [])
                if item.get("document_id") == document_id
            ),
            None,
        )
        if not isinstance(record, dict):
            raise KeyError("knowledge document not found")
        path_field = (
            "document_text_path" if kind == "text" else "document_original_path"
        )
        hash_field = (
            "document_text_sha256" if kind == "text" else "document_original_sha256"
        )
        raw_path = record.get(path_field)
        if not isinstance(raw_path, str):
            raise KeyError("knowledge document not found")
        unresolved = Path(raw_path)
        if (
            not unresolved.is_absolute()
            or ".." in unresolved.parts
            or root not in unresolved.parents
        ):
            raise KeyError("knowledge document integrity check failed")
        relative = unresolved.relative_to(root)
        if any(
            (root.joinpath(*relative.parts[:index])).is_symlink()
            for index in range(1, len(relative.parts) + 1)
        ):
            raise KeyError("knowledge document integrity check failed")
        path = unresolved.resolve()
        if (
            root not in path.parents
            or not path.is_file()
            or sha256_file(path) != record.get(hash_field)
        ):
            raise KeyError("knowledge document integrity check failed")
        return path, record

    @app.get("/api/runs/{run_id}/knowledge/documents/{document_id}/text")
    async def run_knowledge_document_text(
        run_id: str, document_id: str
    ) -> FileResponse:
        path, _ = registered_run_knowledge_document(run_id, document_id, "text")
        return FileResponse(
            path,
            media_type="text/markdown; charset=utf-8",
            headers={
                "Content-Disposition": f'inline; filename="{document_id}.md"',
                "Cache-Control": "private, immutable, max-age=31536000",
            },
        )

    @app.get("/api/runs/{run_id}/knowledge/documents/{document_id}/original")
    async def run_knowledge_document_original(
        run_id: str, document_id: str
    ) -> FileResponse:
        path, record = registered_run_knowledge_document(
            run_id, document_id, "original"
        )
        filename = Path(str(record.get("document_filename") or path.name)).name
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        disposition = "inline" if path.suffix.casefold() == ".pdf" else "attachment"
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Content-Disposition": (
                    f"{disposition}; filename*=UTF-8''{quote(filename)}"
                ),
                "Cache-Control": "private, immutable, max-age=31536000",
            },
        )

    @app.get("/api/runs/{run_id}/events")
    async def get_run_events(
        run_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> list[dict]:
        return store.list_events(run_id, after_id)

    @app.get("/api/runs/{run_id}/events/stream")
    async def stream_run_events(
        request: Request,
        run_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> StreamingResponse:
        store.get_run(run_id)

        async def event_stream():
            cursor = after_id
            heartbeat_ticks = 0
            while True:
                if await request.is_disconnected():
                    return
                events = store.list_events(run_id, cursor)
                for event in events:
                    cursor = max(cursor, int(event["id"]))
                    yield (
                        f"id: {event['id']}\n"
                        "event: run_event\n"
                        f"data: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                    )
                if len(events) >= 500:
                    continue
                run = store.get_run(run_id)
                if run["status"] not in ACTIVE_RUN_STATES:
                    yield (
                        "event: stream_end\n"
                        f"data: {json.dumps({'status': run['status']}, separators=(',', ':'))}\n\n"
                    )
                    return
                heartbeat_ticks += 1
                if heartbeat_ticks >= 30:
                    heartbeat_ticks = 0
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str) -> dict:
        return service.cancel(run_id)

    @app.post("/api/runs/{run_id}/follow-ups", status_code=202)
    async def follow_up(run_id: str, request: FollowUpCreate) -> dict:
        return service.submit_follow_up(
            run_id,
            request.request,
            enable_code=request.enable_code,
        )

    @app.get("/api/runs/{run_id}/discussion")
    async def get_discussion(run_id: str) -> list[dict]:
        return store.list_discussion(run_id)

    @app.post("/api/runs/{run_id}/discussion", status_code=201)
    async def create_discussion(run_id: str, request: DiscussionCreate) -> dict:
        run = service.detail(run_id)
        if run["status"] in ACTIVE_RUN_STATES:
            raise RuntimeError("the report is not complete")
        if run.get("report") is None or not run.get("provenance_dir"):
            raise RuntimeError("the run has no scientific report to discuss")
        history = store.list_discussion(run_id)
        response_id = store.start_discussion(
            run_id, request.message, scientific_settings.gemma.model
        )
        try:
            response = await discussion_runner(
                scientific_settings,
                Path(run["provenance_dir"]),
                history,
                request.message,
            )
            return store.finish_discussion(
                response_id,
                content=response.answer,
                evidence_refs=response.evidence_refs,
                unresolved_uncertainties=response.unresolved_uncertainties,
                suggested_revision_prompt=response.suggested_revision_prompt,
            )
        except Exception as exc:
            logger.warning("Report discussion failed safely (%s)", type(exc).__name__)
            store.fail_discussion(response_id)
            raise RuntimeError("Gemma could not complete the report discussion")

    @app.get("/api/runs/{run_id}/artifacts")
    async def download_artifact(run_id: str, path: str = Query(...)) -> FileResponse:
        if not web_visible_artifact(path):
            raise KeyError("artifact not available in the Web explorer")
        artifact = store.run_artifact(run_id, path)
        media_type = mimetypes.guess_type(artifact.name)[0]
        return FileResponse(artifact, filename=artifact.name, media_type=media_type)

    @app.get("/api/runs/{run_id}/artifact-preview")
    async def preview_text_artifact(run_id: str, path: str = Query(...)) -> dict:
        if not web_visible_artifact(path):
            raise KeyError("artifact not available in the Web explorer")
        artifact = store.run_artifact(run_id, path)
        return _text_preview(artifact, path)

    def registered_reference(run_id: str, source_id: str, kind: str) -> Path:
        root = store.run_root(run_id)
        manifest_path = root / "reference_manifest.json"
        if not manifest_path.is_file():
            raise KeyError("reference not found")
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (
                item
                for item in manifest.get("references", [])
                if item.get("source_id") == source_id
            ),
            None,
        )
        artifact = entry.get(kind) if isinstance(entry, dict) else None
        if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
            raise KeyError("reference not found")
        path = store.run_artifact(run_id, artifact["path"])
        if sha256_file(path) != artifact.get("sha256"):
            raise KeyError("reference integrity check failed")
        return path

    @app.get("/api/runs/{run_id}/references/{source_id}/pdf")
    async def reference_pdf(run_id: str, source_id: str) -> FileResponse:
        path = registered_reference(run_id, source_id, "pdf")
        if path.suffix.lower() != ".pdf" or path.stat().st_size < 5:
            raise KeyError("reference not found")
        with path.open("rb") as handle:
            if handle.read(5) != b"%PDF-":
                raise KeyError("reference integrity check failed")
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'inline; filename="{path.name}"',
                "Cache-Control": "private, max-age=3600",
            },
        )

    def registered_display(
        run_id: str, display_id: str, kind: str
    ) -> tuple[Path, dict]:
        root = store.run_root(run_id)
        manifest_path = root / "display_manifest.json"
        if not manifest_path.is_file():
            raise KeyError("display not found")
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (
                item
                for item in manifest.get("displays", [])
                if item.get("display_id") == display_id and item.get("kind") == kind
            ),
            None,
        )
        if entry is None:
            raise KeyError("display not found")
        path = store.run_artifact(run_id, str(entry.get("path", "")))
        if sha256_file(path) != entry.get("sha256"):
            raise KeyError("display integrity check failed")
        return path, entry

    @app.get("/api/runs/{run_id}/displays/{display_id}/image")
    async def display_image(run_id: str, display_id: str) -> FileResponse:
        path, _ = registered_display(run_id, display_id, "figure")
        media_type = FIGURE_MEDIA_TYPES.get(path.suffix.lower())
        if media_type is None:
            raise KeyError("display not found")
        return FileResponse(
            path,
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="{path.name}"',
                "Cache-Control": "private, max-age=3600",
            },
        )

    @app.get("/api/runs/{run_id}/displays/{display_id}/table")
    async def display_table(run_id: str, display_id: str) -> dict:
        _, entry = registered_display(run_id, display_id, "table")
        return {
            key: entry[key]
            for key in (
                "display_id",
                "number",
                "title",
                "caption",
                "columns",
                "rows",
                "total_rows",
                "total_columns",
                "truncated",
                "claim_ids",
                "evidence_refs",
            )
            if key in entry
        }

    @app.get("/api/runs/{run_id}/bundle")
    async def download_bundle(run_id: str, background: BackgroundTasks) -> FileResponse:
        root = store.run_root(run_id)
        with tempfile.NamedTemporaryFile(
            prefix=f"evidence-bench-{run_id[:8]}-", suffix=".zip", delete=False
        ) as handle:
            archive_path = Path(handle.name)
        try:
            with zipfile.ZipFile(
                archive_path, mode="w", compression=zipfile.ZIP_DEFLATED
            ) as archive:
                for path in sorted(root.rglob("*")):
                    if path.is_file() and not path.is_symlink():
                        archive.write(path, path.relative_to(root))
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise
        background.add_task(archive_path.unlink, missing_ok=True)
        return FileResponse(
            archive_path,
            filename=f"evidence-bench-{run_id}.zip",
            media_type="application/zip",
            background=background,
        )

    if web.a2a_enabled:
        card = build_agent_card(web.public_url)
        executor = EvidenceBenchExecutor(
            store,
            service,
            web.max_upload_bytes,
            default_mcp_servers=default_mcp_servers,
        )
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            agent_card=card,
        )
        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=create_agent_card_routes(card),
            jsonrpc_routes=create_jsonrpc_routes(request_handler, "/a2a"),
        )

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def main() -> None:
    settings = WebSettings()
    uvicorn.run(
        create_app(web_settings=settings),
        host=settings.host,
        port=settings.port,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
    )


if __name__ == "__main__":
    main()
