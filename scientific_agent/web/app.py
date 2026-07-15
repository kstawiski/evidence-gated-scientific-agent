"""FastAPI application for Evidence Bench."""

from __future__ import annotations

import base64
import binascii
import logging
import mimetypes
import secrets
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    add_a2a_routes_to_fastapi,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.background import BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..config import Settings, load_mcp_secrets
from ..discussion import discuss_report
from ..environment import cleanup_workspace_environment
from ..orchestrator import run_scientific_task
from ..provenance import sha256_file
from ..reporting import FIGURE_MEDIA_TYPES
from .a2a import ALLOWED_MCP_SERVERS, EvidenceBenchExecutor, build_agent_card
from .service import Runner, TaskService
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


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class RunCreate(BaseModel):
    objective: str = Field(min_length=3, max_length=20_000)
    enable_code: bool = True
    mcp_servers: list[str] | None = Field(default=None, max_length=3)


class FollowUpCreate(BaseModel):
    request: str = Field(min_length=3, max_length=20_000)
    enable_code: bool | None = None


class DiscussionCreate(BaseModel):
    message: str = Field(min_length=3, max_length=20_000)


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


def create_app(
    web_settings: WebSettings | None = None,
    agent_settings: Settings | None = None,
    runner: Runner = run_scientific_task,
    discussion_runner=discuss_report,
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
    service = TaskService(
        store,
        scientific_settings,
        max_workers=web.max_workers,
        runner=runner,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
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
        }

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

    @app.get("/api/workspaces/{workspace_id}/files/{filename}")
    async def download_file(workspace_id: str, filename: str) -> FileResponse:
        path = store.file_path(workspace_id, filename)
        return FileResponse(path, filename=path.name)

    @app.delete("/api/workspaces/{workspace_id}/files/{filename}", status_code=204)
    async def delete_file(workspace_id: str, filename: str) -> Response:
        store.delete_file(workspace_id, filename)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/runs", status_code=202)
    async def create_run(workspace_id: str, request: RunCreate) -> dict:
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
        )

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return service.detail(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def get_run_events(
        run_id: str,
        after_id: int = Query(default=0, ge=0),
    ) -> list[dict]:
        return store.list_events(run_id, after_id)

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
        artifact = store.run_artifact(run_id, path)
        media_type = mimetypes.guess_type(artifact.name)[0]
        return FileResponse(artifact, filename=artifact.name, media_type=media_type)

    @app.get("/api/runs/{run_id}/artifact-preview")
    async def preview_text_artifact(run_id: str, path: str = Query(...)) -> dict:
        artifact = store.run_artifact(run_id, path)
        size = artifact.stat().st_size
        with artifact.open("rb") as handle:
            if size <= MAX_TEXT_PREVIEW_BYTES:
                data = handle.read(MAX_TEXT_PREVIEW_BYTES + 1)
                truncated = False
            else:
                head_bytes = MAX_TEXT_PREVIEW_BYTES * 3 // 4
                tail_bytes = MAX_TEXT_PREVIEW_BYTES - head_bytes
                head = handle.read(head_bytes)
                handle.seek(-tail_bytes, 2)
                tail = handle.read(tail_bytes)
                marker = PREVIEW_TRUNCATION_MARKER.encode("utf-8")
                data = head + marker + tail
                truncated = True
        if b"\x00" in data:
            raise ValueError("artifact is not UTF-8 text")
        try:
            content = (
                _decode_truncated_utf8(head, tail)
                if truncated
                else data.decode("utf-8-sig")
            )
        except UnicodeDecodeError as exc:
            raise ValueError("artifact is not UTF-8 text") from exc
        return {
            "path": path,
            "content": content,
            "bytes": size,
            "preview_bytes": len(data),
            "truncated": truncated,
        }

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
