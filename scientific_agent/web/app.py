"""FastAPI application for Evidence Bench."""

from __future__ import annotations

import base64
import binascii
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
from ..orchestrator import run_scientific_task
from .a2a import ALLOWED_MCP_SERVERS, EvidenceBenchExecutor, build_agent_card
from .service import Runner, TaskService
from .settings import WebSettings
from .store import WorkspaceStore


STATIC_DIR = Path(__file__).with_name("static")


class WorkspaceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class RunCreate(BaseModel):
    objective: str = Field(min_length=3, max_length=20_000)
    enable_code: bool = True
    mcp_servers: list[str] = Field(default_factory=list, max_length=3)


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
        elif self.settings.auth_enabled and not self._valid_basic_auth(
            authorization
        ):
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
            decoded = base64.b64decode(
                authorization[6:], validate=True
            ).decode("utf-8")
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
) -> FastAPI:
    web = web_settings or WebSettings()
    web.validate()
    web.data_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
    store = WorkspaceStore(web.database_path, web.workspaces_dir)
    service = TaskService(
        store,
        agent_settings or Settings(),
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
        settings = agent_settings or Settings()
        try:
            mcp_secrets = load_mcp_secrets()
        except (OSError, ValueError):
            mcp_secrets = {}
        return {
            "version": __version__,
            "models": {
                "executor": settings.qwen.model,
                "critic": settings.gemma.model,
            },
            "mcp": {
                "context7": bool(mcp_secrets.get("CONTEXT7_API_KEY")),
                "brave-search": bool(mcp_secrets.get("BRAVE_API_KEY")),
                "chrome-devtools": bool(settings.chrome_browser_url),
            },
            "a2a": web.a2a_enabled,
            "browser_auth": web.auth_enabled,
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
        store.delete_workspace(workspace_id)
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

    @app.delete(
        "/api/workspaces/{workspace_id}/files/{filename}", status_code=204
    )
    async def delete_file(workspace_id: str, filename: str) -> Response:
        store.delete_file(workspace_id, filename)
        return Response(status_code=204)

    @app.post("/api/workspaces/{workspace_id}/runs", status_code=202)
    async def create_run(workspace_id: str, request: RunCreate) -> dict:
        selected = _validate_mcp_servers(request.mcp_servers)
        return service.submit(
            workspace_id,
            request.objective,
            request.enable_code,
            selected,
        )

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return service.detail(run_id)

    @app.get("/api/runs/{run_id}/artifacts")
    async def download_artifact(run_id: str, path: str = Query(...)) -> FileResponse:
        artifact = store.run_artifact(run_id, path)
        media_type = mimetypes.guess_type(artifact.name)[0]
        return FileResponse(artifact, filename=artifact.name, media_type=media_type)

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
        executor = EvidenceBenchExecutor(store, service, web.max_upload_bytes)
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
