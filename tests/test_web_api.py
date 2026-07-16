import asyncio
import json
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from PIL import Image

import scientific_agent.web.app as web_app_module
from scientific_agent.config import Settings
from scientific_agent.provenance import sha256_file
from scientific_agent.schemas import ReportDiscussionResponse
from scientific_agent.web.a2a import _run_options
from scientific_agent.web.app import create_app
from scientific_agent.web.settings import WebSettings


async def fake_runner(objective, settings, *, progress, **kwargs):
    del objective, kwargs
    progress("research", "Collecting test evidence")
    root = settings.runs_dir / "fake-provenance"
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "title": "Validated test report",
        "executive_summary": "A deterministic fixture result.",
        "introduction": "The fixture tests the web reporting path.",
        "methods": ["Fixture method"],
        "results": "The fixture completed successfully.",
        "discussion": "The fixture is limited to integration behavior.",
        "conclusions": "The web path returns a structured report.",
        "displays": [],
        "claims": [
            {
                "claim_id": "C1",
                "text": "The fixture completed.",
                "claim_type": "computed",
                "evidence_refs": ["report.md"],
                "status": "supported",
                "limitations": [],
            }
        ],
        "sources": [
            {
                "source_id": "S1",
                "title": "Fixture article",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                "source_type": "primary_study",
                "retrieved_at": "2026-07-16T00:00:00Z",
                "pmid": "123",
                "rights_status": "full_text_available",
            }
        ],
        "unresolved_issues": [],
        "limitations": [],
        "narrative": "Fixture narrative.",
    }
    (root / "report.md").write_text("# Validated test report\n", encoding="utf-8")
    (root / "analysis_notes").write_text(
        "extensionless UTF-8 artifact\n", encoding="utf-8"
    )
    (root / "binary.dat").write_bytes(b"\x00\xff\x00")
    (root / "tool_call_log.jsonl").write_text(
        '{"arguments":{"token":"private"}}\n', encoding="utf-8"
    )
    raw_evidence = root / "evidence" / "raw-tool-result.json"
    raw_evidence.parent.mkdir(parents=True, exist_ok=True)
    raw_evidence.write_text('{"private":"result"}\n', encoding="utf-8")
    large_utf8 = root / "large-utf8.md"
    size = web_app_module.MAX_TEXT_PREVIEW_BYTES + 1_000
    head_end = web_app_module.MAX_TEXT_PREVIEW_BYTES * 3 // 4
    tail_start = size - web_app_module.MAX_TEXT_PREVIEW_BYTES // 4
    large_payload = bytearray(b"a" * size)
    large_payload[head_end - 1 : head_end + 1] = "µ".encode("utf-8")
    large_payload[tail_start - 1 : tail_start + 1] = "β".encode("utf-8")
    large_utf8.write_bytes(large_payload)
    reference_pdf = root / "references" / "pdfs" / "fixture-2026-pmid123.pdf"
    reference_markdown = root / "references" / "markdown" / "fixture-2026-pmid123.md"
    reference_pdf.parent.mkdir(parents=True, exist_ok=True)
    reference_markdown.parent.mkdir(parents=True, exist_ok=True)
    reference_pdf.write_bytes(b"%PDF-1.4\nfixture\n%%EOF\n")
    reference_markdown.write_text("# Fixture article\n\nPMID: 123\n", encoding="utf-8")
    (root / "scientific_report.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "deterministic_validation.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    (root / "run_result.json").write_text(
        json.dumps({"status": "supported"}), encoding="utf-8"
    )
    (root / "reference_manifest.json").write_text(
        json.dumps(
            {
                "references": [
                    {
                        "source_id": "S1",
                        "title": "Fixture article",
                        "canonical_url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                        "pmid": "123",
                        "full_text_status": "full_text_with_pdf",
                        "pdf": {
                            "path": "references/pdfs/fixture-2026-pmid123.pdf",
                            "bytes": reference_pdf.stat().st_size,
                            "sha256": sha256_file(reference_pdf),
                        },
                        "markdown": {
                            "path": "references/markdown/fixture-2026-pmid123.md",
                            "bytes": reference_markdown.stat().st_size,
                            "sha256": sha256_file(reference_markdown),
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "report.md", "bytes": 24, "sha256": "a" * 64},
                    {"path": "analysis_notes", "bytes": 28, "sha256": "b" * 64},
                    {"path": "binary.dat", "bytes": 3, "sha256": "c" * 64},
                    {
                        "path": "tool_call_log.jsonl",
                        "bytes": (root / "tool_call_log.jsonl").stat().st_size,
                        "sha256": sha256_file(root / "tool_call_log.jsonl"),
                    },
                    {
                        "path": "evidence/raw-tool-result.json",
                        "bytes": raw_evidence.stat().st_size,
                        "sha256": sha256_file(raw_evidence),
                    },
                    {
                        "path": "large-utf8.md",
                        "bytes": large_utf8.stat().st_size,
                        "sha256": sha256_file(large_utf8),
                    },
                    {
                        "path": "references/pdfs/fixture-2026-pmid123.pdf",
                        "bytes": reference_pdf.stat().st_size,
                        "sha256": sha256_file(reference_pdf),
                    },
                    {
                        "path": "references/markdown/fixture-2026-pmid123.md",
                        "bytes": reference_markdown.stat().st_size,
                        "sha256": sha256_file(reference_markdown),
                    },
                    {
                        "path": "deterministic_validation.json",
                        "bytes": (root / "deterministic_validation.json")
                        .stat()
                        .st_size,
                        "sha256": sha256_file(root / "deterministic_validation.json"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(status="supported", provenance_dir=str(root))


async def unresolved_runner(objective, settings, *, progress, **kwargs):
    result = await fake_runner(objective, settings, progress=progress, **kwargs)
    return SimpleNamespace(
        status="requires_more_evidence", provenance_dir=result.provenance_dir
    )


async def fake_discussion_runner(settings, run_root, history, message):
    del settings, run_root
    return ReportDiscussionResponse(
        answer=f"The fixture answer addresses: {message}",
        evidence_refs=["C1", "report.md"],
        unresolved_uncertainties=["The fixture has no external validation."],
        suggested_revision_prompt=(
            "Clarify the external-validity limitation, preserve C1, and update only "
            "claims changed by a falsifiable validation check."
        ),
    )


def _client(
    tmp_path: Path,
    runner=fake_runner,
    discussion_runner=fake_discussion_runner,
) -> TestClient:
    web = WebSettings(
        data_dir=tmp_path,
        username="researcher",
        password="correct horse",
        a2a_token="a2a-secret",
        public_url="https://agent.example.test",
        max_workers=1,
    )
    app = create_app(
        web,
        Settings(),
        runner=runner,
        discussion_runner=discussion_runner,
    )
    return TestClient(app)


def _wait_for_run(client, run_id):
    for _ in range(100):
        response = client.get(
            f"/api/runs/{run_id}", auth=("researcher", "correct horse")
        )
        if response.json()["status"] not in {"queued", "running", "cancel_requested"}:
            return response.json()
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_ui_api_auth_workspace_upload_and_run(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/").status_code == 401
        assert client.get("/", auth=("researcher", "correct horse")).status_code == 200
        assert (
            "object-src 'none'"
            in client.get("/", auth=("researcher", "correct horse")).headers[
                "content-security-policy"
            ]
        )

        created = client.post(
            "/api/workspaces",
            auth=("researcher", "correct horse"),
            json={"name": "Trial analysis"},
        )
        assert created.status_code == 201
        workspace_id = created.json()["id"]
        uploaded = client.post(
            f"/api/workspaces/{workspace_id}/files",
            auth=("researcher", "correct horse"),
            files={"upload": ("values.csv", b"x,y\n1,2\n", "text/csv")},
        )
        assert uploaded.status_code == 201
        streamed = client.put(
            f"/api/workspaces/{workspace_id}/files/large%20input.bin",
            auth=("researcher", "correct horse"),
            content=b"streamed-without-multipart-spooling",
            headers={"Content-Type": "application/octet-stream"},
        )
        assert streamed.status_code == 201
        assert streamed.json() == {"name": "large input.bin", "bytes": 35}
        input_preview = client.get(
            f"/api/workspaces/{workspace_id}/file-preview",
            params={"filename": "values.csv"},
            auth=("researcher", "correct horse"),
        )
        assert input_preview.status_code == 200
        assert input_preview.json()["content"] == "x,y\n1,2\n"
        escaped_preview = client.get(
            f"/api/workspaces/{workspace_id}/file-preview",
            params={"filename": "../values.csv"},
            auth=("researcher", "correct horse"),
        )
        assert escaped_preview.status_code == 400

        queued = client.post(
            f"/api/workspaces/{workspace_id}/runs",
            auth=("researcher", "correct horse"),
            json={
                "objective": "Analyze the uploaded values",
                "enable_code": True,
                "mcp_servers": [],
                "requested_outputs": ["pptx_presentation"],
            },
        )
        assert queued.status_code == 202
        result = _wait_for_run(client, queued.json()["id"])
        assert result["status"] == "supported"
        assert result["requested_outputs"] == ["pptx_presentation"]
        assert result["report"]["title"] == "Validated test report"
        assert result["reference_manifest"]["references"][0]["pmid"] == "123"
        imported = client.get(
            "/api/knowledge", auth=("researcher", "correct horse")
        ).json()["documents"]
        assert len(imported) == 1
        assert imported[0]["origin_type"] == "verified_run_article"
        assert imported[0]["origin_run_id"] == result["id"]
        assert imported[0]["pmid"] == "123"
        acquisitions = client.get(
            f"/api/knowledge/{imported[0]['id']}/acquisitions",
            auth=("researcher", "correct horse"),
        ).json()
        assert acquisitions[0]["run_id"] == result["id"]
        visible_paths = {artifact["path"] for artifact in result["artifacts"]}
        assert "tool_call_log.jsonl" not in visible_paths
        assert "evidence/raw-tool-result.json" not in visible_paths
        event_stream = client.get(
            f"/api/runs/{result['id']}/events/stream",
            auth=("researcher", "correct horse"),
        )
        assert event_stream.status_code == 200
        assert event_stream.headers["content-type"].startswith("text/event-stream")
        assert "event: run_event" in event_stream.text
        assert "event: stream_end" in event_stream.text
        assert '"status":"supported"' in event_stream.text
        paper = client.get(
            f"/api/runs/{result['id']}/references/S1/pdf",
            auth=("researcher", "correct horse"),
        )
        assert paper.status_code == 200
        assert paper.headers["content-type"].startswith("application/pdf")
        assert paper.headers["content-disposition"].startswith("inline;")
        assert paper.content.startswith(b"%PDF-")
        artifact = client.get(
            f"/api/runs/{result['id']}/artifacts",
            params={"path": "report.md"},
            auth=("researcher", "correct horse"),
        )
        assert artifact.status_code == 200
        assert artifact.text.startswith("# Validated")
        preview = client.get(
            f"/api/runs/{result['id']}/artifact-preview",
            params={"path": "report.md"},
            auth=("researcher", "correct horse"),
        )
        assert preview.status_code == 200
        assert preview.json() == {
            "path": "report.md",
            "content": "# Validated test report\n",
            "bytes": 24,
            "preview_bytes": 24,
            "truncated": False,
        }
        for hidden_path in ("tool_call_log.jsonl", "evidence/raw-tool-result.json"):
            hidden_preview = client.get(
                f"/api/runs/{result['id']}/artifact-preview",
                params={"path": hidden_path},
                auth=("researcher", "correct horse"),
            )
            hidden_download = client.get(
                f"/api/runs/{result['id']}/artifacts",
                params={"path": hidden_path},
                auth=("researcher", "correct horse"),
            )
            assert hidden_preview.status_code == 404
            assert hidden_download.status_code == 404
        extensionless_preview = client.get(
            f"/api/runs/{result['id']}/artifact-preview",
            params={"path": "analysis_notes"},
            auth=("researcher", "correct horse"),
        )
        assert extensionless_preview.status_code == 200
        assert (
            extensionless_preview.json()["content"] == "extensionless UTF-8 artifact\n"
        )
        binary_preview = client.get(
            f"/api/runs/{result['id']}/artifact-preview",
            params={"path": "binary.dat"},
            auth=("researcher", "correct horse"),
        )
        assert binary_preview.status_code == 400
        large_preview = client.get(
            f"/api/runs/{result['id']}/artifact-preview",
            params={"path": "large-utf8.md"},
            auth=("researcher", "correct horse"),
        )
        assert large_preview.status_code == 200
        assert large_preview.json()["truncated"] is True
        assert "PREVIEW TRUNCATED" in large_preview.json()["content"]
        assert "�" not in large_preview.json()["content"]
        bundle = client.get(
            f"/api/runs/{result['id']}/bundle",
            auth=("researcher", "correct horse"),
        )
        assert bundle.status_code == 200
        with zipfile.ZipFile(BytesIO(bundle.content)) as archive:
            assert "report.md" in archive.namelist()
            assert archive.read("report.md").startswith(b"# Validated")


def test_terminal_sse_stream_drains_more_than_one_event_page(tmp_path):
    with _client(tmp_path) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Long event history"}
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Produce a long event history", "mcp_servers": []},
        ).json()
        run = _wait_for_run(client, queued["id"])
        store = client.app.state.store
        for index in range(520):
            store.append_event(
                run["id"],
                "test_event",
                "Controller",
                "complete",
                f"event-{index}",
            )

        streamed = client.get(
            f"/api/runs/{run['id']}/events/stream",
            auth=auth,
        )

        assert streamed.status_code == 200
        assert streamed.text.count("event: run_event") >= 520
        assert "event-519" in streamed.text
        assert "event: stream_end" in streamed.text


def test_completed_report_supports_persistent_gemma_discussion_and_revision_prompt(
    tmp_path,
):
    with _client(tmp_path) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Gemma discussion"}
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Produce a report to explain", "mcp_servers": []},
        ).json()
        run = _wait_for_run(client, queued["id"])

        assert client.get(f"/api/runs/{run['id']}/discussion", auth=auth).json() == []
        answer = client.post(
            f"/api/runs/{run['id']}/discussion",
            auth=auth,
            json={"message": "Explain the primary result"},
        )

        assert answer.status_code == 201
        assert answer.json()["role"] == "assistant"
        assert answer.json()["status"] == "complete"
        assert answer.json()["evidence_refs"] == ["C1", "report.md"]
        assert (
            "falsifiable validation check" in answer.json()["suggested_revision_prompt"]
        )
        messages = client.get(f"/api/runs/{run['id']}/discussion", auth=auth).json()
        assert [item["role"] for item in messages] == ["user", "assistant"]
        assert messages[0]["content"] == "Explain the primary result"
        assert messages[1]["model"] == Settings().gemma.model


def test_gemma_discussion_failure_is_recorded_without_exposing_model_error(tmp_path):
    async def failed_discussion(*args):
        del args
        raise RuntimeError("private upstream detail")

    with _client(tmp_path, discussion_runner=failed_discussion) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Failed discussion"}
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Produce a report to discuss", "mcp_servers": []},
        ).json()
        run = _wait_for_run(client, queued["id"])

        response = client.post(
            f"/api/runs/{run['id']}/discussion",
            auth=auth,
            json={"message": "Explain this report"},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Gemma could not complete the report discussion"
        )
        messages = client.get(f"/api/runs/{run['id']}/discussion", auth=auth).json()
        assert messages[-1]["status"] == "failed"
        assert "private upstream detail" not in json.dumps(messages)


def test_web_and_a2a_service_use_asymmetric_bounded_planning_by_default(tmp_path):
    observed = []

    async def mode_runner(objective, settings, *, progress, **kwargs):
        observed.append(kwargs.get("simple_mode"))
        return await fake_runner(objective, settings, progress=progress, **kwargs)

    with _client(tmp_path, runner=mode_runner) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Bounded planning"}
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Audit the planning route", "mcp_servers": []},
        ).json()
        _wait_for_run(client, queued["id"])

    assert observed == [True]


def test_browser_ui_uses_structured_dom_without_inner_html():
    script = Path("scientific_agent/web/static/app.js").read_text(encoding="utf-8")
    page = Path("scientific_agent/web/static/index.html").read_text(encoding="utf-8")
    style = Path("scientific_agent/web/static/app.css").read_text(encoding="utf-8")
    assert "innerHTML" not in script
    assert 'data-tab="article"' in page
    assert 'id="cancel-run-button"' in page
    assert 'id="model-output-monitor"' in page
    assert 'id="artifact-preview-dialog"' in page
    assert 'id="model-output-expand"' in page
    assert 'id="activity-stream-state"' in page
    assert "new EventSource" in script
    assert "openWorkspaceFilePreview" in script
    assert 'data-tab="discussion"' in page
    assert "/discussion" in script
    assert 'id="active-run-capabilities"' in page
    assert "reflectActiveProtocol" in script
    assert "appendCitationMarker" in script
    assert 'document.createElement(local?.markdown ? "button" : "a")' in script
    assert 'request.open("PUT"' in script
    assert 'id="requested-output-options"' in page
    assert 'id="knowledge-index-summary"' in page
    assert 'id="knowledge-jobs"' in page
    assert 'id="knowledge-visual-gallery"' in page
    assert 'id="knowledge-visual-preview-dialog"' in page
    assert 'id="knowledge-upload-file" type="file" multiple' in page
    assert 'api("/api/knowledge/search/visuals"' in script
    assert "/events`)" in script
    assert "scheduleKnowledgePolling" in script
    assert "Qwen text → Gemma actual images only" in script
    assert 'for (const [index, file] of files.entries())' in script
    assert ".run-facts dd { white-space: normal; overflow-wrap: anywhere; }" in style
    assert ".task-panel.is-locked .mcp-options input:disabled + span" in style


def test_unresolved_run_is_not_labeled_as_validated(tmp_path):
    with _client(tmp_path, runner=unresolved_runner) as client:
        workspace = client.post(
            "/api/workspaces",
            auth=("researcher", "correct horse"),
            json={"name": "Unresolved result"},
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=("researcher", "correct horse"),
            json={
                "objective": "Find evidence gaps",
                "enable_code": False,
                "mcp_servers": [],
            },
        ).json()

        result = _wait_for_run(client, queued["id"])

        assert result["status"] == "requires_more_evidence"
        assert result["message"] == "Completed with unresolved evidence requirements"


def test_browser_auth_can_be_disabled_without_disabling_a2a_auth(tmp_path):
    web = WebSettings(
        data_dir=tmp_path,
        auth_enabled=False,
        username="",
        password="",
        a2a_token="a2a-secret",
        public_url="https://agent.example.test",
        max_workers=1,
    )
    with TestClient(create_app(web, Settings(), runner=fake_runner)) as client:
        assert client.get("/").status_code == 200
        created = client.post(
            "/api/workspaces", json={"name": "Passwordless internal workspace"}
        )
        assert created.status_code == 201
        assert client.get("/api/config").json()["browser_auth"] is False
        assert client.post("/a2a", json={}).status_code == 401


def test_web_omission_enables_all_available_mcps_and_empty_list_opts_out(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CONTEXT7_API_KEY", "test-context7-key")
    monkeypatch.setenv("BRAVE_API_KEY", "test-brave-key")

    with _client(tmp_path) as client:
        auth = ("researcher", "correct horse")
        config = client.get("/api/config", auth=auth).json()
        assert config["mcp"] == {
            "context7": True,
            "brave-search": True,
            "chrome-devtools": True,
        }
        assert config["default_mcp_servers"] == [
            "context7",
            "brave-search",
            "chrome-devtools",
        ]

        workspace = client.post(
            "/api/workspaces",
            auth=auth,
            json={"name": "Default MCP policy"},
        ).json()
        default_run = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Use the relevant research connections"},
        ).json()
        default_detail = _wait_for_run(client, default_run["id"])
        assert default_detail["mcp_servers"] == [
            "context7",
            "brave-search",
            "chrome-devtools",
        ]

        opt_out_run = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={
                "objective": "Run without external research connections",
                "mcp_servers": [],
            },
        ).json()
        opt_out_detail = _wait_for_run(client, opt_out_run["id"])
        assert opt_out_detail["mcp_servers"] == []


def test_unavailable_mcp_is_not_selected_as_a_web_default(tmp_path, monkeypatch):
    monkeypatch.setattr(
        web_app_module,
        "load_mcp_secrets",
        lambda: {"CONTEXT7_API_KEY": "test-context7-key"},
    )

    with _client(tmp_path) as client:
        config = client.get("/api/config", auth=("researcher", "correct horse")).json()

    assert config["mcp"] == {
        "context7": True,
        "brave-search": False,
        "chrome-devtools": True,
    }
    assert config["default_mcp_servers"] == ["context7", "chrome-devtools"]


def test_a2a_omission_uses_service_defaults_and_empty_list_opts_out():
    defaults = ("context7", "brave-search", "chrome-devtools")

    assert _run_options(SimpleNamespace(metadata={}, message=None), defaults) == (
        False,
        defaults,
        (),
    )
    assert _run_options(
        SimpleNamespace(metadata={"mcp_servers": []}, message=None), defaults
    ) == (False, (), ())
    assert _run_options(
        SimpleNamespace(
            metadata={
                "enable_code": True,
                "requested_outputs": ["pptx_presentation", "data_bundle"],
            },
            message=None,
        ),
        defaults,
    ) == (True, defaults, ("pptx_presentation", "data_bundle"))


def test_browser_auth_flag_loads_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIENTIFIC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_AUTH_ENABLED", "false")
    monkeypatch.setenv("WEB_USERNAME", "")
    monkeypatch.setenv("WEB_PASSWORD", "")
    monkeypatch.setenv("A2A_TOKEN", "a2a-secret")

    settings = WebSettings()
    settings.validate()
    assert settings.auth_enabled is False
    assert settings.max_upload_bytes == 4 * 1024**3


def test_a2a_card_and_jsonrpc_execution(tmp_path):
    with _client(tmp_path) as client:
        card = client.get("/.well-known/agent-card.json")
        assert card.status_code == 200
        assert card.json()["supportedInterfaces"] == [
            {
                "url": "https://agent.example.test/a2a",
                "protocolBinding": "JSONRPC",
                "protocolVersion": "1.0",
            }
        ]
        denied = client.post("/a2a", json={})
        assert denied.status_code == 401

        response = client.post(
            "/a2a",
            headers={"Authorization": "Bearer a2a-secret", "A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "integration-test",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "message-1",
                        "role": "ROLE_USER",
                        "parts": [{"text": "Analyze this fixture scientifically"}],
                        "metadata": {"enable_code": True, "mcp_servers": []},
                    },
                    "configuration": {"returnImmediately": False},
                },
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert "error" not in payload, payload
        task = payload["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_COMPLETED"

        retrieved = client.post(
            "/a2a",
            headers={"Authorization": "Bearer a2a-secret", "A2A-Version": "1.0"},
            json={
                "jsonrpc": "2.0",
                "id": "get-task-test",
                "method": "GetTask",
                "params": {"id": task["id"]},
            },
        )
        assert retrieved.status_code == 200, retrieved.text
        retrieved_payload = retrieved.json()
        assert "error" not in retrieved_payload, retrieved_payload
        assert retrieved_payload["result"]["id"] == task["id"]
        assert retrieved_payload["result"]["status"]["state"] == "TASK_STATE_COMPLETED"
        artifact_names = {
            artifact["name"] for artifact in retrieved_payload["result"]["artifacts"]
        }
        assert artifact_names == {
            "run-summary.json",
            "report.md",
        }


def test_a2a_streaming_emits_status_artifacts_and_completion(tmp_path):
    with _client(tmp_path) as client:
        with client.stream(
            "POST",
            "/a2a",
            headers={
                "Authorization": "Bearer a2a-secret",
                "A2A-Version": "1.0",
                "Accept": "text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": "stream-test",
                "method": "SendStreamingMessage",
                "params": {
                    "message": {
                        "messageId": "stream-message",
                        "role": "ROLE_USER",
                        "parts": [{"text": "Analyze this fixture scientifically"}],
                        "metadata": {
                            "enable_code": False,
                            "mcp_servers": [],
                        },
                    },
                    "configuration": {},
                },
            },
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = [
                json.loads(line.removeprefix("data: "))
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        states = []
        artifacts = set()
        for event in events:
            result = event["result"]
            if "task" in result:
                states.append(result["task"]["status"]["state"])
            if "statusUpdate" in result:
                states.append(result["statusUpdate"]["status"]["state"])
            if "artifactUpdate" in result:
                artifacts.add(result["artifactUpdate"]["artifact"]["name"])

        assert states == [
            "TASK_STATE_SUBMITTED",
            "TASK_STATE_WORKING",
            "TASK_STATE_COMPLETED",
        ]
        assert artifacts == {"run-summary.json", "report.md"}


def test_a2a_cancel_task_stops_an_active_scientific_run(tmp_path):
    started = threading.Event()

    async def slow_runner(objective, settings, *, on_provenance_ready, **kwargs):
        del objective, kwargs
        root = settings.runs_dir / "a2a-cancel-provenance"
        root.mkdir()
        on_provenance_ready(root)
        started.set()
        while True:
            await asyncio.sleep(0.1)

    with _client(tmp_path, runner=slow_runner) as client:
        submitted = client.post(
            "/a2a",
            headers={
                "Authorization": "Bearer a2a-secret",
                "A2A-Version": "1.0",
            },
            json={
                "jsonrpc": "2.0",
                "id": "cancel-submit",
                "method": "SendMessage",
                "params": {
                    "message": {
                        "messageId": "cancel-message",
                        "role": "ROLE_USER",
                        "parts": [{"text": "Run a deliberately long analysis"}],
                    },
                    "configuration": {"returnImmediately": True},
                },
            },
        )
        assert submitted.status_code == 200
        task = submitted.json()["result"]["task"]
        assert task["status"]["state"] == "TASK_STATE_SUBMITTED"
        assert started.wait(timeout=2)

        cancelled = client.post(
            "/a2a",
            headers={
                "Authorization": "Bearer a2a-secret",
                "A2A-Version": "1.0",
            },
            json={
                "jsonrpc": "2.0",
                "id": "cancel-task",
                "method": "CancelTask",
                "params": {"id": task["id"]},
            },
        )
        assert cancelled.status_code == 200
        payload = cancelled.json()
        assert "error" not in payload, payload
        assert payload["result"]["status"]["state"] == "TASK_STATE_CANCELED"


def test_failed_run_retains_partial_provenance(tmp_path):
    async def failed_runner(objective, settings, **kwargs):
        del objective, kwargs
        root = settings.runs_dir / "partial-provenance"
        root.mkdir()
        (root / "analysis.log").write_text("partial evidence", encoding="utf-8")
        raise TimeoutError("critic unavailable")

    with _client(tmp_path, runner=failed_runner) as client:
        workspace = client.post(
            "/api/workspaces",
            auth=("researcher", "correct horse"),
            json={"name": "Partial run"},
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=("researcher", "correct horse"),
            json={"objective": "Test partial provenance", "mcp_servers": []},
        ).json()
        result = _wait_for_run(client, queued["id"])

        assert result["status"] == "failed"
        assert result["provenance_dir"]
        paths = {artifact["path"] for artifact in result["artifacts"]}
        assert {"analysis.log", "run_failure.json"}.issubset(paths)


def test_active_run_streams_events_and_artifacts_blocks_upload_and_cancels(tmp_path):
    async def slow_runner(
        objective, settings, *, on_provenance_ready, activity, **kwargs
    ):
        del objective
        root = settings.runs_dir / "slow-provenance"
        root.mkdir()
        on_provenance_ready(root)
        live = root / "visible-output.txt"
        live.write_text("observable model output", encoding="utf-8")
        activity(
            "model_output",
            "Qwen",
            "research",
            "Visible research output is available",
            str(live),
        )
        while not kwargs["cancel_event"].is_set():
            await asyncio.sleep(0.02)
        raise asyncio.CancelledError

    with _client(tmp_path, runner=slow_runner) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Live cancellation"}
        ).json()
        queued = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={"objective": "Run a slow scientific task", "mcp_servers": []},
        ).json()
        for _ in range(100):
            detail = client.get(f"/api/runs/{queued['id']}", auth=auth).json()
            if detail["provenance_dir"]:
                break
            time.sleep(0.01)
        assert detail["status"] == "running"
        assert "visible-output.txt" in {
            artifact["path"] for artifact in detail["artifacts"]
        }
        for _ in range(100):
            events = client.get(f"/api/runs/{queued['id']}/events", auth=auth).json()
            if any(
                event["actor"] == "Qwen"
                and event["artifact_path"] == "visible-output.txt"
                for event in events
            ):
                break
            time.sleep(0.01)
        else:
            raise AssertionError("live model-output event was not persisted")
        live_preview = client.get(
            f"/api/runs/{queued['id']}/artifact-preview",
            params={"path": "visible-output.txt"},
            auth=auth,
        )
        assert live_preview.status_code == 200
        assert live_preview.json()["content"] == "observable model output"
        blocked = client.post(
            f"/api/workspaces/{workspace['id']}/files",
            auth=auth,
            files={"upload": ("late.csv", b"x\n1\n", "text/csv")},
        )
        assert blocked.status_code == 409

        cancel = client.post(f"/api/runs/{queued['id']}/cancel", auth=auth)
        assert cancel.status_code == 202
        terminal = _wait_for_run(client, queued["id"])
        assert terminal["status"] == "cancelled"
        assert terminal["report"] is None
        assert "run_cancelled.json" in {
            artifact["path"] for artifact in terminal["artifacts"]
        }
        repeated = client.post(f"/api/runs/{queued['id']}/cancel", auth=auth)
        assert repeated.status_code == 202
        assert repeated.json()["status"] == "cancelled"


def test_registered_display_endpoints_are_integrity_gated(tmp_path):
    with _client(tmp_path) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Displays"}
        ).json()
        store = client.app.state.store
        run = store.create_run(workspace["id"], "Render the report", False, ())
        _, runs_dir = store.paths(workspace["id"])
        root = runs_dir / "display-run"
        displays = root / "displays"
        displays.mkdir(parents=True)
        image = displays / "effect.png"
        Image.new("RGB", (640, 400), color=(240, 248, 246)).save(image)
        table = displays / "effects.csv"
        table.write_text("group,estimate\nA,1.25\n", encoding="utf-8")
        (root / "display_manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "displays": [
                        {
                            "display_id": "effect-plot",
                            "kind": "figure",
                            "number": 1,
                            "title": "Effect plot",
                            "caption": "Point estimate with uncertainty.",
                            "path": "displays/effect.png",
                            "sha256": sha256_file(image),
                        },
                        {
                            "display_id": "effect-table",
                            "kind": "table",
                            "number": 1,
                            "title": "Effects",
                            "caption": "Exact estimates.",
                            "path": "displays/effects.csv",
                            "sha256": sha256_file(table),
                            "columns": ["group", "estimate"],
                            "rows": [["A", "1.25"]],
                            "total_rows": 1,
                            "total_columns": 2,
                            "truncated": False,
                            "claim_ids": [],
                            "evidence_refs": [],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        store.finish_run(
            run["id"],
            status="supported",
            phase="complete",
            message="Ready",
            finished_at="2026-07-14T00:00:00Z",
            provenance_dir=str(root),
        )

        figure = client.get(
            f"/api/runs/{run['id']}/displays/effect-plot/image", auth=auth
        )
        assert figure.status_code == 200
        assert figure.headers["content-type"] == "image/png"
        assert figure.headers["content-disposition"].startswith("inline")
        preview = client.get(
            f"/api/runs/{run['id']}/displays/effect-table/table", auth=auth
        )
        assert preview.json()["rows"] == [["A", "1.25"]]
        assert (
            client.get(
                f"/api/runs/{run['id']}/displays/missing/image", auth=auth
            ).status_code
            == 404
        )

        image.write_bytes(b"tampered")
        assert (
            client.get(
                f"/api/runs/{run['id']}/displays/effect-plot/image", auth=auth
            ).status_code
            == 404
        )


def test_follow_up_creates_immutable_child_with_inherited_settings(tmp_path):
    calls = []

    async def revision_runner(objective, settings, *, on_provenance_ready, **kwargs):
        calls.append((objective, kwargs))
        root = settings.runs_dir / f"revision-fixture-{len(calls)}"
        root.mkdir()
        on_provenance_ready(root)
        report = {
            "title": f"Report {len(calls)}",
            "executive_summary": "A deterministic revision fixture.",
            "introduction": "The fixture tests immutable report revisions.",
            "methods": ["A fixed integration method"],
            "results": "The revision path completed.",
            "discussion": "The result is limited to integration behavior.",
            "conclusions": "The parent and child are separate records.",
            "displays": [],
            "claims": [],
            "sources": [],
            "unresolved_issues": [],
            "limitations": [],
        }
        (root / "scientific_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
        (root / "report.md").write_text("# Fixture\n", encoding="utf-8")
        (root / "run_result.json").write_text(
            json.dumps({"status": "supported"}), encoding="utf-8"
        )
        (root / "manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")
        return SimpleNamespace(status="supported", provenance_dir=str(root))

    with _client(tmp_path, runner=revision_runner) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Revisions"}
        ).json()
        parent = client.post(
            f"/api/workspaces/{workspace['id']}/runs",
            auth=auth,
            json={
                "objective": "Create the original report",
                "enable_code": True,
                "mcp_servers": ["context7"],
            },
        ).json()
        parent = _wait_for_run(client, parent["id"])
        parent_report = Path(parent["provenance_dir"]) / "scientific_report.json"
        parent_hash = sha256_file(parent_report)

        child = client.post(
            f"/api/runs/{parent['id']}/follow-ups",
            auth=auth,
            json={"request": "Clarify the methods and limitations"},
        )
        assert child.status_code == 202
        child = _wait_for_run(client, child.json()["id"])

        assert child["parent_run_id"] == parent["id"]
        assert child["run_kind"] == "revision"
        assert child["enable_code"] is True
        assert child["mcp_servers"] == ["context7"]
        assert calls[1][1]["revision_request"] == "Clarify the methods and limitations"
        assert calls[1][1]["parent_provenance_dir"] == Path(parent["provenance_dir"])

        writing_only = client.post(
            f"/api/runs/{parent['id']}/follow-ups",
            auth=auth,
            json={
                "request": "Correct captions without recomputing results",
                "enable_code": False,
            },
        )
        assert writing_only.status_code == 202
        writing_only = _wait_for_run(client, writing_only.json()["id"])
        assert writing_only["enable_code"] is False
        assert sha256_file(parent_report) == parent_hash


def test_workspace_delete_is_authenticated_and_cleanup_failure_rolls_back(
    tmp_path, monkeypatch
):
    cleaned = []

    def cleanup(settings, workspace_id):
        del settings
        cleaned.append(workspace_id)
        return {
            "status": "deleted",
            "workspace_id": workspace_id,
            "removed_bytes": 0,
        }

    monkeypatch.setattr(
        "scientific_agent.web.app.cleanup_workspace_environment", cleanup
    )
    with _client(tmp_path) as client:
        auth = ("researcher", "correct horse")
        workspace = client.post(
            "/api/workspaces", auth=auth, json={"name": "Delete through API"}
        ).json()

        denied = client.delete(f"/api/workspaces/{workspace['id']}")
        assert denied.status_code == 401
        assert cleaned == []

        deleted = client.delete(f"/api/workspaces/{workspace['id']}", auth=auth)
        assert deleted.status_code == 204
        assert cleaned == [workspace["id"]]
        assert (
            client.get(f"/api/workspaces/{workspace['id']}", auth=auth).status_code
            == 404
        )

        retained = client.post(
            "/api/workspaces", auth=auth, json={"name": "Rollback target"}
        ).json()

        def fail_cleanup(settings, workspace_id):
            del settings, workspace_id
            raise RuntimeError("package worker cleanup request failed")

        monkeypatch.setattr(
            "scientific_agent.web.app.cleanup_workspace_environment", fail_cleanup
        )
        failed = client.delete(f"/api/workspaces/{retained['id']}", auth=auth)

        assert failed.status_code == 409
        assert "cleanup request failed" in failed.json()["detail"]
        assert (
            client.get(f"/api/workspaces/{retained['id']}", auth=auth).status_code
            == 200
        )
