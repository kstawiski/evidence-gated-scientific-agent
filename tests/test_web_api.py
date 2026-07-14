import json
import time
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from scientific_agent.config import Settings
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
        "methods": ["Fixture method"],
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
        "sources": [],
        "unresolved_issues": [],
        "limitations": [],
        "narrative": "Fixture narrative.",
    }
    (root / "report.md").write_text("# Validated test report\n", encoding="utf-8")
    (root / "scientific_report.json").write_text(json.dumps(report), encoding="utf-8")
    (root / "run_result.json").write_text(
        json.dumps({"status": "supported"}), encoding="utf-8"
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"path": "report.md", "bytes": 24, "sha256": "a" * 64}
                ]
            }
        ),
        encoding="utf-8",
    )
    return SimpleNamespace(status="supported", provenance_dir=str(root))


def _client(tmp_path: Path, runner=fake_runner) -> TestClient:
    web = WebSettings(
        data_dir=tmp_path,
        username="researcher",
        password="correct horse",
        a2a_token="a2a-secret",
        public_url="https://agent.example.test",
        max_workers=1,
    )
    app = create_app(web, Settings(), runner=runner)
    return TestClient(app)


def _wait_for_run(client, run_id):
    for _ in range(100):
        response = client.get(f"/api/runs/{run_id}", auth=("researcher", "correct horse"))
        if response.json()["status"] not in {"queued", "running"}:
            return response.json()
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def test_ui_api_auth_workspace_upload_and_run(tmp_path):
    with _client(tmp_path) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/").status_code == 401
        assert client.get("/", auth=("researcher", "correct horse")).status_code == 200

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

        queued = client.post(
            f"/api/workspaces/{workspace_id}/runs",
            auth=("researcher", "correct horse"),
            json={
                "objective": "Analyze the uploaded values",
                "enable_code": True,
                "mcp_servers": [],
            },
        )
        assert queued.status_code == 202
        result = _wait_for_run(client, queued.json()["id"])
        assert result["status"] == "supported"
        assert result["report"]["title"] == "Validated test report"
        artifact = client.get(
            f"/api/runs/{result['id']}/artifacts",
            params={"path": "report.md"},
            auth=("researcher", "correct horse"),
        )
        assert artifact.status_code == 200
        assert artifact.text.startswith("# Validated")
        bundle = client.get(
            f"/api/runs/{result['id']}/bundle",
            auth=("researcher", "correct horse"),
        )
        assert bundle.status_code == 200
        with zipfile.ZipFile(BytesIO(bundle.content)) as archive:
            assert "report.md" in archive.namelist()
            assert archive.read("report.md").startswith(b"# Validated")


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


def test_browser_auth_flag_loads_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("SCIENTIFIC_AGENT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WEB_AUTH_ENABLED", "false")
    monkeypatch.setenv("WEB_USERNAME", "")
    monkeypatch.setenv("WEB_PASSWORD", "")
    monkeypatch.setenv("A2A_TOKEN", "a2a-secret")

    settings = WebSettings()
    settings.validate()
    assert settings.auth_enabled is False


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
        assert payload["result"]["task"]["status"]["state"] == "TASK_STATE_COMPLETED"


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
