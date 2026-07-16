import hashlib
import io
import json
import zipfile
from pathlib import Path, PurePosixPath

from fastapi.testclient import TestClient

from scientific_agent import __version__
from scientific_agent.config import Settings
from scientific_agent.web.app import create_app
from scientific_agent.web.settings import WebSettings


def _client(tmp_path: Path, *, a2a_enabled: bool = True) -> TestClient:
    return TestClient(
        create_app(
            WebSettings(
                data_dir=tmp_path,
                auth_enabled=False,
                username="",
                password="",
                a2a_enabled=a2a_enabled,
                a2a_token="a2a-test-token" if a2a_enabled else "",
                public_url="http://10.20.102.122",
                max_workers=1,
            ),
            Settings(),
        )
    )


def _zip(response) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(response.content))


def _assert_safe_reproducible_zip(response, second_response) -> None:
    digest = hashlib.sha256(response.content).hexdigest()
    assert response.content == second_response.content
    assert response.headers["x-checksum-sha256"] == digest
    assert response.headers["etag"] == f'"sha256:{digest}"'
    assert response.headers["content-type"] == "application/zip"
    with _zip(response) as archive:
        assert archive.namelist() == sorted(archive.namelist())
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            assert not path.is_absolute()
            assert ".." not in path.parts
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert (info.external_attr >> 16) & 0o170000 == 0o100000


def _assert_inner_checksums(archive: zipfile.ZipFile, root: str) -> None:
    manifest = archive.read(f"{root}/SHA256SUMS").decode("utf-8")
    for line in manifest.splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256(archive.read(f"{root}/{name}")).hexdigest() == digest


def test_webui_catalog_and_skill_archive_are_verifiable_and_installable(tmp_path):
    with _client(tmp_path) as client:
        catalog = client.get("/api/integrations")
        response = client.get("/api/integrations/skill")
        second = client.get("/api/integrations/skill")

    assert catalog.status_code == 200
    skill = next(item for item in catalog.json()["downloads"] if item["id"] == "skill")
    assert skill["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert skill["url"] == "/api/integrations/skill"
    assert skill["filename"] == f"evidence-bench-skill-v{__version__}.zip"
    assert "a2a-test-token" not in json.dumps(catalog.json())
    assert response.status_code == 200
    assert response.headers["content-disposition"] == (
        f'attachment; filename="evidence-bench-skill-v{__version__}.zip"'
    )
    _assert_safe_reproducible_zip(response, second)
    with _zip(response) as archive:
        expected = {
            "evidence-bench/SKILL.md",
            "evidence-bench/agents/openai.yaml",
            "evidence-bench/scripts/evidence_bench.py",
            "evidence-bench/SHA256SUMS",
        }
        assert set(archive.namelist()) == expected
        for relative in ("SKILL.md", "agents/openai.yaml", "scripts/evidence_bench.py"):
            assert (
                archive.read(f"evidence-bench/{relative}")
                == (Path("skills/evidence-bench") / relative).read_bytes()
            )
        _assert_inner_checksums(archive, "evidence-bench")


def test_a2a_starter_has_deployment_urls_but_never_the_bearer_token(tmp_path):
    with _client(tmp_path) as client:
        catalog = client.get("/api/integrations").json()
        response = client.get("/api/integrations/a2a")
        second = client.get("/api/integrations/a2a")

    a2a = next(item for item in catalog["downloads"] if item["id"] == "a2a")
    assert a2a["sha256"] == hashlib.sha256(response.content).hexdigest()
    assert a2a["filename"] == f"evidence-bench-a2a-v{__version__}.zip"
    assert b"a2a-test-token" not in response.content
    _assert_safe_reproducible_zip(response, second)
    with _zip(response) as archive:
        assert set(archive.namelist()) == {
            "evidence-bench-a2a/README.md",
            "evidence-bench-a2a/SHA256SUMS",
            "evidence-bench-a2a/a2a_client.py",
            "evidence-bench-a2a/connection.json",
        }
        connection = json.loads(archive.read("evidence-bench-a2a/connection.json"))
        assert connection == {
            "a2a_enabled": True,
            "a2a_url": "http://10.20.102.122/a2a",
            "agent_card_url": ("http://10.20.102.122/.well-known/agent-card.json"),
            "protocol_binding": "JSONRPC",
            "protocol_version": "1.0",
            "service_url": "http://10.20.102.122",
            "service_version": __version__,
        }
        _assert_inner_checksums(archive, "evidence-bench-a2a")


def test_integration_routes_do_not_accept_user_controlled_paths(tmp_path):
    with _client(tmp_path, a2a_enabled=False) as client:
        catalog = client.get("/api/integrations").json()
        traversal = client.get("/api/integrations/%2e%2e/pyproject.toml")
        archive = client.get("/api/integrations/a2a")

    assert catalog["a2a_enabled"] is False
    assert traversal.status_code == 404
    with _zip(archive) as bundle:
        connection = json.loads(bundle.read("evidence-bench-a2a/connection.json"))
        assert connection["a2a_enabled"] is False


def test_webui_exposes_agent_downloads_and_checksum_fields():
    page = Path("scientific_agent/web/static/index.html").read_text(encoding="utf-8")
    script = Path("scientific_agent/web/static/app.js").read_text(encoding="utf-8")

    assert 'href="/api/integrations/skill"' in page
    assert 'href="/api/integrations/a2a"' in page
    assert "skill-integration-checksum" in page
    assert "a2a-integration-checksum" in page
    assert 'api("/api/integrations")' in script
