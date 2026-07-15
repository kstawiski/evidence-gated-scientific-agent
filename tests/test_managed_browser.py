from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from scientific_agent.config import Settings
from scientific_agent.web.app import create_app
from scientific_agent.web.settings import WebSettings


def test_compose_keeps_cdp_internal_and_downloads_read_only():
    compose = yaml.safe_load(Path("compose.yaml").read_text(encoding="utf-8"))
    services = compose["services"]
    browser = services["browser"]
    ui_gateway = services["browser-ui-gateway"]
    egress = services["browser-egress"]
    gateway = services["browser-cdp-gateway"]
    sandbox = services["sandbox-worker"]
    sandbox_gateway = services["sandbox-gateway"]
    packages = services["environment-worker"]
    package_gateway = services["environment-gateway"]
    app = services["evidence-bench"]

    assert browser["expose"] == ["9222"]
    assert "ports" not in browser
    assert ui_gateway["ports"] == [
        "${BROWSER_BIND_ADDRESS:-127.0.0.1}:${BROWSER_NOVNC_PORT:-6080}:6080"
    ]
    assert all("9222:" not in value for value in ui_gateway["ports"])
    assert browser["shm_size"] == "2gb"
    assert browser["user"] == "${BROWSER_RUNTIME_USER:-0:0}"
    assert (
        browser["environment"]["BROWSER_DOWNLOADS_MODE"]
        == "${BROWSER_DOWNLOADS_MODE:-2750}"
    )
    assert "9222/json/version" in " ".join(browser["healthcheck"]["test"])
    assert app["environment"]["CHROME_DEVTOOLS_BROWSER_URL"] == (
        "http://browser-cdp-gateway:9222"
    )
    assert browser["networks"] == ["browser-control"]
    assert set(app["networks"]) == {
        "default",
        "browser-client",
        "sandbox-client",
        "package-client",
    }
    assert app["environment"]["SCIENTIFIC_AGENT_SANDBOX_WORKER_URL"] == (
        "http://sandbox-gateway:8090"
    )
    assert app["environment"]["SCIENTIFIC_AGENT_PACKAGE_WORKER_URL"] == (
        "http://environment-gateway:8091"
    )
    assert set(gateway["networks"]) == {"browser-control", "browser-client"}
    assert set(ui_gateway["networks"]) == {"browser-control", "browser-ui"}
    assert ui_gateway["cap_drop"] == ["ALL"]
    assert gateway["command"] == [
        "TCP-LISTEN:9222,reuseaddr,fork",
        "TCP:browser:9222",
    ]
    assert gateway["cap_drop"] == ["ALL"]
    assert set(egress["networks"]) == {
        "public-egress",
        "browser-control",
        "packages",
    }
    assert all(
        "public-egress" not in service.get("networks", [])
        for name, service in services.items()
        if name != "browser-egress"
    )
    assert egress["cap_drop"] == ["ALL"]
    assert sandbox["networks"] == ["sandbox"]
    assert set(sandbox_gateway["networks"]) == {"sandbox", "sandbox-client"}
    assert sandbox_gateway["command"] == [
        "TCP-LISTEN:8090,reuseaddr,fork",
        "TCP:sandbox-worker:8090",
    ]
    assert packages["networks"] == ["packages"]
    assert packages["environment"]["SCIENTIFIC_AGENT_PACKAGE_PROXY_URL"] == (
        "http://browser-egress:3128"
    )
    assert packages["depends_on"]["browser-egress"]["condition"] == ("service_healthy")
    assert set(package_gateway["networks"]) == {"packages", "package-client"}
    assert package_gateway["command"] == [
        "TCP-LISTEN:8091,reuseaddr,fork",
        "TCP:environment-worker:8091",
    ]
    assert sandbox_gateway["cap_drop"] == ["ALL"]
    assert package_gateway["cap_drop"] == ["ALL"]
    assert compose["networks"]["browser-control"]["internal"] is True
    assert compose["networks"]["browser-client"]["internal"] is True
    assert compose["networks"]["sandbox"]["internal"] is True
    assert compose["networks"]["sandbox-client"]["internal"] is True
    assert compose["networks"]["packages"]["internal"] is True
    assert compose["networks"]["package-client"]["internal"] is True
    assert any(
        value.endswith("/downloads:/browser-downloads:ro") for value in app["volumes"]
    )
    assert (
        app["environment"]["SCIENTIFIC_AGENT_BROWSER_DOWNLOADS"] == "/browser-downloads"
    )


def test_browser_image_is_service_owned_and_passwordless():
    entrypoint = Path("browser/entrypoint.sh").read_text(encoding="utf-8")
    dockerfile = Path("browser/Dockerfile").read_text(encoding="utf-8")
    nginx = Path("browser/nginx.conf").read_text(encoding="utf-8")
    squid = Path("browser/squid.conf").read_text(encoding="utf-8")

    assert "--remote-debugging-port=9223" in entrypoint
    assert "nginx.conf" in dockerfile
    assert "listen 0.0.0.0:9222" in nginx
    assert nginx.index("error_log /tmp/nginx-error.log warn;") < nginx.index("events {")
    assert "-localhost" in entrypoint
    assert "-nopw" in entrypoint
    assert "websockify" in entrypoint
    assert "--proxy-server=http://browser-egress:3128" in entrypoint
    assert "--proxy-bypass-list='<-loopback>'" in entrypoint
    assert "downloads_mode=${BROWSER_DOWNLOADS_MODE:-2750}" in entrypoint
    assert 'chmod "$downloads_mode" "$downloads"' in entrypoint
    assert "DOWNLOAD_GID=10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "squid" in dockerfile
    assert "socat" in dockerfile
    for blocked_network in (
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "fc00::/7",
        "fe80::/10",
    ):
        assert blocked_network in squid
    assert "http_access deny non_public" in squid
    assert "http_access deny CONNECT !safe_https" in squid
    assert "http_access allow CONNECT safe_https" in squid
    assert "100.64." not in entrypoint + dockerfile
    assert "10.20." not in entrypoint + dockerfile


def test_web_config_derives_current_host_browser_and_allows_frame(tmp_path):
    web = WebSettings(
        data_dir=tmp_path,
        auth_enabled=False,
        a2a_enabled=False,
        browser_novnc_port=6087,
    )
    with TestClient(
        create_app(web, Settings(chrome_browser_url="http://browser:9222"))
    ) as client:
        config = client.get("/api/config").json()
        assert config["browser"] == {
            "enabled": True,
            "public_url": "",
            "novnc_port": 6087,
        }
        csp = client.get("/").headers["content-security-policy"]
        assert "frame-src 'self' http://*:6087 https://*:6087" in csp


def test_browser_public_url_rejects_credentials(tmp_path):
    settings = WebSettings(
        data_dir=tmp_path,
        auth_enabled=False,
        a2a_enabled=False,
        browser_public_url="http://user:secret@example.test:6080/vnc.html",
    )
    with pytest.raises(ValueError, match="without credentials"):
        settings.validate()


def test_browser_panel_remains_available_during_active_runs():
    page = Path("scientific_agent/web/static/index.html").read_text(encoding="utf-8")
    script = Path("scientific_agent/web/static/app.js").read_text(encoding="utf-8")

    assert 'id="research-browser-dialog"' in page
    assert 'id="research-browser-frame"' in page
    launch = page.split('id="research-browser-button"', 1)[1].split(">", 1)[0]
    assert "data-run-mutable" not in launch
    assert "configuredBrowserUrl" in script
    assert 'url.pathname = "/vnc.html"' in script
