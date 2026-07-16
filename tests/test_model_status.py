import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scientific_agent.config import Settings
from scientific_agent.web.app import create_app
from scientific_agent.web.model_status import (
    ModelStatusMonitor,
    ModelStatusTarget,
    validate_status_base_url,
)
from scientific_agent.web.settings import WebSettings


QWEN_METRICS = b"""\
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="private-backend-id"} 2
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="private-backend-id"} 1
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="private-backend-id"} 0.375
"""
GEMMA_METRICS = b"""\
# TYPE llamacpp:requests_processing gauge
llamacpp:requests_processing 1
# TYPE llamacpp:requests_deferred gauge
llamacpp:requests_deferred 0
"""


def test_monitor_reports_authoritative_vllm_queue_and_llama_slots_with_cache():
    calls = []

    def fetch(url, timeout):
        calls.append((url, timeout))
        if url == "http://qwen.internal/metrics":
            return QWEN_METRICS
        if url == "http://gemma.internal/metrics":
            return GEMMA_METRICS
        if url == "http://gemma.internal/slots":
            return json.dumps([{"id": 0, "is_processing": True}]).encode()
        raise OSError("unexpected fixed probe")

    monitor = ModelStatusMonitor(
        (
            ModelStatusTarget("executor", "umed-qwen", "vllm", "http://qwen.internal"),
            ModelStatusTarget(
                "critic", "s8-gemma", "llama_cpp", "http://gemma.internal"
            ),
        ),
        timeout_seconds=1.25,
        cache_seconds=30,
        fetch=fetch,
    )

    first = monitor.snapshot()
    second = monitor.snapshot()

    assert second is first
    assert len(calls) == 3
    assert first["models"] == [
        {
            "role": "executor",
            "model": "umed-qwen",
            "provider": "vllm",
            "reachable": True,
            "state": "queued",
            "active_requests": 2,
            "queued_requests": 1,
            "slots_total": None,
            "slots_busy": None,
            "cache_usage_percent": 37.5,
            "message": "1 request waiting",
        },
        {
            "role": "critic",
            "model": "s8-gemma",
            "provider": "llama_cpp",
            "reachable": True,
            "state": "saturated",
            "active_requests": 1,
            "queued_requests": 0,
            "slots_total": 1,
            "slots_busy": 1,
            "cache_usage_percent": None,
            "message": "All inference slots are busy",
        },
    ]
    assert first["summary"]["queued_requests"] == 1
    assert first["summary"]["saturated_models"] == 1


def test_vllm_cache_usage_is_averaged_across_multiple_engines():
    metrics = QWEN_METRICS + (
        b'vllm:kv_cache_usage_perc{engine="1",model_name="private-backend-id"} 0.625\n'
    )
    monitor = ModelStatusMonitor(
        (ModelStatusTarget("executor", "umed-qwen", "vllm", "http://qwen.internal"),),
        fetch=lambda url, timeout: metrics,
    )

    model = monitor.snapshot()["models"][0]

    assert model["cache_usage_percent"] == 50.0


def test_monitor_fails_closed_without_leaking_backend_url_or_errors():
    def broken_fetch(url, timeout):
        raise OSError(f"secret upstream failed: {url}?token=do-not-return")

    monitor = ModelStatusMonitor(
        (ModelStatusTarget("executor", "umed-qwen", "vllm", "http://secret.local"),),
        fetch=broken_fetch,
    )

    payload = monitor.snapshot()

    assert payload["models"][0]["state"] == "offline"
    assert payload["models"][0]["message"] == "Backend status is unavailable"
    assert "secret.local" not in json.dumps(payload)
    assert "do-not-return" not in json.dumps(payload)


@pytest.mark.parametrize(
    "value",
    [
        "file:///etc/passwd",
        "http://user:password@backend.internal",
        "http://backend.internal/metrics",
        "http://backend.internal?target=http://169.254.169.254",
        "http://backend.internal/#fragment",
    ],
)
def test_status_origin_rejects_non_origin_and_credential_urls(value):
    with pytest.raises(ValueError, match=r"HTTP\(S\) origin"):
        validate_status_base_url(value, "QWEN_STATUS_BASE_URL")


def test_web_settings_bound_probe_timeout_and_cache(monkeypatch):
    monkeypatch.setenv("WEB_AUTH_ENABLED", "false")
    monkeypatch.setenv("A2A_ENABLED", "false")
    monkeypatch.setenv("MODEL_STATUS_TIMEOUT_SECONDS", "6")
    with pytest.raises(ValueError, match="between 0.1 and 5"):
        WebSettings().validate()

    monkeypatch.setenv("MODEL_STATUS_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("MODEL_STATUS_CACHE_SECONDS", "0")
    with pytest.raises(ValueError, match="between 1 and 60"):
        WebSettings().validate()


def test_model_status_api_is_auth_protected_and_returns_no_probe_url(tmp_path):
    class StaticMonitor:
        @staticmethod
        def snapshot():
            return {
                "updated_at": "2026-07-16T12:00:00+00:00",
                "models": [
                    {
                        "role": "executor",
                        "model": "umed-qwen",
                        "provider": "vllm",
                        "reachable": True,
                        "state": "ready",
                        "active_requests": 0,
                        "queued_requests": 0,
                        "slots_total": None,
                        "slots_busy": None,
                        "cache_usage_percent": 0.0,
                        "message": "Available now",
                    }
                ],
                "summary": {
                    "queued_requests": 0,
                    "saturated_models": 0,
                    "offline_models": 0,
                    "unconfigured_models": 0,
                    "message": "Both scientific models are available now",
                },
            }

    web = WebSettings(
        data_dir=tmp_path,
        username="researcher",
        password="correct horse",
        a2a_token="a2a-secret",
        public_url="https://agent.example.test",
        max_workers=1,
    )
    with TestClient(
        create_app(web, Settings(), model_status_monitor=StaticMonitor())
    ) as client:
        assert client.get("/api/model-status").status_code == 401
        response = client.get("/api/model-status", auth=("researcher", "correct horse"))

    assert response.status_code == 200
    assert response.json()["models"][0]["state"] == "ready"
    assert "url" not in json.dumps(response.json()).lower()


def test_model_queue_panel_refreshes_without_unsafe_html():
    page = Path("scientific_agent/web/static/index.html").read_text(encoding="utf-8")
    script = Path("scientific_agent/web/static/app.js").read_text(encoding="utf-8")

    assert 'id="model-queue-panel"' in page
    assert 'id="model-status-executor"' in page
    assert 'id="model-status-critic"' in page
    assert 'api("/api/model-status")' in script
    assert "scheduleModelStatus" in script
    assert "innerHTML" not in script
