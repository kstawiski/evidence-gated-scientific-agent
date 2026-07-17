import io
import json

import httpx
import pytest

from evals.run_a2a_live_gate import (
    DEFAULT_MCP_SERVERS,
    GateConfig,
    GateError,
    config_from_args,
    main,
    run_gate,
)


TOKEN = "token-that-must-not-be-printed"
REPORT = """# Scientific comparison

## Introduction
Context.

## Methods
Methods.

## Results
Results.

## Discussion
Discussion.

## Conclusions
Conclusions.
"""


def _event(result):
    return "data: " + json.dumps(
        {"jsonrpc": "2.0", "id": "a2a-live-gate-stream", "result": result}
    )


def _transport(
    *,
    get_task_id="server-assigned-task",
    scientific_status="supported_with_comments",
):
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.method == "GET":
            assert request.url.path == "/.well-known/agent-card.json"
            assert "authorization" not in request.headers
            return httpx.Response(
                200,
                json={
                    "name": "Evidence Bench",
                    "version": "0.4.1",
                    "supportedInterfaces": [
                        {
                            "url": "https://agent.example.test/a2a",
                            "protocolBinding": "JSONRPC",
                            "protocolVersion": "1.0",
                        }
                    ],
                    "capabilities": {"streaming": True},
                },
            )

        assert request.url.path == "/a2a"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert request.headers["a2a-version"] == "1.0"
        payload = json.loads(request.content)
        if payload["method"] == "SendStreamingMessage":
            message = payload["params"]["message"]
            assert "taskId" not in message
            assert message["metadata"] == {
                "enable_code": True,
                "mcp_servers": list(DEFAULT_MCP_SERVERS),
            }
            body = "\n\n".join(
                [
                    _event(
                        {
                            "task": {
                                "id": "server-assigned-task",
                                "contextId": "server-context",
                                "status": {"state": "TASK_STATE_SUBMITTED"},
                            }
                        }
                    ),
                    _event(
                        {
                            "statusUpdate": {
                                "taskId": "server-assigned-task",
                                "contextId": "server-context",
                                "status": {"state": "TASK_STATE_WORKING"},
                                "final": False,
                            }
                        }
                    ),
                    _event(
                        {
                            "artifactUpdate": {
                                "taskId": "server-assigned-task",
                                "contextId": "server-context",
                                "artifact": {"name": "run-summary.json"},
                            }
                        }
                    ),
                    _event(
                        {
                            "artifactUpdate": {
                                "taskId": "server-assigned-task",
                                "contextId": "server-context",
                                "artifact": {"name": "report.md"},
                            }
                        }
                    ),
                    _event(
                        {
                            "statusUpdate": {
                                "taskId": "server-assigned-task",
                                "contextId": "server-context",
                                "status": {"state": "TASK_STATE_COMPLETED"},
                                "final": True,
                            }
                        }
                    ),
                ]
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=body,
            )

        assert payload == {
            "jsonrpc": "2.0",
            "id": "a2a-live-gate-get-task",
            "method": "GetTask",
            "params": {"id": "server-assigned-task"},
        }
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": "a2a-live-gate-get-task",
                "result": {
                    "id": get_task_id,
                    "contextId": "server-context",
                    "status": {"state": "TASK_STATE_COMPLETED"},
                    "artifacts": [
                        {
                            "name": "run-summary.json",
                            "parts": [
                                {
                                    "data": {
                                        "run_id": "run-1",
                                        "workspace_id": "workspace-1",
                                        "status": scientific_status,
                                    },
                                    "mediaType": "application/json",
                                }
                            ],
                        },
                        {
                            "name": "report.md",
                            "parts": [{"text": REPORT, "mediaType": "text/markdown"}],
                        },
                    ],
                },
            },
        )

    return httpx.MockTransport(handler), observed


def _config():
    return GateConfig(
        base_url="https://agent.example.test",
        token=TOKEN,
        timeout_seconds=30,
        prompt="Perform the scientific fixture.",
        enable_code=True,
    )


def test_live_gate_verifies_stream_artifacts_get_task_and_imrad():
    transport, observed = _transport()

    result = run_gate(_config(), transport=transport)

    assert result == {
        "passed": True,
        "agent_version": "0.4.1",
        "protocol_binding": "JSONRPC",
        "protocol_version": "1.0",
        "task_id": "server-assigned-task",
        "run_id": "run-1",
        "workspace_id": "workspace-1",
        "scientific_status": "supported_with_comments",
        "stream_states": [
            "TASK_STATE_SUBMITTED",
            "TASK_STATE_WORKING",
            "TASK_STATE_COMPLETED",
        ],
        "artifacts": ["report.md", "run-summary.json"],
        "mcp_servers": list(DEFAULT_MCP_SERVERS),
        "enable_code": True,
    }
    assert [request.method for request in observed] == ["GET", "POST", "POST"]


def test_live_gate_rejects_get_task_id_mismatch():
    transport, _ = _transport(get_task_id="different-task")

    with pytest.raises(GateError, match="different task ID"):
        run_gate(_config(), transport=transport)


@pytest.mark.parametrize(
    "scientific_status",
    [
        "contradicted",
        "inconclusive",
        "requires_more_evidence",
        "requires_human_decision",
    ],
)
def test_live_gate_rejects_non_success_scientific_status(scientific_status):
    transport, _ = _transport(scientific_status=scientific_status)

    with pytest.raises(GateError, match="non-success scientific status"):
        run_gate(_config(), transport=transport)


def test_cli_reads_configuration_from_env_and_never_prints_token():
    transport, _ = _transport()
    stdout = io.StringIO()
    stderr = io.StringIO()
    env = {
        "A2A_TOKEN": TOKEN,
        "A2A_BASE_URL": "https://agent.example.test/",
        "A2A_TIMEOUT_SECONDS": "30",
        "A2A_PROMPT": "Perform the scientific fixture.",
        "A2A_MCP_SERVERS": "context7,brave-search,chrome-devtools,context7",
        "A2A_ENABLE_CODE": "true",
    }

    exit_code = main(
        [],
        environ=env,
        transport=transport,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert json.loads(stdout.getvalue())["passed"] is True
    assert stderr.getvalue() == ""
    assert TOKEN not in stdout.getvalue()
    assert TOKEN not in stderr.getvalue()


def test_cli_defaults_to_all_research_mcps():
    config = config_from_args([], {"A2A_TOKEN": TOKEN})

    assert config.mcp_servers == DEFAULT_MCP_SERVERS


def test_cli_allows_explicit_empty_mcp_opt_out():
    config = config_from_args([], {"A2A_TOKEN": TOKEN, "A2A_MCP_SERVERS": ","})

    assert config.mcp_servers == ()

    config = config_from_args(
        ["--mcp-server", ""],
        {"A2A_TOKEN": TOKEN},
    )
    assert config.mcp_servers == ()
