import json
import asyncio
from dataclasses import replace
from pathlib import Path
import threading

import httpx
import pytest
from pydantic import BaseModel

from scientific_agent.config import ModelEndpoint
from scientific_agent.schemas import VerificationReport
from scientific_agent.structured_client import (
    _StreamRepetitionGuard,
    _private_reasoning_no_final_limit,
    _stream_is_repeating,
    request_structured,
)


class Answer(BaseModel):
    value: str


def _endpoint():
    return ModelEndpoint(
        base_url="http://model.invalid/v1",
        model="local-model",
        api_key="",
        max_tokens=1000,
        temperature=0.2,
        top_p=0.9,
    )


@pytest.mark.asyncio
async def test_schema_incomplete_critic_fail_is_preserved_as_blockers_after_repair():
    calls = 0
    visible: list[str] = []
    incomplete = json.dumps(
        {
            "verdict": "fail",
            "unsupported_claims": [
                "The Results reports a diagnostic without a matching ClaimRecord."
            ],
            "evidence_refs": ["src-python"],
        }
    )

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["stream"] is True
        event = {
            "choices": [{"delta": {"content": incomplete}, "finish_reason": "stop"}]
        }
        return httpx.Response(
            200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Audit the report.",
        payload={"report": "example"},
        output_type=VerificationReport,
        temperature=0.4,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert calls == 2
    assert result.verdict == "fail"
    assert len(result.blocking_findings) == 1
    assert result.blocking_findings[0].problem.startswith("The Results reports")
    assert result.unsupported_claims
    assert "no approval was inferred" in "".join(visible)


@pytest.mark.asyncio
async def test_schema_repair_cannot_erase_malformed_explicit_critic_fail():
    calls = 0
    visible: list[str] = []
    malformed_fail = """{
      "verdict": "fail",
      "blocking_findings" ,
      "location" , "Panel A",
      "evidence" , "Only 15 of 20 reported subjects are visibly recoverable because points overlap.",
      "why_it_matters" , "The raw-data display does not transparently represent every reported individual.",
      "correction" , "Jitter only the categorical coordinate so all observations are countable."
    ]}"""
    repaired_pass = json.dumps(
        {
            "verdict": "pass",
            "blocking_findings": [],
            "nonblocking_findings": [],
            "protocol_deviations": [],
            "unsupported_claims": [],
            "proposed_falsification_tests": [],
            "evidence_refs": ["display-reviewed:figure-1"],
        }
    )

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        if calls == 2:
            repair = json.loads(body["messages"][1]["content"])
            assert "never change fail to pass" in repair["repair_instruction"]
        content = malformed_fail if calls == 1 else repaired_pass
        event = {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}]}
        return httpx.Response(
            200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n"
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Audit the display.",
        payload={"display": "example"},
        output_type=VerificationReport,
        temperature=0.4,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert calls == 2
    assert result.verdict == "fail"
    assert any("15 of 20" in item.problem for item in result.blocking_findings)
    assert "cannot erase" in "".join(visible)


def test_stream_repetition_detector_targets_sentence_loops_not_json_structure():
    repeated_sentence = (
        "The reported treatment effect requires independent external validation."
    )
    assert _stream_is_repeating("\n".join([repeated_sentence] * 40))

    structured = json.dumps(
        {
            "blocking_findings": [
                {
                    "finding_id": f"finding-{index}",
                    "proposed_falsification_tests": [],
                    "protocol_deviations": [],
                }
                for index in range(12)
            ]
        },
        indent=2,
    )
    assert not _stream_is_repeating(structured)


def test_private_reasoning_no_final_guard_is_stricter_for_gemma():
    qwen_limit = _private_reasoning_no_final_limit(
        replace(_endpoint(), model="umed-qwen")
    )
    gemma_limit = _private_reasoning_no_final_limit(
        replace(_endpoint(), model="s8-gemma")
    )

    assert qwen_limit == 384_000
    assert gemma_limit == 192_000


def test_stream_repetition_detector_catches_unfinished_schema_fragment_loop():
    fragment = "_and_remedy_description_and_remedy_type_and_remedy_value"
    assert _stream_is_repeating('{"finding"' + fragment * 60)


def test_stream_repetition_detector_allows_repeated_valid_schema_items():
    repeated_item = {
        "finding_id": "shared-control-note",
        "problem": "The same control cohort legitimately applies to every endpoint.",
        "evidence_refs": ["table-1"],
        "correction": (
            "Retain the shared cohort reference while evaluating each endpoint "
            "against its distinct estimate and uncertainty interval."
        ),
    }
    structured = json.dumps({"findings": [repeated_item] * 12})
    assert len(structured) > 2_048
    assert not _stream_is_repeating(structured)


def test_stream_repetition_detector_allows_interleaved_unique_evidence():
    boilerplate = "The reported estimate requires independent external validation."
    progressing = "\n".join(
        f"{boilerplate}\nDistinct evidence record {index} changes the assessment."
        for index in range(40)
    )
    assert len(progressing) > 2_048
    assert not _stream_is_repeating(progressing)


def test_stream_repetition_guard_is_independent_of_transport_chunking():
    repeated = "_and_remedy_description_and_remedy_type_and_remedy_value" * 100

    one_chunk = _StreamRepetitionGuard()
    assert one_chunk.feed(repeated)

    small_chunks = _StreamRepetitionGuard()
    detected = any(
        small_chunks.feed(repeated[offset : offset + 8])
        for offset in range(0, len(repeated), 8)
    )
    assert detected


def test_stream_repetition_guard_allows_valid_collection_regardless_of_chunking():
    item = {
        "finding_id": "same-finding",
        "problem": "The same documented qualifier applies to every endpoint.",
        "correction": "Retain it because each endpoint still has distinct evidence.",
    }
    value = json.dumps({"findings": [item] * 20})

    one_chunk = _StreamRepetitionGuard()
    assert not one_chunk.feed(value)

    small_chunks = _StreamRepetitionGuard()
    assert not any(
        small_chunks.feed(value[offset : offset + 8])
        for offset in range(0, len(value), 8)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_structured_call_honors_cancel_event_during_delayed_request(streaming):
    request_started = asyncio.Event()

    async def handler(request: httpx.Request):
        del request
        request_started.set()
        await asyncio.sleep(5)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"too late"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    cancel_event = threading.Event()
    task = asyncio.create_task(
        request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={"question": "cancel this"},
            output_type=Answer,
            temperature=0.2,
            max_tokens=100,
            timeout=10,
            repair_attempts=0,
            on_visible_text=(lambda _: None) if streaming else None,
            cancel_event=cancel_event,
            transport=httpx.MockTransport(handler),
        )
    )
    await asyncio.wait_for(request_started.wait(), timeout=1)
    started = asyncio.get_running_loop().time()
    cancel_event.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=0.75)

    assert asyncio.get_running_loop().time() - started < 0.6


@pytest.mark.asyncio
async def test_native_json_schema_request_and_validation():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"ok"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        transport=httpx.MockTransport(handler),
    )
    assert result.value == "ok"
    assert seen["response_format"]["type"] == "json_schema"
    assert seen["response_format"]["json_schema"]["strict"] is True
    assert "chat_template_kwargs" not in seen


@pytest.mark.asyncio
async def test_prompt_schema_preserves_maximum_thinking_without_client_ceiling():
    seen = {}
    endpoint = ModelEndpoint(
        base_url="http://model.invalid/v1",
        model="reasoning-model",
        api_key="",
        max_tokens=None,
        temperature=0.2,
        top_p=0.9,
        enable_thinking=None,
        native_json_schema=False,
    )

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "private reasoning must be ignored",
                            "content": '{"value":"maximal"}',
                        },
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        endpoint,
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=None,
        timeout=2,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "maximal"
    assert "max_tokens" not in seen
    assert "response_format" not in seen
    assert "chat_template_kwargs" not in seen
    assert "STRUCTURED FINAL OUTPUT REQUIREMENT" in seen["messages"][0]["content"]
    assert '"value"' in seen["messages"][0]["content"]


@pytest.mark.asyncio
async def test_prompt_schema_honors_explicit_client_ceiling():
    seen = {}
    endpoint = replace(
        _endpoint(), max_tokens=None, native_json_schema=False, enable_thinking=None
    )

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"bounded"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        endpoint,
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "bounded"
    assert seen["max_tokens"] == 100


@pytest.mark.asyncio
async def test_stream_ignores_reasoning_content_and_exposes_only_final_json():
    visible: list[str] = []
    endpoint = ModelEndpoint(
        base_url="http://model.invalid/v1",
        model="reasoning-model",
        api_key="",
        max_tokens=None,
        temperature=0.2,
        top_p=0.9,
        enable_thinking=True,
        native_json_schema=False,
    )

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["stream"] is True
        lines = [
            'data: {"choices":[{"delta":{"reasoning_content":"private"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"safe\\"}"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        endpoint,
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "safe"
    assert "".join(visible) == '{"value":"safe"}'


@pytest.mark.asyncio
async def test_one_repair_attempt_is_bounded():
    calls = 0
    token_budgets = []
    repair_instructions = []

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        request_body = json.loads(request.content)
        token_budgets.append(request_body["max_tokens"])
        if calls == 2:
            repaired_payload = json.loads(request_body["messages"][1]["content"])
            repair_instructions.append(repaired_payload["repair_instruction"])
        content = '{"wrong":"shape"}' if calls == 1 else '{"value":"fixed"}'
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        transport=httpx.MockTransport(handler),
    )
    assert result.value == "fixed"
    assert calls == 2
    assert token_budgets == [100, 200]
    assert "VALIDATION ERROR" in repair_instructions[0]
    assert "value" in repair_instructions[0]
    assert "Field required" in repair_instructions[0]


@pytest.mark.asyncio
async def test_structured_call_has_total_attempt_timeout():
    async def handler(request: httpx.Request):
        del request
        await asyncio.sleep(0.2)
        return httpx.Response(200, json={})

    with pytest.raises(TimeoutError):
        await request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={"question": "test"},
            output_type=Answer,
            temperature=0.2,
            max_tokens=100,
            timeout=0.01,
            repair_attempts=0,
            transport=httpx.MockTransport(handler),
        )


@pytest.mark.asyncio
async def test_transient_transport_error_is_retried_once():
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadError("connection reset", request=request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"recovered"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status_code", [500, 503])
async def test_transient_model_restart_status_is_retried(
    monkeypatch, streaming, status_code
):
    calls = 0

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("scientific_agent.structured_client.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(status_code, request=request)
        if streaming:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines) + "\n\n")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"recovered"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=(lambda _: None) if streaming else None,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_capacity_rejection_waits_beyond_old_three_attempt_limit(
    monkeypatch, streaming
):
    calls = 0
    visible: list[str] = []

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("scientific_agent.structured_client.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls <= 4:
            return httpx.Response(429, request=request)
        if streaming:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"admitted\\"}"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines) + "\n\n")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"admitted"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=visible.append if streaming else None,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "admitted"
    assert calls == 5
    if streaming:
        rendered = "".join(visible)
        assert "waiting for local model capacity after HTTP 429" in rendered
        assert "run remains cancellable" in rendered


@pytest.mark.asyncio
async def test_capacity_wait_budget_exhaustion_fails_closed(monkeypatch):
    calls = 0
    endpoint = replace(_endpoint(), capacity_wait_seconds=9)

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr("scientific_agent.structured_client.asyncio.sleep", no_sleep)

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(503, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        await request_structured(
            endpoint,
            system_prompt="Return an answer.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            timeout=2,
            repair_attempts=0,
            transport=httpx.MockTransport(handler),
        )

    assert calls == 2


@pytest.mark.asyncio
async def test_invalid_request_does_not_enter_capacity_wait():
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request)

    with pytest.raises(httpx.HTTPStatusError):
        await request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            timeout=2,
            repair_attempts=0,
            transport=httpx.MockTransport(handler),
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_capacity_backoff_is_immediately_cancellable():
    cancel_event = threading.Event()
    visible: list[str] = []

    def record(text: str) -> None:
        visible.append(text)
        if "waiting for local model capacity" in text:
            cancel_event.set()

    def handler(request: httpx.Request):
        return httpx.Response(429, request=request)

    with pytest.raises(asyncio.CancelledError):
        await request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            timeout=2,
            repair_attempts=0,
            on_visible_text=record,
            cancel_event=cancel_event,
            transport=httpx.MockTransport(handler),
        )

    assert "run remains cancellable" in "".join(visible)


@pytest.mark.asyncio
async def test_visual_inputs_precede_the_audit_payload(tmp_path: Path):
    image = tmp_path / "figure.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": '{"value":"audited"}'},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    result = await request_structured(
        _endpoint(),
        system_prompt="Audit the actual figure.",
        payload={"display_id": "effect-plot"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        image_paths=(image,),
        transport=httpx.MockTransport(handler),
    )

    content = seen["messages"][1]["content"]
    assert result.value == "audited"
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[-1]["type"] == "text"
    assert json.loads(content[-1]["text"])["display_id"] == "effect-plot"


@pytest.mark.asyncio
async def test_visual_inputs_reject_unsafe_or_oversized_files(tmp_path: Path):
    unsafe = tmp_path / "figure.svg"
    unsafe.write_text("<svg/>", encoding="utf-8")
    with pytest.raises(ValueError, match="PNG, JPEG, or WebP"):
        await request_structured(
            _endpoint(),
            system_prompt="Audit.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            max_tokens=100,
            timeout=2,
            image_paths=(unsafe,),
            transport=httpx.MockTransport(lambda _: httpx.Response(500)),
        )


@pytest.mark.asyncio
async def test_structured_call_streams_visible_content():
    chunks: list[str] = []

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["stream"] is True
        lines = [
            'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"st"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"reamed\\"}"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=chunks.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "streamed"
    assert len(chunks) == 2
    assert "".join(chunks) == '{"value":"streamed"}'


@pytest.mark.asyncio
async def test_schema_valid_final_channel_closes_stream_without_done_event():
    visible: list[str] = []

    def handler(request: httpx.Request):
        assert json.loads(request.content)["stream"] is True
        lines = [
            'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"complete\\"}"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"unexpected trailing output"},"finish_reason":null}]}',
        ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "complete"
    assert "".join(visible) == '{"value":"complete"}'


@pytest.mark.asyncio
async def test_multiple_top_level_json_values_trigger_one_bounded_repair():
    calls = 0
    visible: list[str] = []

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["stream"] is True
        if calls == 1:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"reviews\\":[]}{\\"reviews\\":[]}"},"finish_reason":null}]}',
            ]
        else:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2
    assert "retrying once" in "".join(visible)


@pytest.mark.asyncio
async def test_complete_schema_invalid_json_closes_stream_and_repairs_without_done():
    calls = 0
    visible: list[str] = []

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["stream"] is True
        if calls == 1:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"reviews\\":[]}"},"finish_reason":null}]}',
                'data: {"choices":[{"delta":{"content":"should not be consumed"},"finish_reason":null}]}',
            ]
        else:
            assert body["messages"][1]["content"]
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"repaired\\"}"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "repaired"
    assert calls == 2
    assert "rejected a schema-invalid final value" in "".join(visible)


@pytest.mark.asyncio
async def test_repetitive_stream_uses_bounded_schema_repair_attempt():
    calls = 0
    repeated = "_and_remedy_description_and_remedy_type_and_remedy_value"

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["stream"] is True
        if calls == 1:
            lines = [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"content": repeated},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                for _ in range(80)
            ]
        else:
            repaired_payload = json.loads(body["messages"][1]["content"])
            assert "invalid_previous_output" not in repaired_payload
            assert "fresh independent" in repaired_payload["retry_instruction"]
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}'
            ]
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=lambda _: None,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_repetitive_private_reasoning_is_killed_and_retried_without_exposure():
    calls = 0
    visible: list[str] = []
    repeated = "The same hidden verification sentence repeats without progress. "

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        body = json.loads(request.content)
        assert body["stream"] is True
        if calls == 1:
            lines = [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning_content": repeated},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
                for _ in range(80)
            ]
        else:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}'
            ]
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={"question": "test"},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2
    rendered = "".join(visible)
    assert "stopped a repetitive model stream" in rendered
    assert rendered.endswith('{"value":"recovered"}')
    assert "hidden verification" not in rendered


@pytest.mark.asyncio
async def test_single_large_repetitive_reasoning_chunk_is_killed_and_retried():
    calls = 0
    repeated = "The hidden reviewer repeats this same sentence without progress. "

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["stream"] is True
        reasoning = repeated * 80
        lines = (
            [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning_content": reasoning},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            ]
            if calls == 1
            else [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}'
            ]
        )
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=lambda _: None,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2


@pytest.mark.asyncio
async def test_repetitive_reasoning_without_repair_fails_closed_without_retry_claim():
    visible: list[str] = []
    private = "This private sentence repeats because generation made no progress. "

    def handler(request: httpx.Request):
        assert json.loads(request.content)["stream"] is True
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {"reasoning_content": private * 80},
                            "finish_reason": None,
                        }
                    ]
                }
            ),
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    with pytest.raises(RuntimeError) as raised:
        await request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            max_tokens=100,
            timeout=2,
            repair_attempts=0,
            on_visible_text=visible.append,
            transport=httpx.MockTransport(handler),
        )

    rendered = "".join(visible)
    assert "repair allowance is exhausted" in rendered
    assert "retrying" not in rendered
    assert "private sentence" not in rendered
    assert "private sentence" not in str(raised.value)


@pytest.mark.asyncio
async def test_long_progressing_private_reasoning_is_not_mistaken_for_a_loop():
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        assert json.loads(request.content)["stream"] is True
        lines = [
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "delta": {
                                "reasoning_content": (
                                    f"Review step {index} evaluates distinct evidence "
                                    f"record {index * 17}. "
                                )
                            },
                            "finish_reason": None,
                        }
                    ]
                }
            )
            for index in range(64)
        ]
        lines.extend(
            [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"valid\\"}"},"finish_reason":"stop"}]}',
                "data: [DONE]",
            ]
        )
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=1,
        on_visible_text=lambda _: None,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "valid"
    assert calls == 1


@pytest.mark.asyncio
async def test_gemma_private_reasoning_without_final_progress_is_retried():
    calls = 0
    visible: list[str] = []
    progressing = " ".join(
        f"independent-evidence-record-{index}" for index in range(8_000)
    )

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls == 1:
            lines = [
                "data: "
                + json.dumps(
                    {
                        "choices": [
                            {
                                "delta": {"reasoning_content": progressing},
                                "finish_reason": None,
                            }
                        ]
                    }
                )
            ]
        else:
            lines = [
                'data: {"choices":[{"delta":{"content":"{\\"value\\":\\"recovered\\"}"},"finish_reason":"stop"}]}'
            ]
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        replace(_endpoint(), model="s8-gemma", max_tokens=None),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=None,
        max_private_reasoning_bytes_without_final=1024,
        timeout=2,
        repair_attempts=1,
        on_visible_text=visible.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "recovered"
    assert calls == 2
    assert "Gemma stream with no final-channel progress" in "".join(visible)


@pytest.mark.asyncio
async def test_private_reasoning_override_must_be_positive():
    with pytest.raises(
        ValueError, match="max_private_reasoning_bytes_without_final must be positive"
    ):
        await request_structured(
            _endpoint(),
            system_prompt="Return an answer.",
            payload={},
            output_type=Answer,
            temperature=0.2,
            max_private_reasoning_bytes_without_final=0,
            timeout=2,
        )


@pytest.mark.asyncio
async def test_structured_stream_suppresses_qwen_reasoning_prefix():
    chunks: list[str] = []

    def handler(request: httpx.Request):
        assert json.loads(request.content)["stream"] is True
        lines = [
            'data: {"choices":[{"delta":{"content":"private plan"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"</thi"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"nk>\\n{\\"value\\":\\"safe"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{"content":"\\"}"},"finish_reason":"stop"}]}',
            "data: [DONE]",
        ]
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n")

    result = await request_structured(
        _endpoint(),
        system_prompt="Return an answer.",
        payload={},
        output_type=Answer,
        temperature=0.2,
        max_tokens=100,
        timeout=2,
        repair_attempts=0,
        on_visible_text=chunks.append,
        transport=httpx.MockTransport(handler),
    )

    assert result.value == "safe"
    assert "private" not in "".join(chunks)
    assert "think" not in "".join(chunks)
    assert "".join(chunks) == '\n{"value":"safe"}'
