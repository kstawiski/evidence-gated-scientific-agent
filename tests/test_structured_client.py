import json
import asyncio

import httpx
import pytest
from pydantic import BaseModel

from scientific_agent.config import ModelEndpoint
from scientific_agent.structured_client import request_structured


class Answer(BaseModel):
    value: str


def _endpoint():
    return ModelEndpoint(
        base_url="http://model.invalid/v1",
        model="local-model",
        api_key="",
        max_tokens=100,
        temperature=0.2,
        top_p=0.9,
    )


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
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_one_repair_attempt_is_bounded():
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        content = '{"wrong":"shape"}' if calls == 1 else '{"value":"fixed"}'
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}
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
    assert result.value == "fixed"
    assert calls == 2


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
