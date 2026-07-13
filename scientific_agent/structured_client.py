"""Strict structured calls to OpenAI-compatible local model endpoints.

ADK remains the workflow and tool runtime. Schema-only workflow nodes use the
servers' native JSON-schema contract because some third-party ADK graph/model
combinations currently surface a raw Content object instead of the validated
node output.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import ModelEndpoint


T = TypeVar("T", bound=BaseModel)


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return value


def _message_content(response: dict[str, Any]) -> tuple[str, str | None]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("model response has no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError("model response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("model response has no message object")
    content = message.get("content")
    if not isinstance(content, str):
        raise RuntimeError("model response content is not text")
    finish_reason = choice.get("finish_reason")
    return content, finish_reason if isinstance(finish_reason, str) else None


async def request_structured(
    endpoint: ModelEndpoint,
    *,
    system_prompt: str,
    payload: Any,
    output_type: type[T],
    temperature: float,
    max_tokens: int,
    timeout: float,
    enable_thinking: bool = False,
    repair_attempts: int = 1,
    transport: httpx.AsyncBaseTransport | None = None,
) -> T:
    """Request one schema-valid value, with at most one explicit repair call."""

    original_input = _jsonable(payload)
    user_payload: Any = original_input
    last_error = "no response"
    last_finish_reason: str | None = None
    last_content = ""
    url = f"{endpoint.base_url.rstrip('/')}/chat/completions"
    schema_name = output_type.__name__[:64]

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        for attempt in range(repair_attempts + 1):
            request_body = {
                "model": endpoint.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            default=str,
                        ),
                    },
                ],
                "temperature": temperature,
                "top_p": endpoint.top_p,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": enable_thinking},
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": output_type.model_json_schema(),
                    },
                },
            }
            headers = (
                {"Authorization": f"Bearer {endpoint.api_key}"}
                if endpoint.api_key
                else None
            )
            for transport_attempt in range(2):
                try:
                    async with asyncio.timeout(timeout):
                        response = await client.post(
                            url, json=request_body, headers=headers
                        )
                    break
                except httpx.TransportError:
                    if transport_attempt == 1:
                        raise
                    await asyncio.sleep(0.25)
            response.raise_for_status()
            content, last_finish_reason = _message_content(response.json())
            last_content = content
            try:
                return output_type.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc).splitlines()[0]
            except ValueError as exc:
                last_error = str(exc).splitlines()[0]

            if attempt < repair_attempts:
                user_payload = {
                    "original_input": original_input,
                    "invalid_previous_output": last_content[-8000:],
                    "repair_instruction": (
                        f"Return one complete value matching {schema_name}. "
                        "Correct the schema violation; emit JSON only."
                    ),
                }

    raise RuntimeError(
        f"endpoint produced no valid {schema_name} after {repair_attempts + 1} "
        f"attempt(s); finish_reason={last_finish_reason!r}; "
        f"content_chars={len(last_content)}; validation={last_error}"
    )
