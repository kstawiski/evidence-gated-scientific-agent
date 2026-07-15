"""Strict structured calls to OpenAI-compatible local model endpoints.

ADK remains the workflow and tool runtime. Schema-only workflow nodes use native
JSON schema when the endpoint supports it alongside reasoning, or prompt-level
schema instructions followed by the same local Pydantic validation otherwise.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Callable, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import ModelEndpoint
from .visibility import VisibleTextFilter, strip_reasoning_envelope


T = TypeVar("T", bound=BaseModel)
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}
MAX_IMAGE_COUNT = 5
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 20 * 1024 * 1024
TRANSIENT_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
TRANSPORT_ATTEMPTS = 3
TRANSPORT_BACKOFF_SECONDS = (5.0, 15.0)


class _RepetitiveStreamError(RuntimeError):
    """A streamed sample entered a deterministic no-progress loop."""


class _StructuredStreamComplete(RuntimeError):
    """The final channel already contains one complete structured value."""


def _has_complete_top_level_json_value(value: str) -> bool:
    """Return whether visible output starts with one complete JSON value."""

    position = 0
    while position < len(value) and value[position].isspace():
        position += 1
    if position >= len(value):
        return False
    try:
        json.JSONDecoder().raw_decode(value, position)
    except ValueError:
        return False
    return True


def _has_multiple_top_level_json_values(value: str) -> bool:
    """Return whether visible output contains two complete JSON values.

    Structured calls require exactly one top-level value. Some local models loop
    by emitting a sequence of individually valid but schema-invalid objects. That
    pattern is deterministic protocol failure even when the objects are not byte
    identical, and should trigger the bounded repair path promptly.
    """

    decoder = json.JSONDecoder()
    position = 0
    decoded = 0
    while position < len(value):
        while position < len(value) and value[position].isspace():
            position += 1
        if position >= len(value):
            return False
        try:
            _, position = decoder.raw_decode(value, position)
        except ValueError:
            return False
        decoded += 1
        if decoded >= 2:
            return True
    return False


def _stream_repetition_signature(value: str) -> str | None:
    """Return a content-free signature for sustained contiguous repetition."""

    tail = value[-12_000:]
    if len(tail) < 2_048:
        return None
    # Some local endpoints loop inside an unfinished JSON key/value fragment
    # without emitting whitespace or sentence boundaries. Require at least 2 KiB
    # of contiguous repetition so ordinary repeated schema fields are not loops.
    compact_tail = tail[-4096:]
    looks_like_structured_collection = (
        tail.lstrip().startswith(("{", "["))
        and tail.count("{") >= 4
        and tail.count("}") >= 3
    )
    if not looks_like_structured_collection:
        for period in range(16, 1_025):
            repetitions = max(4, (2_048 + period - 1) // period)
            repeated_chars = period * repetitions
            if repeated_chars > len(compact_tail):
                continue
            block = compact_tail[-period:]
            if len(set(block)) >= 4 and compact_tail.endswith(block * repetitions):
                return f"periodic:{period}"
    sentences = [
        " ".join(item.casefold().split())
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", tail)
        if item.strip()
    ]
    if not sentences:
        return None
    repeated_sentence = sentences[-1]
    if (
        len(repeated_sentence) < 24
        or len(re.findall(r"\b[a-z]{2,}\b", repeated_sentence)) < 6
    ):
        return None
    repeated_count = 0
    repeated_chars = 0
    for sentence in reversed(sentences):
        if sentence != repeated_sentence:
            break
        repeated_count += 1
        repeated_chars += len(sentence) + 1
    if repeated_count >= 8 and repeated_chars >= 2_048:
        digest = hashlib.sha256(repeated_sentence.encode("utf-8")).hexdigest()[:16]
        return f"sentence:{digest}"
    return None


def _stream_is_repeating(value: str) -> bool:
    """Detect a sustained no-progress suffix without judging model content."""

    return _stream_repetition_signature(value) is not None


class _StreamRepetitionGuard:
    """Confirm one repeat signature at consecutive byte-based checkpoints."""

    def __init__(self) -> None:
        self._tail = ""
        self._bytes = 0
        self._next_checkpoint = 2_048
        self._previous_signature: str | None = None

    def feed(self, chunk: str) -> bool:
        pending: list[str] = []
        for character in chunk:
            pending.append(character)
            self._bytes += len(character.encode("utf-8"))
            if self._bytes < self._next_checkpoint:
                continue
            self._tail = (self._tail + "".join(pending))[-12_000:]
            pending.clear()
            signature = _stream_repetition_signature(self._tail)
            if signature is not None and signature == self._previous_signature:
                return True
            self._previous_signature = signature
            self._next_checkpoint += 1_024
        if pending:
            self._tail = (self._tail + "".join(pending))[-12_000:]
        return False


async def _await_cancellable(
    awaitable, timeout: float, cancel_event: threading.Event | None
):
    task = asyncio.create_task(awaitable)
    try:
        async with asyncio.timeout(timeout):
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                    raise asyncio.CancelledError
                done, _ = await asyncio.wait({task}, timeout=0.25)
                if done:
                    return await task
    finally:
        if not task.done():
            task.cancel()


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


def _user_content(payload: Any, image_paths: tuple[Path, ...]) -> str | list[dict]:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    if not image_paths:
        return text
    if len(image_paths) > MAX_IMAGE_COUNT:
        raise ValueError(f"at most {MAX_IMAGE_COUNT} visual inputs are supported")
    blocks: list[dict] = []
    total_bytes = 0
    for path in image_paths:
        resolved = path.resolve()
        media_type = IMAGE_MEDIA_TYPES.get(resolved.suffix.lower())
        if media_type is None:
            raise ValueError("visual inputs must be PNG, JPEG, or WebP")
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("visual input must be a regular file")
        size = resolved.stat().st_size
        if size < 1 or size > MAX_IMAGE_BYTES:
            raise ValueError("visual input exceeds the per-image size limit")
        total_bytes += size
        if total_bytes > MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("visual inputs exceed the total size limit")
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        blocks.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{encoded}"},
            }
        )
    blocks.append({"type": "text", "text": text})
    return blocks


async def request_structured(
    endpoint: ModelEndpoint,
    *,
    system_prompt: str,
    payload: Any,
    output_type: type[T],
    temperature: float,
    max_tokens: int | None = None,
    timeout: float,
    enable_thinking: bool | None = None,
    repair_attempts: int = 1,
    image_paths: tuple[Path, ...] = (),
    on_visible_text: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
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
    schema = output_type.model_json_schema()
    thinking_setting = (
        endpoint.enable_thinking if enable_thinking is None else enable_thinking
    )
    request_system_prompt = system_prompt
    effective_timeout = endpoint.request_timeout_seconds or timeout
    if not endpoint.native_json_schema:
        request_system_prompt = (
            f"{system_prompt}\n\n"
            "STRUCTURED FINAL OUTPUT REQUIREMENT:\n"
            "After completing your private reasoning, place exactly one JSON object "
            "matching the following JSON Schema in the final answer channel. Do not "
            "wrap it in Markdown or include commentary outside the JSON.\n"
            f"{json.dumps(schema, ensure_ascii=False, separators=(',', ':'))}"
        )

    async with httpx.AsyncClient(
        timeout=effective_timeout, transport=transport
    ) as client:
        for attempt in range(repair_attempts + 1):
            repair_notice_emitted = False
            repetition_detected = False
            request_body = {
                "model": endpoint.model,
                "messages": [
                    {"role": "system", "content": request_system_prompt},
                    {
                        "role": "user",
                        "content": _user_content(user_payload, image_paths),
                    },
                ],
                "temperature": temperature,
                "top_p": endpoint.top_p,
            }
            if thinking_setting is not None:
                request_body["chat_template_kwargs"] = {
                    "enable_thinking": thinking_setting
                }
            if endpoint.max_tokens is not None:
                request_body["max_tokens"] = endpoint.max_tokens
                if max_tokens is not None:
                    request_body["max_tokens"] = min(
                        endpoint.max_tokens,
                        max_tokens * (attempt + 1),
                    )
            if endpoint.native_json_schema:
                request_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                }
            headers = (
                {"Authorization": f"Bearer {endpoint.api_key}"}
                if endpoint.api_key
                else None
            )
            if on_visible_text is None:
                for transport_attempt in range(TRANSPORT_ATTEMPTS):
                    try:
                        response = await _await_cancellable(
                            client.post(url, json=request_body, headers=headers),
                            effective_timeout,
                            cancel_event,
                        )
                        response.raise_for_status()
                        break
                    except httpx.TransportError:
                        if transport_attempt == TRANSPORT_ATTEMPTS - 1:
                            raise
                        await asyncio.sleep(
                            TRANSPORT_BACKOFF_SECONDS[transport_attempt]
                        )
                    except httpx.HTTPStatusError as exc:
                        if (
                            exc.response.status_code not in TRANSIENT_HTTP_STATUS_CODES
                            or transport_attempt == TRANSPORT_ATTEMPTS - 1
                        ):
                            raise
                        await asyncio.sleep(
                            TRANSPORT_BACKOFF_SECONDS[transport_attempt]
                        )
                content, last_finish_reason = _message_content(response.json())
                content = strip_reasoning_envelope(content)
            else:
                chunks: list[str] = []
                visible_boundary = VisibleTextFilter()
                reasoning_guard = _StreamRepetitionGuard()
                content_guard = _StreamRepetitionGuard()
                stream_started = False

                async def consume_stream() -> None:
                    nonlocal last_finish_reason
                    nonlocal stream_started
                    async with client.stream(
                        "POST",
                        url,
                        json={**request_body, "stream": True},
                        headers=headers,
                    ) as response:
                        response.raise_for_status()
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if not data or data == "[DONE]":
                                continue
                            event = json.loads(data)
                            choices = event.get("choices")
                            if not isinstance(choices, list) or not choices:
                                continue
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                continue
                            finish_reason = choice.get("finish_reason")
                            if isinstance(finish_reason, str):
                                last_finish_reason = finish_reason
                            delta = choice.get("delta")
                            if not isinstance(delta, dict):
                                continue
                            reasoning_chunk = delta.get("reasoning_content")
                            if isinstance(reasoning_chunk, str) and reasoning_chunk:
                                stream_started = True
                                # Inspect only a bounded in-memory suffix. Never
                                # persist or expose private reasoning text.
                                if reasoning_guard.feed(reasoning_chunk):
                                    raise _RepetitiveStreamError(
                                        "model reasoning stream entered a repetitive "
                                        "output loop"
                                    )
                            chunk = delta.get("content")
                            if not isinstance(chunk, str) or not chunk:
                                continue
                            stream_started = True
                            chunks.append(chunk)
                            if content_guard.feed(chunk):
                                raise _RepetitiveStreamError(
                                    "model stream entered a repetitive output loop"
                                )
                            visible_chunk = visible_boundary.feed(chunk)
                            if visible_chunk:
                                try:
                                    on_visible_text(visible_chunk)
                                except Exception:
                                    # Monitoring is observational and cannot alter a result.
                                    pass
                            if chunk.rstrip().endswith(("}", "]")):
                                candidate = strip_reasoning_envelope("".join(chunks))
                                try:
                                    output_type.model_validate_json(candidate)
                                except (ValidationError, ValueError):
                                    if _has_multiple_top_level_json_values(candidate):
                                        raise _RepetitiveStreamError(
                                            "model stream emitted multiple top-level "
                                            "JSON values"
                                        )
                                    if _has_complete_top_level_json_value(candidate):
                                        # Appending a correction would create a second
                                        # top-level value and violate the exact-one
                                        # contract. Close now so the bounded schema
                                        # repair can start even if the gateway omits
                                        # or delays its transport terminator.
                                        last_finish_reason = "schema_invalid"
                                        raise _StructuredStreamComplete
                                else:
                                    last_finish_reason = "schema_complete"
                                    raise _StructuredStreamComplete

                for transport_attempt in range(TRANSPORT_ATTEMPTS):
                    try:
                        await _await_cancellable(
                            consume_stream(), effective_timeout, cancel_event
                        )
                        break
                    except _StructuredStreamComplete:
                        # Some compatible gateways do not promptly emit [DONE]
                        # after a complete final-channel JSON value. Closing the
                        # stream here preserves all preceding reasoning and avoids
                        # waiting for a redundant transport terminator.
                        break
                    except httpx.TransportError:
                        if (
                            stream_started
                            or transport_attempt == TRANSPORT_ATTEMPTS - 1
                        ):
                            raise
                        await asyncio.sleep(
                            TRANSPORT_BACKOFF_SECONDS[transport_attempt]
                        )
                    except httpx.HTTPStatusError as exc:
                        if (
                            stream_started
                            or exc.response.status_code
                            not in TRANSIENT_HTTP_STATUS_CODES
                            or transport_attempt == TRANSPORT_ATTEMPTS - 1
                        ):
                            raise
                        await asyncio.sleep(
                            TRANSPORT_BACKOFF_SECONDS[transport_attempt]
                        )
                    except _RepetitiveStreamError:
                        # The current sample is unusable, but a bounded schema
                        # repair attempt may still recover without failing the
                        # entire scientific workflow.
                        try:
                            notice = (
                                "retrying once with a schema-repair request"
                                if attempt < repair_attempts
                                else "the bounded repair allowance is exhausted"
                            )
                            on_visible_text(
                                "\n[Evidence Bench stopped a repetitive model stream; "
                                f"{notice}.]\n"
                            )
                            repair_notice_emitted = True
                        except Exception:
                            pass
                        repetition_detected = True
                        break
                visible_tail = visible_boundary.finish()
                if visible_tail:
                    try:
                        on_visible_text(visible_tail)
                    except Exception:
                        pass
                content = "".join(chunks)
                content = strip_reasoning_envelope(content)
            last_content = content
            try:
                return output_type.model_validate_json(content)
            except ValidationError as exc:
                last_error = str(exc)[:4000]
            except ValueError as exc:
                last_error = str(exc)[:4000]

            if attempt < repair_attempts:
                if on_visible_text is not None and not repair_notice_emitted:
                    try:
                        on_visible_text(
                            "\n[Evidence Bench rejected a schema-invalid final "
                            "value; retrying once with a bounded repair request.]\n"
                        )
                    except Exception:
                        pass
                if repetition_detected:
                    user_payload = {
                        "original_input": original_input,
                        "retry_instruction": (
                            "The preceding sample was terminated by deterministic "
                            "no-progress detection. Start a fresh independent "
                            f"reasoning sample and return one complete {schema_name} "
                            "value; emit JSON only. Do not continue or imitate the "
                            "terminated sample."
                        ),
                    }
                else:
                    user_payload = {
                        "original_input": original_input,
                        "invalid_previous_output": last_content[-8000:],
                        "repair_instruction": (
                            f"Return one complete value matching {schema_name}. "
                            "Correct the schema violation shown below; emit JSON only.\n"
                            f"VALIDATION ERROR:\n{last_error}"
                        ),
                    }

    raise RuntimeError(
        f"endpoint produced no valid {schema_name} after {repair_attempts + 1} "
        f"attempt(s); finish_reason={last_finish_reason!r}; "
        f"content_chars={len(last_content)}; validation={last_error}"
    )
