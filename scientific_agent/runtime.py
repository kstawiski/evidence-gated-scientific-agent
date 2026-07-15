"""Small ADK runner facade that returns typed final outputs."""

from __future__ import annotations

import json
import asyncio
import threading
import uuid
from typing import Any, Callable, TypeVar

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from pydantic import BaseModel

from .visibility import VisibleTextFilter, strip_reasoning_envelope


T = TypeVar("T", bound=BaseModel)


def _visible_text(parts) -> str:
    return "".join(
        part.text or ""
        for part in parts
        if getattr(part, "text", None) and not getattr(part, "thought", False)
    )


def _prompt_text(payload: Any) -> str:
    if isinstance(payload, BaseModel):
        return payload.model_dump_json()
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, default=str)


async def run_text(
    root_agent,
    payload: Any,
    *,
    max_chars: int = 60_000,
    on_visible_text: Callable[[str], None] | None = None,
    on_model_turn: Callable[[], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> str:
    """Run a tool-capable ADK agent and return only its visible final response."""

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="evidence-gated-scientific-agent",
        session_service=session_service,
        auto_create_session=True,
    )
    message = types.Content(role="user", parts=[types.Part(text=_prompt_text(payload))])
    final_texts: list[str] = []
    partial_boundary = VisibleTextFilter()
    partial_stream_active = False
    partial_model_turn_active = False
    async for event in runner.run_async(
        user_id="scientist", session_id=uuid.uuid4().hex, new_message=message
    ):
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError
        content = getattr(event, "content", None)
        if content is not None and getattr(content, "role", None) == "model":
            if getattr(event, "partial", False):
                if not partial_model_turn_active and on_model_turn is not None:
                    on_model_turn()
                partial_model_turn_active = True
            else:
                if not partial_model_turn_active and on_model_turn is not None:
                    on_model_turn()
                partial_model_turn_active = False
        if event.content and event.content.parts:
            raw_visible = _visible_text(event.content.parts)
            if raw_visible and on_visible_text is not None:
                if getattr(event, "partial", False):
                    partial_stream_active = True
                    safe_visible = partial_boundary.feed(raw_visible)
                else:
                    # Non-partial ADK events are complete model responses. Each
                    # tool turn may contain a fresh Qwen reasoning prefix, so
                    # sanitize every response independently.
                    safe_visible = strip_reasoning_envelope(raw_visible)
            else:
                safe_visible = ""
            if safe_visible and on_visible_text is not None:
                try:
                    on_visible_text(safe_visible)
                except Exception:
                    # Monitoring visible output is observational only.
                    pass
            if raw_visible and (
                not hasattr(event, "is_final_response") or event.is_final_response()
            ):
                final_texts.append(strip_reasoning_envelope(raw_visible))
            if partial_stream_active and (
                getattr(event, "turn_complete", False)
                or getattr(event, "finish_reason", None) is not None
            ):
                tail = partial_boundary.finish()
                partial_boundary = VisibleTextFilter()
                partial_stream_active = False
                if tail and on_visible_text is not None:
                    try:
                        on_visible_text(tail)
                    except Exception:
                        pass
        if getattr(event, "turn_complete", False):
            partial_model_turn_active = False
    if on_visible_text is not None and partial_stream_active:
        tail = partial_boundary.finish()
        if tail:
            try:
                on_visible_text(tail)
            except Exception:
                pass
    if not final_texts:
        raise RuntimeError("agent produced no visible final response")
    result = final_texts[-1]
    if len(result) > max_chars:
        raise RuntimeError(f"agent final response exceeds {max_chars} characters")
    return result


async def run_typed(
    root_agent, payload: Any, output_type: type[T], *, repair_attempts: int = 1
) -> T:
    prompt = _prompt_text(payload)
    last_text = ""
    for attempt in range(repair_attempts + 1):
        session_service = InMemorySessionService()
        runner = Runner(
            agent=root_agent,
            app_name="evidence-gated-scientific-agent",
            session_service=session_service,
            auto_create_session=True,
        )
        message = types.Content(role="user", parts=[types.Part(text=prompt)])
        session_id = uuid.uuid4().hex
        outputs: list[Any] = []
        texts: list[str] = []
        async for event in runner.run_async(
            user_id="scientist", session_id=session_id, new_message=message
        ):
            output = getattr(event, "output", None)
            if output is not None:
                outputs.append(output)
            if event.content and event.content.parts:
                text = _visible_text(event.content.parts)
                if text and (
                    not hasattr(event, "is_final_response") or event.is_final_response()
                ):
                    texts.append(text)

        for candidate in reversed(outputs):
            if isinstance(candidate, output_type):
                return candidate
            try:
                return output_type.model_validate(candidate)
            except Exception:
                pass
        for text in reversed(texts):
            last_text = text
            try:
                return output_type.model_validate_json(text)
            except Exception:
                continue
        if attempt < repair_attempts:
            prompt = json.dumps(
                {
                    "original_input": payload.model_dump(mode="json")
                    if isinstance(payload, BaseModel)
                    else payload,
                    "invalid_previous_output": last_text[-12000:],
                    "repair_instruction": (
                        f"Return one complete JSON value matching {output_type.__name__}. "
                        "Do not add prose, markdown, or a reasoning trace."
                    ),
                },
                default=str,
            )
    raise RuntimeError(
        f"agent produced no valid {output_type.__name__} output after "
        f"{repair_attempts + 1} attempt(s)"
    )
