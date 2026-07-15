from types import SimpleNamespace

import pytest
from google.genai import types

from scientific_agent.runtime import run_text


class _Event:
    def __init__(self, text: str, *, final: bool):
        self.content = types.Content(role="model", parts=[types.Part(text=text)])
        self.partial = None
        self.turn_complete = None
        self.finish_reason = None
        self._final = final

    def is_final_response(self) -> bool:
        return self._final


class _Runner:
    def __init__(self, **kwargs):
        del kwargs

    async def run_async(self, **kwargs):
        del kwargs
        yield _Event("private first turn</think>\nVisible tool preface", final=False)
        yield _Event("private second turn</think>\nFinal visible answer", final=True)


@pytest.mark.asyncio
async def test_each_adk_tool_turn_gets_an_independent_visibility_boundary(monkeypatch):
    monkeypatch.setattr("scientific_agent.runtime.Runner", _Runner)
    streamed: list[str] = []

    result = await run_text(
        SimpleNamespace(name="fake"),
        "test",
        on_visible_text=streamed.append,
    )

    assert result == "\nFinal visible answer"
    assert "".join(streamed) == "\nVisible tool preface\nFinal visible answer"
    assert "private" not in "".join(streamed)
    assert "think" not in "".join(streamed)


@pytest.mark.asyncio
async def test_run_text_reports_each_nonstreaming_model_turn(monkeypatch):
    monkeypatch.setattr("scientific_agent.runtime.Runner", _Runner)
    turns: list[int] = []

    await run_text(
        SimpleNamespace(name="fake"),
        "test",
        on_model_turn=lambda: turns.append(1),
    )

    assert len(turns) == 2


@pytest.mark.asyncio
async def test_run_text_fails_closed_when_turn_controller_raises(monkeypatch):
    monkeypatch.setattr("scientific_agent.runtime.Runner", _Runner)

    def reject_turn():
        raise RuntimeError("research budget exceeded")

    with pytest.raises(RuntimeError, match="research budget exceeded"):
        await run_text(
            SimpleNamespace(name="fake"),
            "test",
            on_model_turn=reject_turn,
        )
