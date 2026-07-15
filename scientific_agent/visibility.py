"""Streaming boundary that keeps model reasoning tags out of live monitoring."""

from __future__ import annotations

import re


_OPEN = "<think>"
_CLOSE = "</think>"
_OPEN_PATTERN = re.compile(re.escape(_OPEN), re.IGNORECASE | re.ASCII)
_CLOSE_PATTERN = re.compile(re.escape(_CLOSE), re.IGNORECASE | re.ASCII)


class VisibleTextFilter:
    """Emit answer text while suppressing tagged or prefix-only reasoning.

    Some OpenAI-compatible Qwen servers omit the opening ``<think>`` token but
    still emit the reasoning text followed by ``</think>``. Until the stream is
    clearly JSON or that closing marker appears, the prefix is buffered. This
    trades a short initial delay for not exposing hidden reasoning in the UI.
    """

    def __init__(self) -> None:
        self._decided = False
        self._in_thought = False
        self._buffer = ""
        self._marker_tail = ""

    def feed(self, text: str) -> str:
        if not text:
            return ""
        if not self._decided:
            self._buffer += text
            closing = _CLOSE_PATTERN.search(self._buffer)
            if closing is not None:
                remainder = self._buffer[closing.end() :]
                self._buffer = ""
                self._decided = True
                return self._feed_visible(remainder)
            stripped = self._buffer.lstrip()
            if stripped.startswith(("{", "[")):
                buffered = self._buffer
                self._buffer = ""
                self._decided = True
                return self._feed_visible(buffered)
            return ""
        return self._feed_visible(text)

    def _feed_visible(self, text: str) -> str:
        data = self._marker_tail + text
        self._marker_tail = ""
        output: list[str] = []
        while data:
            marker = _CLOSE if self._in_thought else _OPEN
            pattern = _CLOSE_PATTERN if self._in_thought else _OPEN_PATTERN
            match = pattern.search(data)
            if match is not None:
                if not self._in_thought:
                    output.append(data[: match.start()])
                data = data[match.end() :]
                self._in_thought = not self._in_thought
                continue
            keep = 0
            for size in range(1, min(len(marker) - 1, len(data)) + 1):
                if data[-size:].lower() == marker[:size]:
                    keep = size
            complete = data[:-keep] if keep else data
            if not self._in_thought:
                output.append(complete)
            self._marker_tail = data[-keep:] if keep else ""
            break
        return "".join(output)

    def finish(self) -> str:
        if not self._decided:
            buffered = self._buffer
            self._buffer = ""
            # An opening marker without a closing marker is incomplete reasoning,
            # not safe answer text. Untagged natural-language output is released
            # only when the response is complete.
            if _OPEN_PATTERN.search(buffered) is not None:
                return ""
            return buffered
        if self._in_thought:
            self._marker_tail = ""
            return ""
        tail = self._marker_tail
        self._marker_tail = ""
        return tail


def strip_reasoning_envelope(text: str) -> str:
    """Return complete visible output using the same policy as live streams."""

    boundary = VisibleTextFilter()
    return boundary.feed(text) + boundary.finish()
