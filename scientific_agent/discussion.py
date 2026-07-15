"""Evidence-bounded Gemma discussion of a completed scientific report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .config import Settings
from .prompts import REPORT_DISCUSSION
from .schemas import ReportDiscussionResponse
from .structured_client import request_structured


DISCUSSION_CONTEXT_FILES = (
    "scientific_report.json",
    "deterministic_validation.json",
    "gemma_review.json",
    "reference_manifest.json",
    "display_manifest.json",
)
MAX_CONTEXT_BYTES = 768 * 1024
MAX_HISTORY_MESSAGES = 16


def _discussion_context(run_root: Path) -> dict[str, Any]:
    root = run_root.resolve(strict=True)
    context: dict[str, Any] = {}
    total = 0
    for name in DISCUSSION_CONTEXT_FILES:
        path = (root / name).resolve()
        if path.parent != root or not path.is_file() or path.is_symlink():
            continue
        size = path.stat().st_size
        if size > MAX_CONTEXT_BYTES or total + size > MAX_CONTEXT_BYTES:
            raise ValueError("report discussion context exceeds its bounded size")
        context[name] = json.loads(path.read_text(encoding="utf-8"))
        total += size
    if "scientific_report.json" not in context:
        raise ValueError("the run has no completed scientific report")
    return context


async def discuss_report(
    settings: Settings,
    run_root: Path,
    history: Sequence[dict[str, Any]],
    message: str,
) -> ReportDiscussionResponse:
    """Ask configured Gemma to explain one immutable report record."""

    bounded_history = [
        {
            "role": item.get("role"),
            "content": str(item.get("content", ""))[:12_000],
            "suggested_revision_prompt": item.get("suggested_revision_prompt"),
        }
        for item in history[-MAX_HISTORY_MESSAGES:]
        if item.get("status") == "complete"
        and item.get("role") in {"user", "assistant"}
    ]
    return await request_structured(
        settings.gemma,
        system_prompt=REPORT_DISCUSSION,
        payload={
            "immutable_run_record": _discussion_context(run_root),
            "discussion_history": bounded_history,
            "user_message": message,
        },
        output_type=ReportDiscussionResponse,
        temperature=settings.gemma.temperature,
        timeout=300,
        repair_attempts=1,
        # Activate the streaming repetition/duplicate-object safeguards while
        # intentionally discarding in-progress structured JSON.
        on_visible_text=lambda _: None,
    )
