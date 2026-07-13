"""ADK LiteLLM adapters for the two isolated self-hosted endpoints."""

from __future__ import annotations

from google.adk.models.lite_llm import LiteLlm

from .config import ModelEndpoint, Settings


def _model(
    endpoint: ModelEndpoint,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 180,
) -> LiteLlm:
    return LiteLlm(
        model=f"openai/{endpoint.model.removeprefix('openai/')}",
        api_base=endpoint.base_url,
        api_key=endpoint.api_key or "no-auth-needed",
        max_tokens=endpoint.max_tokens if max_tokens is None else max_tokens,
        temperature=endpoint.temperature if temperature is None else temperature,
        top_p=endpoint.top_p,
        timeout=timeout,
        max_retries=1,
    )


def qwen_model(
    settings: Settings,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 180,
) -> LiteLlm:
    return _model(
        settings.qwen,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def gemma_model(
    settings: Settings,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int = 180,
) -> LiteLlm:
    return _model(
        settings.gemma,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
