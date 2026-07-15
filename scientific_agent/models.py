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
    selected_max_tokens = endpoint.max_tokens
    if selected_max_tokens is not None and max_tokens is not None:
        selected_max_tokens = min(selected_max_tokens, max_tokens)
    kwargs = dict(
        model=f"openai/{endpoint.model.removeprefix('openai/')}",
        api_base=endpoint.base_url,
        api_key=endpoint.api_key or "no-auth-needed",
        temperature=endpoint.temperature if temperature is None else temperature,
        top_p=endpoint.top_p,
        timeout=endpoint.request_timeout_seconds or timeout,
        max_retries=1,
    )
    if selected_max_tokens is not None:
        kwargs["max_tokens"] = selected_max_tokens
    if endpoint.enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": endpoint.enable_thinking}
        }
    return LiteLlm(**kwargs)


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
