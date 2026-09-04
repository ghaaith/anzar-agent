"""Shared catalog of supported LLM providers and their models.

Single source of truth used by the web UI (/api/providers), the CLI model
toggle, and the Settings page. The first model for each provider is its
default in LLMProvider.create.
"""

from __future__ import annotations

from typing import Optional

SUPPORTED_PROVIDERS: dict[str, dict] = {
    "openrouter": {
        "name": "OpenRouter",
        "models": [
            "openai/gpt-oss-20b:free",
            "poolside/laguna-xs-2.1:free",
            "nvidia/nemotron-3-nano-30b-a3b:free",
            "nvidia/nemotron-3-super-120b-a12b:free",
        ],
    },
    "groq": {
        "name": "Groq",
        "models": [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.6-27b",
        ],
    },
    "gemini": {
        "name": "Gemini",
        "models": [
            "gemini-3.6-flash",
            "gemini-3.1-pro-preview",
            "gemini-3.1-flash-lite",
        ],
    },
    "openai": {
        "name": "OpenAI",
        "models": [
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
        ],
    },
    "anthropic": {
        "name": "Anthropic",
        "models": [
            "claude-opus-5",
            "claude-sonnet-5",
            "claude-opus-4-8",
        ],
    },
    "ollama": {
        "name": "Ollama",
        "models": [
            "llama3.1",
            "codellama",
            "deepseek-coder",
        ],
    },
}


def provider_names() -> list[str]:
    """Provider ids in display order."""
    return list(SUPPORTED_PROVIDERS.keys())


def provider_models(provider: str) -> list[str]:
    """Models available for a provider (empty list if unknown)."""
    info = SUPPORTED_PROVIDERS.get(provider)
    return list(info["models"]) if info else []


def provider_name(provider: str) -> str:
    """Human-friendly provider name."""
    info = SUPPORTED_PROVIDERS.get(provider)
    return info["name"] if info else provider


def default_model(provider: str) -> Optional[str]:
    """The default model for a provider (its first entry), or None."""
    models = provider_models(provider)
    return models[0] if models else None


# Conservative per-provider caps on the *input request* context (tokens).
# These bound the agent's safe input budget: ``ContextBudget`` takes
# ``min(provider_limit, configured)`` and subtracts output/overhead reserves
# and a safety margin. Kept as best-effort guidance for known providers; an
# unknown provider returns ``None`` (no invented cap — the configured default
# bounds the request instead).
PROVIDER_CONTEXT_LIMITS: dict[str, int] = {
    "groq": 60000,
    "openrouter": 32000,
    "openai": 128000,
    "anthropic": 200000,
    "gemini": 1000000,
    "ollama": 8192,
}


def context_limit(provider: str) -> Optional[int]:
    """Best-effort input-request cap (tokens) for a provider, or None when unknown.

    ``None`` means "unknown" — callers must not pretend a limit exists and
    should fall back to the configured default budget.
    """
    return PROVIDER_CONTEXT_LIMITS.get(provider)

