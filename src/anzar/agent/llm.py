"""LLM provider abstraction for Anzar agent."""

from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel

from anzar.agent.context import PayloadTooLargeError

logger = logging.getLogger("anzar.agent.llm")


class LLMProvider:
    """Unified interface to create LangChain chat models from provider config."""

    @staticmethod
    def create(
        provider: str,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> BaseChatModel:
        """Create a LangChain chat model for the given provider.

        Args:
            provider: One of "gemini", "openai", "anthropic", "ollama"
            model: Model name (provider-specific default if None)
            api_key: API key (required for cloud providers)
            temperature: Sampling temperature
            max_tokens: Upper bound on generated tokens (prevents runaway responses)
            **kwargs: Extra provider-specific options

        Returns:
            A BaseChatModel instance ready for use.
        """
        provider = provider.lower().strip()

        # Resolve missing model from the catalog so defaults always land on a
        # model that still exists for the provider.
        if not model:
            from anzar.agent.providers import default_model

            model = default_model(provider)

        if provider == "openrouter":
            return LLMProvider._create_openrouter(model, api_key, temperature, max_tokens, **kwargs)
        elif provider == "groq":
            return LLMProvider._create_groq(model, api_key, temperature, max_tokens, **kwargs)
        elif provider == "gemini":
            return LLMProvider._create_gemini(model, api_key, temperature, max_tokens, **kwargs)
        elif provider == "openai":
            return LLMProvider._create_openai(model, api_key, temperature, max_tokens, **kwargs)
        elif provider == "anthropic":
            return LLMProvider._create_anthropic(model, api_key, temperature, max_tokens, **kwargs)
        elif provider == "ollama":
            return LLMProvider._create_ollama(model, temperature, max_tokens, **kwargs)
        else:
            raise ValueError(
                f"Unknown provider: {provider}. "
                f"Supported: openrouter, groq, gemini, openai, anthropic, ollama"
            )

    @staticmethod
    def _create_openrouter(
        model: str | None, api_key: str | None, temperature: float, max_tokens: int, **kwargs
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "openai/gpt-oss-20b:free",
            api_key=api_key or "no-key",
            base_url="https://openrouter.ai/api/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _create_groq(
        model: str | None, api_key: str | None, temperature: float, max_tokens: int, **kwargs
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "openai/gpt-oss-120b",
            api_key=api_key or "no-key",
            base_url="https://api.groq.com/openai/v1",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _create_gemini(
        model: str | None, api_key: str | None, temperature: float, max_tokens: int, **kwargs
    ) -> BaseChatModel:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model or "gemini-3.6-flash",
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _create_openai(
        model: str | None, api_key: str | None, temperature: float, max_tokens: int, **kwargs
    ) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model or "gpt-5.6-sol",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _create_anthropic(
        model: str | None, api_key: str | None, temperature: float, max_tokens: int, **kwargs
    ) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model or "claude-opus-5",
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    @staticmethod
    def _create_ollama(model: str | None, temperature: float, max_tokens: int, **kwargs) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        base_url = kwargs.pop("base_url", "http://localhost:11434")
        return ChatOllama(
            model=model or "llama3.1",
            base_url=base_url,
            temperature=temperature,
            num_predict=max_tokens,
            **kwargs,
        )


def friendly_error_message(exc: Exception) -> str:
    """Turn a raw provider exception into a short, actionable user message."""
    if isinstance(exc, PayloadTooLargeError):
        return (
            "The conversation is too large for the model's request limit, even "
            "after automatic compaction. Start a new conversation, or split the "
            "task into smaller steps."
        )
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    mod = type(exc).__module__

    if isinstance(status, int):
        if status == 401:
            return "Authentication failed (HTTP 401) — check your API key in Settings or .env."
        if status == 403:
            return "Access denied (HTTP 403) — this API key lacks permission for that model."
        if status == 404:
            return "Model not found (HTTP 404) — check the model name with /model."
        if status == 429:
            return "Rate limit reached (HTTP 429) — wait a moment and try again."
        if status == 524:
            return (
                "Provider timed out (HTTP 524) — the model took too long to respond. "
                "Retry, or switch to a faster model with /model."
            )
        if status == 408:
            return "Request timed out (HTTP 408) — try again."
        if status >= 500:
            return f"Provider server error (HTTP {status}) — try again, or use a faster model with /model."
        return f"Request rejected (HTTP {status}) — check the model/config and try again."

    if "Timeout" in name:
        return "The provider took too long to respond — try again, or use a faster model with /model."

    try:
        import httpx
    except ImportError:  # pragma: no cover
        httpx = None

    if httpx is not None and isinstance(exc, httpx.HTTPError):
        return "Could not reach the AI provider — check your internet connection and try again."
    if "Connection" in name or "ConnectionError" in mod:
        return "Could not reach the AI provider — check your internet connection and try again."

    text = str(exc).strip()
    if text and text != name:
        return (text[:300] + "…") if len(text) > 300 else text
    return name
