"""Tests for the shared provider registry and CLI model-toggle helpers."""

from __future__ import annotations

import tempfile
import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import anzar.db.models  # noqa: F401 — register models
from anzar.agent.llm import LLMProvider, friendly_error_message
from anzar.agent.providers import (
    SUPPORTED_PROVIDERS,
    default_model,
    provider_models,
    provider_name,
    provider_names,
)
from anzar.cli import _apply_model, _provider_has_key
from anzar.config import AnzarConfig
from anzar.db.base import Base


def test_registry_has_all_six_providers():
    assert set(SUPPORTED_PROVIDERS.keys()) == {
        "openrouter", "groq", "gemini", "openai", "anthropic", "ollama",
    }


def test_registry_first_model_matches_config_default():
    for pid in SUPPORTED_PROVIDERS:
        config = AnzarConfig(provider=pid, model=None)
        assert config.get_default_model() == default_model(pid), pid


def test_registry_models_are_constructible():
    for pid, info in SUPPORTED_PROVIDERS.items():
        model = LLMProvider.create(provider=pid, model=info["models"][0], api_key="test-key")
        assert model is not None, pid


def test_provider_helpers():
    assert provider_names() == list(SUPPORTED_PROVIDERS.keys())
    assert provider_models("openrouter") == SUPPORTED_PROVIDERS["openrouter"]["models"]
    assert provider_models("nope") == []
    assert provider_name("openrouter") == "OpenRouter"
    assert provider_name("nope") == "nope"
    assert default_model("groq") == "openai/gpt-oss-120b"
    assert default_model("openrouter") == "openai/gpt-oss-20b:free"
    assert default_model("nope") is None


def test_provider_has_key(monkeypatch):
    monkeypatch.delenv("ANZAR_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert _provider_has_key("ollama")
    assert not _provider_has_key("openrouter")

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    assert _provider_has_key("openrouter")
    assert not _provider_has_key("groq")

    monkeypatch.delenv("OPENROUTER_API_KEY")
    monkeypatch.setenv("ANZAR_API_KEY", "sk-any")
    assert _provider_has_key("gemini")
    assert _provider_has_key("anthropic")


def _make_db():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)


def test_apply_model_switches_and_persists(monkeypatch):
    from anzar.db.models import Conversation, Settings, User, Workspace

    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)

    Session = _make_db()
    db = Session()
    user = User(
        email=f"cli-model-{uuid.uuid4().hex[:8]}@test.local",
        name="T",
        plan="pro",
        auth_provider="cli",
    )
    db.add(user)
    db.flush()
    ws = Workspace(
        user_id=user.id,
        name="w",
        disk_path=tempfile.mkdtemp(),
        disk_limit_mb=10,
    )
    db.add(ws)
    db.add(Settings(user_id=user.id, provider="groq", model=None))
    conv = Conversation(user_id=user.id, title="Test")
    db.add(conv)
    db.commit()
    db.refresh(user)
    db.refresh(conv)

    config = AnzarConfig(provider="groq", model=None)
    toolbar = {"provider": "groq", "model": "llama-3.3-70b-versatile", "msgs": "0"}

    agent, provider, model = _apply_model(
        db, user, conv, ws.disk_path, config, "openrouter", "poolside/laguna-xs-2.1:free", toolbar
    )
    assert agent is not None
    assert provider == "openrouter"
    assert model == "poolside/laguna-xs-2.1:free"
    assert toolbar["provider"] == "openrouter"
    assert toolbar["model"] == "poolside/laguna-xs-2.1:free"
    assert config.provider == "openrouter"
    assert config.model == "poolside/laguna-xs-2.1:free"

    # Switching with no model resolves to the provider default
    agent2, provider2, model2 = _apply_model(
        db, user, conv, ws.disk_path, config, "openrouter", None, toolbar
    )
    assert agent2 is not None
    assert provider2 == "openrouter"
    assert model2 is None
    assert toolbar["model"] == "openai/gpt-oss-20b:free"

    # Switching to a provider with no API key fails gracefully (no crash, no change)
    no_key_agent, no_key_provider, no_key_model = _apply_model(
        db, user, conv, ws.disk_path, config, "gemini", None, toolbar
    )
    assert no_key_agent is None
    assert no_key_provider == "openrouter"
    assert no_key_model is None
    assert config.provider == "openrouter"

    db.close()


class _StubError(Exception):
    status_code = None


def test_friendly_error_maps_http_status_codes():
    e = _StubError("boom")
    e.status_code = 524
    msg = friendly_error_message(e)
    assert "524" in msg
    assert "faster model" in msg

    e.status_code = 429
    assert "Rate limit" in friendly_error_message(e)

    e.status_code = 401
    assert "API key" in friendly_error_message(e)

    e.status_code = 404
    assert "model" in friendly_error_message(e).lower()

    e.status_code = 503
    assert "503" in friendly_error_message(e)

    e.status_code = 400
    assert "400" in friendly_error_message(e)


def test_friendly_error_maps_timeouts_and_connection():
    import httpx

    assert "too long" in friendly_error_message(httpx.TimeoutException("slow"))
    assert "too long" in friendly_error_message(httpx.ReadTimeout("slow"))
    assert "too long" in friendly_error_message(httpx.ConnectTimeout("conn"))
    assert "internet" in friendly_error_message(httpx.ConnectError("conn"))


def test_friendly_error_fallback_truncates_long_messages():
    long = "x" * 500
    msg = friendly_error_message(ValueError(long))
    assert msg.endswith("…")
    assert len(msg) <= 301

    assert friendly_error_message(ValueError("simple")) == "simple"
