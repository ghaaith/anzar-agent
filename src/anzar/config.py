"""Configuration management for Anzar."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: three levels up from src/anzar/config.py → project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_FILE = str(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Server settings for SaaS mode."""

    database_url: str = f"sqlite:///{Path.home() / '.anzar' / 'anzar.db'}"
    redis_url: str = "redis://localhost:6379"

    jwt_secret: str = "anzar-dev-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    google_client_id: Optional[str] = None
    google_client_secret: Optional[str] = None
    github_client_id: Optional[str] = None
    github_client_secret: Optional[str] = None

    default_workspace_limit_mb: int = 500
    pro_workspace_limit_mb: int = 2048
    team_workspace_limit_mb: int = 5120

    max_upload_size_mb: int = 50

    docker_host: str = "unix:///var/run/docker.sock"
    sandbox_image: str = "anzar-sandbox:latest"
    idle_timeout_minutes: int = 30
    health_check_interval: int = 60

    google_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    langsmith_api_key: Optional[str] = None
    langsmith_tracing: bool = False

    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


class AnzarConfig(BaseModel):
    """CLI configuration loaded from file or environment."""

    provider: str = "groq"
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_steps: Optional[int] = None
    ollama_url: str = "http://localhost:11434"
    last_conversation_id: Optional[str] = None
    last_workspace: Optional[str] = None

    @classmethod
    def load(cls) -> AnzarConfig:
        """Load config from ~/.anzar/config.yaml and environment variables."""
        from dotenv import load_dotenv

        # Load .env from project root (not CWD)
        load_dotenv(_PROJECT_ROOT / ".env")

        config_data: dict = {}

        config_path = Path.home() / ".anzar" / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config_data = yaml.safe_load(f) or {}

        provider = os.environ.get("ANZAR_PROVIDER", config_data.get("provider", "groq"))
        model = os.environ.get("ANZAR_MODEL", config_data.get("model"))
        ollama_url = os.environ.get("ANZAR_OLLAMA_URL", config_data.get("ollama_url", "http://localhost:11434"))

        # Max tool rounds per turn: env var > config.yaml > None (module default).
        raw_steps = os.environ.get("ANZAR_MAX_STEPS", config_data.get("max_steps"))
        max_steps = None
        if raw_steps is not None:
            try:
                max_steps = int(raw_steps)
            except (TypeError, ValueError):
                max_steps = None

        # Always auto-detect API key from provider-specific env vars (never trust persisted key)
        api_key = os.environ.get("ANZAR_API_KEY")
        if not api_key:
            provider_key_map = {
                "openrouter": "OPENROUTER_API_KEY",
                "groq": "GROQ_API_KEY",
                "gemini": "GOOGLE_API_KEY",
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
            }
            env_var = provider_key_map.get(provider)
            if env_var:
                api_key = os.environ.get(env_var)

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            max_steps=max_steps,
            ollama_url=ollama_url,
            last_conversation_id=os.environ.get("ANZAR_LAST_CONV", config_data.get("last_conversation_id")),
            last_workspace=os.environ.get("ANZAR_LAST_WS", config_data.get("last_workspace")),
        )

    def save(self) -> None:
        """Save config to ~/.anzar/config.yaml."""
        config_dir = Path.home() / ".anzar"
        config_dir.mkdir(parents=True, exist_ok=True)

        config_path = config_dir / "config.yaml"
        config_data = {
            "provider": self.provider,
            "model": self.model,
            "max_steps": self.max_steps,
            "ollama_url": self.ollama_url,
            "last_conversation_id": self.last_conversation_id,
            "last_workspace": self.last_workspace,
        }

        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

    def get_default_model(self) -> str:
        """Get the default model for the current provider."""
        if self.model:
            return self.model

        from anzar.agent.providers import default_model

        return default_model(self.provider) or "openai/gpt-oss-20b:free"
