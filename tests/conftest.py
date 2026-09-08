"""Shared pytest fixtures for the anzar HTTP API integration tests.

The FastAPI server binds its SQLAlchemy engine from ``DATABASE_URL`` at import
time, so this conftest points it at a fresh temp SQLite database *before* the
server is imported anywhere in the process. The single in-process server
instance is shared by every API test file.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Configure a fresh temp SQLite DB before importing the server.
_TMP = tempfile.mkdtemp(prefix="anzar-api-")
_TEST_DB = Path(_TMP) / "test_anzar.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TEST_DB}"

from anzar.db.base import init_db  # noqa: E402
from anzar.server import app  # noqa: E402

init_db()

from fastapi.testclient import TestClient  # noqa: E402

_api_client = TestClient(app)


def register_user(
    client: TestClient, email: str, password: str = "pass123", name: str = "Test User"
) -> dict:
    """Register a user against the test API and return a live token set."""
    r = client.post("/api/auth/register", json={"name": name, "email": email, "password": password})
    assert r.status_code == 200, r.text
    data = r.json()
    return {"email": email, "password": password, "name": name, **data}


def _llm_provider_configured() -> bool:
    from anzar.config import settings

    return bool(
        settings.openrouter_api_key
        or settings.groq_api_key
        or settings.google_api_key
        or settings.openai_api_key
        or settings.anthropic_api_key
    )


#: Mark for tests that call the real LLM provider. Skipped when no provider key
#: is configured so the committed test suite never depends on external APIs.
needs_llm = pytest.mark.skipif(not _llm_provider_configured(), reason="No LLM provider configured")


@pytest.fixture(scope="session")
def api_client():
    """A TestClient bound to the shared in-process server."""
    return _api_client


@pytest.fixture(scope="session")
def client(api_client):
    """Short alias used by the converted phase test files."""
    return api_client
