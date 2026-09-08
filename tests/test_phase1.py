"""Phase 1 + Phase 2: auth, settings, conversations, chat, and workspace API tests.

Real pytest conversion of the former script-style harness.
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_llm, register_user


@pytest.fixture(scope="module")
def user(client):
    return register_user(client, "phase1@test.com", password="123456")


@pytest.fixture(scope="module")
def auth_headers(user):
    return {"Authorization": f"Bearer {user['access_token']}"}


def _start(client, headers):
    r = client.post("/api/workspace/start", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


def _stop(client, headers):
    r = client.post("/api/workspace/stop", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------------------
# Phase 1: Auth
# ---------------------------------------------------------------------------


def test_register_returns_tokens(user):
    assert user["access_token"]
    assert user["refresh_token"]


def test_login_ok(client, user):
    r = client.post("/api/auth/login", json={"email": user["email"], "password": "123456"})
    assert r.status_code == 200


def test_login_wrong_password_401(client, user):
    r = client.post("/api/auth/login", json={"email": user["email"], "password": "wrong"})
    assert r.status_code == 401


def test_refresh_rotates_token(client, user):
    r = client.post("/api/auth/refresh", json={"refresh_token": user["refresh_token"]})
    assert r.status_code == 200
    assert r.json()["access_token"]


def test_me_returns_plan(client, auth_headers):
    r = client.get("/api/auth/me", headers=auth_headers)
    assert r.status_code == 200
    assert "plan" in r.json()
    assert r.json()["email"] == "phase1@test.com"


def test_duplicate_register_400(client, user):
    r = client.post(
        "/api/auth/register",
        json={"name": "Dup", "email": user["email"], "password": "123456"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Phase 1: Settings
# ---------------------------------------------------------------------------


def test_settings_update_and_read(client, auth_headers):
    r = client.put(
        "/api/settings",
        headers=auth_headers,
        json={"provider": "gemini", "model": "gemini-2.5-pro"},
    )
    assert r.status_code == 200
    r = client.get("/api/settings", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["model"] == "gemini-2.5-pro"


def test_settings_requires_auth_401(client):
    r = client.get("/api/settings")
    assert r.status_code == 401


# ---------------------------------------------------------------------------
# Phase 1: Conversations + Chat
# ---------------------------------------------------------------------------


def test_conversations_create_and_list(client, auth_headers):
    r = client.post("/api/conversations", headers=auth_headers, params={"title": "Test Chat"})
    assert r.status_code == 200
    conv_id = r.json()["id"]

    r = client.get("/api/conversations", headers=auth_headers)
    assert r.status_code == 200
    assert [c["id"] for c in r.json()] == [conv_id]


def test_chat_requires_auth_401(client):
    r = client.post("/api/chat", json={"message": "Hello"})
    assert r.status_code == 401


@needs_llm
def test_chat_with_conversation(client, auth_headers):
    r = client.post("/api/conversations", headers=auth_headers, params={"title": "Chat Test"})
    conv_id = r.json()["id"]

    r = client.post(
        "/api/chat", headers=auth_headers, json={"message": "Hello", "conversation_id": conv_id}
    )
    assert r.status_code == 200
    assert r.json()["conversation_id"] == conv_id
    assert r.json()["response"]


@needs_llm
def test_chat_creates_conversation(client, auth_headers):
    r = client.post("/api/chat", headers=auth_headers, json={"message": "Hello"})
    assert r.status_code == 200
    assert r.json()["conversation_id"]
    assert r.json()["response"]


# ---------------------------------------------------------------------------
# Phase 2: Workspace / Sandbox
# ---------------------------------------------------------------------------


def test_workspace_status_stopped(client, auth_headers):
    r = client.get("/api/workspace/status", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "stopped"
    assert data["disk_path"]
    assert data["disk_limit_mb"]


def test_workspace_start_and_stop(client, auth_headers):
    data = _start(client, auth_headers)
    assert data["status"] == "running"

    r = client.get("/api/workspace/status", headers=auth_headers)
    assert r.json()["status"] == "running"

    data = _stop(client, auth_headers)
    assert data["status"] == "stopped"


def test_workspace_exec_subprocess(client, auth_headers):
    _start(client, auth_headers)
    r = client.post("/api/workspace/exec", headers=auth_headers, json={"command": "echo hello"})
    assert r.status_code == 200
    assert "hello" in r.json().get("stdout", "")
    _stop(client, auth_headers)


def test_workspace_exec_requires_running(client, auth_headers):
    r = client.post("/api/workspace/exec", headers=auth_headers, json={"command": "echo hi"})
    assert r.status_code == 400


def test_workspace_destroy_requires_confirm(client, auth_headers):
    r = client.post("/api/workspace/destroy", headers=auth_headers)
    assert r.status_code == 400


def test_workspace_destroy_with_confirm(client, auth_headers):
    r = client.post("/api/workspace/destroy", headers=auth_headers, params={"confirm": "DESTROY"})
    assert r.status_code == 200
    assert r.json()["status"] == "destroyed"
