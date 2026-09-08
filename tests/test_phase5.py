"""Phase 5: profile, password, usage, rate-limit, and no-auth API tests.

Real pytest conversion of the former script-style harness.
"""

from __future__ import annotations

import pytest

from tests.conftest import needs_llm, register_user


@pytest.fixture(scope="module")
def user(client):
    return register_user(client, "phase5@test.com", password="pass123")


@pytest.fixture(scope="module")
def other_user(client):
    return register_user(client, "other@test.com", name="Other")


@pytest.fixture(scope="module")
def auth_headers(user):
    return {"Authorization": f"Bearer {user['access_token']}"}


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def test_get_profile(client, auth_headers, user):
    r = client.get("/api/profile", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert data["email"] == user["email"]
    assert data["name"] == "Test User"
    assert data["plan"] == "free"
    assert data["auth_provider"] == "email"


def test_update_profile_name(client, auth_headers):
    r = client.put("/api/profile", headers=auth_headers, json={"name": "Updated Name"})
    assert r.status_code == 200
    assert r.json()["name"] == "Updated Name"


def test_update_profile_same_email_ok(client, auth_headers, user):
    r = client.put("/api/profile", headers=auth_headers, json={"email": user["email"]})
    assert r.status_code == 200


def test_update_profile_duplicate_email_400(client, auth_headers, other_user):
    r = client.put("/api/profile", headers=auth_headers, json={"email": other_user["email"]})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Password
# ---------------------------------------------------------------------------


def test_change_password_and_login(client, auth_headers, user):
    r = client.put(
        "/api/profile/password",
        headers=auth_headers,
        json={"current_password": "pass123", "new_password": "newpass456"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.post("/api/auth/login", json={"email": user["email"], "password": "newpass456"})
    assert r.status_code == 200

    # Restore the original password so subsequent tests stay valid.
    r = client.put(
        "/api/profile/password",
        headers=auth_headers,
        json={"current_password": "newpass456", "new_password": "pass123"},
    )
    assert r.status_code == 200


def test_change_password_wrong_current_400(client, auth_headers):
    r = client.put(
        "/api/profile/password",
        headers=auth_headers,
        json={"current_password": "wrong", "new_password": "another123"},
    )
    assert r.status_code == 400


def test_change_password_too_short_400(client, auth_headers):
    r = client.put(
        "/api/profile/password",
        headers=auth_headers,
        json={"current_password": "pass123", "new_password": "123"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_summary_shape(client, auth_headers):
    r = client.get("/api/usage/summary", headers=auth_headers)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["total_tokens"], int)
    assert isinstance(data["total_actions"], int)
    assert data["plan"] == "free"
    assert data["token_limit"] == 10000
    assert isinstance(data["daily"], list)


def test_usage_history_shape(client, auth_headers):
    r = client.get("/api/usage/history", headers=auth_headers)
    assert r.status_code == 200
    assert isinstance(r.json(), list)


@needs_llm
def test_usage_recorded_after_chat(client, auth_headers):
    r = client.post("/api/conversations", headers=auth_headers, params={"title": "Usage Test"})
    conv_id = r.json()["id"]
    r = client.post(
        "/api/chat", headers=auth_headers, json={"message": "Hello", "conversation_id": conv_id}
    )
    assert r.status_code == 200

    r = client.get("/api/usage/summary", headers=auth_headers)
    assert r.json()["total_actions"] > 0

    r = client.get("/api/usage/history", headers=auth_headers)
    assert len(r.json()) > 0


# ---------------------------------------------------------------------------
# Rate limiting / no-auth
# ---------------------------------------------------------------------------


def test_health_unauthenticated(client):
    r = client.get("/api/health")
    assert r.status_code == 200


def test_authenticated_request_works(client, auth_headers):
    r = client.get("/api/usage/summary", headers=auth_headers)
    assert r.status_code == 200


def test_profile_without_auth_401(client):
    assert client.get("/api/profile").status_code == 401


def test_usage_without_auth_401(client):
    assert client.get("/api/usage/summary").status_code == 401


def test_update_profile_without_auth_401(client):
    r = client.put("/api/profile", json={"name": "No Auth"})
    assert r.status_code == 401
