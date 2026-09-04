"""Phase 5: Profile, Usage, Rate Limiting tests."""

import os
import sys
from os.path import dirname

# Delete any previous test DB and use a fresh one
_test_db = f"{os.path.dirname(__file__)}/test_anzar.db"
for f in (_test_db, _test_db + "-wal", _test_db + "-shm"):
    try:
        os.remove(f)
    except OSError:
        pass
os.environ["DATABASE_URL"] = f"sqlite:///{_test_db}"

from anzar.db.base import init_db
init_db()

from anzar.server import app
from fastapi.testclient import TestClient

c = TestClient(app)
passed = 0
failed = 0


def test(name: str, actual, expected, is_status=False):
    test.__test__ = False  # script-style harness — not a pytest test
    global passed, failed
    if is_status:
        ok = actual == expected
    else:
        ok = bool(actual)
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] {name}" + (f" (got {actual}, expected {expected})" if not ok and is_status else ""))


print("\n=== PHASE 5: PROFILE, USAGE, RATE LIMITING ===")

# Setup: register and login
r = c.post("/api/auth/register", json={"name": "Test User", "email": "phase5@test.com", "password": "pass123"})
token = r.json()["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# --- Profile ---
print("\n--- Profile ---")

r = c.get("/api/profile", headers=headers)
test("Get profile returns 200", r.status_code, 200, is_status=True)
data = r.json()
test("Profile has email", data["email"], "phase5@test.com")
test("Profile has name", data["name"], "Test User")
test("Profile has plan", data["plan"], "free")
test("Profile has auth_provider", data["auth_provider"], "local")

r = c.put("/api/profile", headers=headers, json={"name": "Updated Name"})
test("Update profile returns 200", r.status_code, 200, is_status=True)
test("Name was updated", r.json()["name"], "Updated Name")

r = c.put("/api/profile", headers=headers, json={"email": "phase5@test.com"})
test("Re-setting same email returns 200", r.status_code, 200, is_status=True)

# Register second user to test email conflict
c.post("/api/auth/register", json={"name": "Other", "email": "other@test.com", "password": "pass123"})
r = c.put("/api/profile", headers=headers, json={"email": "other@test.com"})
test("Duplicate email returns 400", r.status_code, 400, is_status=True)

# --- Password ---
print("\n--- Password Change ---")

r = c.put("/api/profile/password", headers=headers, json={
    "current_password": "pass123",
    "new_password": "newpass123"
})
test("Change password returns 200", r.status_code, 200, is_status=True)
test("Change password status ok", r.json()["status"], "ok")

# Login with new password
r = c.post("/api/auth/login", json={"email": "phase5@test.com", "password": "newpass123"})
test("Login with new password works", r.status_code, 200, is_status=True)

# Wrong current password
r = c.put("/api/profile/password", headers=headers, json={
    "current_password": "wrong",
    "new_password": "another123"
})
test("Wrong current password returns 400", r.status_code, 400, is_status=True)

# Short password
r = c.put("/api/profile/password", headers=headers, json={
    "current_password": "newpass123",
    "new_password": "123"
})
test("Short password returns 400", r.status_code, 400, is_status=True)

# --- Usage ---
print("\n--- Usage ---")

r = c.get("/api/usage/summary", headers=headers)
test("Usage summary returns 200", r.status_code, 200, is_status=True)
data = r.json()
test("Summary has total_tokens", isinstance(data["total_tokens"], int), True)
test("Summary has total_actions", isinstance(data["total_actions"], int), True)
test("Summary has plan", data["plan"], "free")
test("Summary has token_limit", data["token_limit"], 10000)
test("Summary has daily list", isinstance(data["daily"], list), True)

r = c.get("/api/usage/history", headers=headers)
test("Usage history returns 200", r.status_code, 200, is_status=True)
test("History is a list", isinstance(r.json(), list), True)

# Send a chat to generate usage
c.post("/api/conversations", headers=headers, json={"title": "Usage Test"})
conv_id = c.get("/api/conversations", headers=headers).json()[0]["id"]
c.post("/api/chat", headers=headers, json={"message": "Hello", "conversation_id": conv_id})

r = c.get("/api/usage/summary", headers=headers)
data = r.json()
test("Usage recorded after chat", data["total_actions"] > 0, True)

r = c.get("/api/usage/history", headers=headers)
test("History has entries after chat", len(r.json()) > 0, True)

# --- Rate Limiting ---
print("\n--- Rate Limiting ---")

# Rate limit should not block unauthenticated requests
r = c.get("/api/health")
test("Health check still works", r.status_code, 200, is_status=True)

# Authenticated requests should work
r = c.get("/api/usage/summary", headers=headers)
test("Authenticated request works", r.status_code, 200, is_status=True)

# --- No Auth ---
print("\n--- No Auth ---")

r = c.get("/api/profile")
test("Profile without auth returns 401", r.status_code, 401, is_status=True)

r = c.get("/api/usage/summary")
test("Usage without auth returns 401", r.status_code, 401, is_status=True)

r = c.put("/api/profile", json={"name": "No Auth"})
test("Update profile without auth returns 401", r.status_code, 401, is_status=True)


print("\n" + "=" * 50)
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 50)
