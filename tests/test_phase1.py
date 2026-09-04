"""Phase 1 + Phase 2 integration tests."""

from __future__ import annotations

import os
import sys

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


print("\n=== PHASE 1: AUTH ===")

# Register
r = c.post('/api/auth/register', json={'email':'test@test.com','password':'123456','name':'Test User'})
test("Register", r.status_code, 200, is_status=True)
data = r.json()
access = data['access_token']
refresh = data['refresh_token']
h = {'Authorization': f'Bearer {access}'}

# Login
r = c.post('/api/auth/login', json={'email':'test@test.com','password':'123456'})
test("Login", r.status_code, 200, is_status=True)

# Refresh
r = c.post('/api/auth/refresh', json={'refresh_token': refresh})
test("Refresh token", r.status_code, 200, is_status=True)

# Me
r = c.get('/api/auth/me', headers=h)
test("Get me", r.status_code, 200, is_status=True)
test("Me has plan", r.json().get('plan'), True)

# Settings
r = c.put('/api/settings', headers=h, json={'provider':'gemini','model':'gemini-2.5-pro'})
test("Update settings", r.status_code, 200, is_status=True)
r = c.get('/api/settings', headers=h)
test("Get settings", r.json().get('model'), 'gemini-2.5-pro')

# Conversations
r = c.post('/api/conversations', headers=h, params={'title':'Test Chat'})
test("Create conversation", r.status_code, 200, is_status=True)
conv_id = r.json()['id']

r = c.get('/api/conversations', headers=h)
test("List conversations", len(r.json()), 1)

# Chat
r = c.post('/api/chat', headers=h, json={'message':'Hello','conversation_id': conv_id})
test("Chat", r.status_code, 200, is_status=True)

# Error cases
r = c.post('/api/auth/register', json={'email':'test@test.com','password':'123456','name':'Dup'})
test("Duplicate register (400)", r.status_code, 400, is_status=True)
r = c.post('/api/auth/login', json={'email':'test@test.com','password':'wrong'})
test("Wrong password (401)", r.status_code, 401, is_status=True)
r = c.get('/api/settings')
test("No auth (401)", r.status_code, 401, is_status=True)


print("\n=== PHASE 2: WORKSPACE / SANDBOX ===")

# Get workspace status
r = c.get('/api/workspace/status', headers=h)
test("Workspace status", r.status_code, 200, is_status=True)
ws_data = r.json()
test("Workspace has disk_path", ws_data.get('disk_path'), True)
test("Workspace status is stopped", ws_data.get('status'), 'stopped')
test("Workspace has disk_limit", ws_data.get('disk_limit_mb'), True)

# Start workspace (subprocess mode since no Docker)
r = c.post('/api/workspace/start', headers=h)
test("Start workspace", r.status_code, 200, is_status=True)
start_data = r.json()
test("Start returns running", start_data.get('status'), 'running')

# Check status after start
r = c.get('/api/workspace/status', headers=h)
test("Status after start", r.json().get('status'), 'running')

# Stop workspace
r = c.post('/api/workspace/stop', headers=h)
test("Stop workspace", r.status_code, 200, is_status=True)
test("Stop returns stopped", r.json().get('status'), 'stopped')

# Check status after stop
r = c.get('/api/workspace/status', headers=h)
test("Status after stop", r.json().get('status'), 'stopped')

# Destroy without confirmation
r = c.post('/api/workspace/destroy', headers=h)
test("Destroy without confirm (400)", r.status_code, 400, is_status=True)

# Destroy with confirmation
r = c.post('/api/workspace/destroy', headers=h, params={'confirm': 'DESTROY'})
test("Destroy with confirm", r.status_code, 200, is_status=True)

# Exec in subprocess mode
r = c.post('/api/workspace/start', headers=h)
r = c.post('/api/workspace/exec', headers=h, json={'command': 'echo hello'})
test("Exec command", r.status_code, 200, is_status=True)
test("Exec returns stdout", 'hello' in r.json().get('stdout', ''), True)

# Exec in stopped workspace (should fail)
r = c.post('/api/workspace/stop', headers=h)
r = c.post('/api/workspace/exec', headers=h, json={'command': 'echo hello'})
test("Exec on stopped (400)", r.status_code, 400, is_status=True)


print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*40}\n")

if __name__ == "__main__":
    sys.exit(1 if failed > 0 else 0)
