"""Unit tests for the deterministic post-modification verifier.

Covers: project/command detection (python + node), output parsing,
UNAVAILABLE handling, the CommandPolicy gate (no security bypass), the
guarantee that the verifier executes only through SandboxManager (never
subprocess directly), and the mutation snapshot helpers.
"""

import json
import subprocess
import time

from anzar.agent.verifier import (
    MAX_VERIFICATION_ATTEMPTS,
    VerificationResult,
    VerificationStatus,
    WorkspaceVerifier,
    snapshot_changed,
    snapshot_workspace,
    verification_message,
)


class _FakeManager:
    def __init__(self, result=None):
        self.calls = []
        self.result = result or {
            "exit_code": 0,
            "stdout": "5 passed in 0.20s",
            "stderr": "",
            "executor": "subprocess",
        }

    def exec_workspace_command(self, command, workspace_path, timeout=30):
        self.calls.append((command, workspace_path, timeout))
        return self.result


# --------------------------------------------------------------------------
# Snapshot-based mutation detection
# --------------------------------------------------------------------------


def test_snapshot_detects_edits(tmp_path):
    f = tmp_path / "main.py"
    f.write_text("a = 1", encoding="utf-8")
    before = snapshot_workspace(str(tmp_path))
    time.sleep(0.01)
    f.write_text("a = 2\nb = 3", encoding="utf-8")
    after = snapshot_workspace(str(tmp_path))
    assert snapshot_changed(before, after)


def test_snapshot_detects_new_file(tmp_path):
    (tmp_path / "main.py").write_text("x", encoding="utf-8")
    before = snapshot_workspace(str(tmp_path))
    (tmp_path / "new.py").write_text("y", encoding="utf-8")
    assert snapshot_changed(before, snapshot_workspace(str(tmp_path)))


def test_snapshot_no_change(tmp_path):
    (tmp_path / "main.py").write_text("x", encoding="utf-8")
    s1 = snapshot_workspace(str(tmp_path))
    s2 = snapshot_workspace(str(tmp_path))
    assert not snapshot_changed(s1, s2)


def test_snapshot_ignores_cache_dirs(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("code", encoding="utf-8")
    cache = tmp_path / ".pytest_cache"
    cache.mkdir()
    (cache / "x").write_text("junk", encoding="utf-8")
    pycache = tmp_path / "src" / "__pycache__"
    pycache.mkdir()
    (pycache / "main.cpython-312.pyc").write_bytes(b"\x00")
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "pkg").write_text("x", encoding="utf-8")
    snap = snapshot_workspace(str(tmp_path))
    keys = set(snap)
    assert "src/main.py" in keys
    assert not any(k for k in keys if ".pytest_cache" in k or "__pycache__" in k or "node_modules" in k)


# --------------------------------------------------------------------------
# Project / command selection
# --------------------------------------------------------------------------


def test_python_detection_pytest_ini(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    cmd = WorkspaceVerifier(str(tmp_path))._select_command()
    assert cmd and "-m pytest -q" in cmd


def test_python_detection_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]", encoding="utf-8")
    cmd = WorkspaceVerifier(str(tmp_path))._select_command()
    assert cmd and "-m pytest -q" in cmd


def test_python_detection_tests_dir(tmp_path):
    (tmp_path / "tests").mkdir()
    assert WorkspaceVerifier(str(tmp_path))._select_command() is not None


def test_unavailable_on_empty_project(tmp_path):
    result = WorkspaceVerifier(str(tmp_path)).verify()
    assert result.status is VerificationStatus.UNAVAILABLE
    assert result.command is None


def _write_package(tmp_path, scripts):
    (tmp_path / "package.json").write_text(json.dumps({"scripts": scripts}), encoding="utf-8")


def test_node_script_priority(tmp_path):
    _write_package(tmp_path, {"lint": "eslint .", "test": "jest", "typecheck": "tsc --noEmit"})
    assert WorkspaceVerifier(str(tmp_path))._select_command() == "npm run test"


def test_node_lint_fallback(tmp_path):
    _write_package(tmp_path, {"lint": "eslint ."})
    assert WorkspaceVerifier(str(tmp_path))._select_command() == "npm run lint"


def test_node_typecheck_fallback(tmp_path):
    _write_package(tmp_path, {"typecheck": "tsc --noEmit"})
    assert WorkspaceVerifier(str(tmp_path))._select_command() == "npm run typecheck"


def test_node_no_scripts_unavailable(tmp_path):
    _write_package(tmp_path, {})
    result = WorkspaceVerifier(str(tmp_path)).verify()
    assert result.status is VerificationStatus.UNAVAILABLE


# --------------------------------------------------------------------------
# Output classification
# --------------------------------------------------------------------------


def test_classify_pytest_failures():
    out = (
        "src/x.py:12: in test_a\n"
        "E   assert 1 == 2\n"
        "FAILED tests/test_x.py::test_a - AssertionError: assert 1 == 2\n"
        "1 passed, 1 failed in 0.12s\n"
    )
    r = WorkspaceVerifier(".")._classify("pytest -q", 1, out, "", 10)
    assert r.status is VerificationStatus.FAILED
    assert r.tests_passed == 1 and r.tests_failed == 1
    assert r.failures and "FAILED tests/test_x.py" in r.failures[0]


def test_classify_pass():
    r = WorkspaceVerifier(".")._classify("pytest -q", 0, "5 passed in 0.20s\n", "", 15)
    assert r.status is VerificationStatus.PASSED
    assert r.tests_passed == 5 and r.tests_failed == 0


def test_classify_no_tests_unavailable():
    r = WorkspaceVerifier(".")._classify("pytest -q", 5, "no tests ran in 0.01s\n", "", 9)
    assert r.status is VerificationStatus.UNAVAILABLE


def test_classify_npm_output():
    out = (
        "PASS src/app.test.ts\n"
        "Tests: 8 passed, 2 failed\n"
        "Test Suites: 1 failed, 1 total\n"
    )
    r = WorkspaceVerifier(".")._classify("npm run test", 1, out, "", 100)
    assert r.status is VerificationStatus.FAILED
    assert r.tests_failed == 2 and r.tests_passed == 8


def test_classify_error_no_summary():
    r = WorkspaceVerifier(".")._classify("pytest -q", 1, "", "'pytest' is not recognized", 5)
    assert r.status is VerificationStatus.ERROR


def test_failures_capped():
    out = "\n".join(f"FAILED tests/test_f{i}.py::test_i - boom" for i in range(50))
    passed, failed, failures = WorkspaceVerifier._parse_output(out, "")
    assert failed == 0
    assert len(failures) == 20


# --------------------------------------------------------------------------
# Execution path: SandboxManager only, CommandPolicy gate only
# --------------------------------------------------------------------------


def test_verifier_runs_via_manager(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    manager = _FakeManager()
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: manager)

    result = WorkspaceVerifier(str(tmp_path)).verify()

    assert result.status is VerificationStatus.PASSED
    assert len(manager.calls) == 1
    command, ws, timeout = manager.calls[0]
    assert command.endswith("-m pytest -q")
    assert ws == str(tmp_path)
    assert timeout == 120  # verifier timeout, not the 30s command policy timeout


def test_verifier_never_calls_subprocess(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: _FakeManager())

    def boom(*args, **kwargs):
        raise AssertionError("verifier must not call subprocess directly")

    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr(subprocess, "run", boom)

    result = WorkspaceVerifier(str(tmp_path)).verify()
    assert result.status is VerificationStatus.PASSED


def test_policy_gate_blocks_disallowed_command(tmp_path, monkeypatch):
    manager = _FakeManager()
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: manager)
    verifier = WorkspaceVerifier(str(tmp_path))
    monkeypatch.setattr(verifier, "_select_command", lambda: "sudo rm -rf .")

    result = verifier.verify()

    assert result.status is VerificationStatus.ERROR
    assert manager.calls == []  # never executed — the gate refused it
    assert "not allowed by policy" in result.failures[0]


def test_verify_timeout_reported(tmp_path, monkeypatch):
    (tmp_path / "pytest.ini").write_text("[pytest]", encoding="utf-8")
    manager = _FakeManager(result={"exit_code": -1, "stdout": "", "stderr": "", "timed_out": True})
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: manager)
    result = WorkspaceVerifier(str(tmp_path)).verify()
    assert result.status is VerificationStatus.TIMEOUT


# --------------------------------------------------------------------------
# Rendering + constants
# --------------------------------------------------------------------------


def test_verification_message_format():
    result = VerificationResult(
        status=VerificationStatus.FAILED,
        command="py -m pytest -q",
        tests_passed=1,
        tests_failed=2,
        failures=["FAILED tests/test_x.py::test_a - assert 1 == 2"],
    )
    msg = verification_message(result, attempt=1, baseline_failures=2)
    assert "[VERIFICATION: failed]" in msg
    assert "attempt 1/3" in msg
    assert "[PASSED: 1] [FAILED: 2]" in msg
    assert "[BASELINE: 2" in msg
    assert "FAILED tests/test_x.py" in msg


def test_verification_message_passed():
    msg = verification_message(
        VerificationResult(status=VerificationStatus.PASSED, tests_passed=9),
        attempt=1,
        baseline_failures=0,
    )
    assert "[VERIFICATION: passed]" in msg
    assert "continue" in msg


def test_max_verification_attempts_constant():
    assert MAX_VERIFICATION_ATTEMPTS == 3
