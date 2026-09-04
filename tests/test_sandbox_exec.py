"""Integration tests: run_command tool + sandbox executor against a temp workspace."""

import os
import shutil
import subprocess
import sys

import pytest

from anzar.agent.policy import CommandPolicy
from anzar.agent.tools import create_tools


def _interpreter_works(name: str) -> bool:
    try:
        result = subprocess.run(
            [name, "-c", "print(1)"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return result.returncode == 0
    except Exception:
        return False


def _real_interpreter() -> str | None:
    """Return a Python launcher on PATH that actually works.

    ``shutil.which("python")`` alone is unreliable: the Windows Store stub
    resolves on PATH but fails at runtime with exit 9009. We probe each
    candidate so tests only run where execution is genuinely possible.
    """
    for name in ("python", "python3", "py"):
        if shutil.which(name) and _interpreter_works(name):
            return name
    return None


_INTERPRETER = _real_interpreter()

pytestmark = pytest.mark.skipif(
    _INTERPRETER is None,
    reason="no usable python interpreter on PATH (checked python, python3, py)",
)

PY = _INTERPRETER or "python"


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "main.py").write_text('print("hello world")', encoding="utf-8")
    (tmp_path / "gen.py").write_text("print('x' * 1000)", encoding="utf-8")
    (tmp_path / "err.py").write_text(
        "import sys; sys.stderr.write('e' * 1000); print('ok')", encoding="utf-8"
    )
    (tmp_path / "fail.py").write_text("import sys; sys.exit(3)", encoding="utf-8")
    (tmp_path / "sleep.py").write_text("import time; time.sleep(30)", encoding="utf-8")
    (tmp_path / "cwd.py").write_text("import os; print(os.getcwd())", encoding="utf-8")
    return tmp_path


def _run_tool(workspace, command, policy=None, **kwargs):
    tools = create_tools(str(workspace), policy=policy)
    rc = next(t for t in tools if t.name == "run_command")
    return rc.invoke({"command": command, **kwargs})


class _ExplodingManager:
    """Raises if the executor is ever reached."""

    def __getattr__(self, name):
        raise AssertionError(f"executor should not be called (got {name})")


def _patch_manager(monkeypatch):
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: _ExplodingManager())


# -- happy path ----------------------------------------------------------------


def test_success_execution(workspace):
    result = _run_tool(workspace, f"{PY} main.py")
    assert "[STATUS: success]" in result
    assert "hello world" in result


def test_error_exit_code(workspace):
    result = _run_tool(workspace, f"{PY} fail.py")
    assert "[STATUS: error]" in result
    assert "[EXIT: 3]" in result


def test_runs_in_workspace_cwd(workspace):
    result = _run_tool(workspace, f"{PY} cwd.py")
    ws = os.path.normcase(os.path.realpath(str(workspace)))
    assert ws in os.path.normcase(result)


def test_timeout(workspace):
    policy = CommandPolicy(str(workspace), timeout=1)
    result = _run_tool(workspace, f"{PY} sleep.py", policy=policy)
    assert "[STATUS: timeout]" in result
    assert "timed out" in result.lower()


def test_stdout_truncated(workspace):
    policy = CommandPolicy(str(workspace), max_stdout_chars=100)
    result = _run_tool(workspace, f"{PY} gen.py", policy=policy)
    assert "[STATUS: success]" in result
    assert "STDOUT TRUNCATED" in result
    stdout_part = result.split("[NOTE: STDOUT TRUNCATED]", 1)[1].strip()
    assert len(stdout_part) <= 100


def test_stderr_truncated(workspace):
    policy = CommandPolicy(str(workspace), max_stderr_chars=100)
    result = _run_tool(workspace, f"{PY} err.py", policy=policy)
    assert "STDERR TRUNCATED" in result


# -- policy gates ---------------------------------------------------------------


def test_blocked_command_does_not_execute(workspace, monkeypatch):
    _patch_manager(monkeypatch)
    result = _run_tool(workspace, "rm -rf /")
    assert "[STATUS: blocked]" in result
    assert "not executed" in result.lower() or "refused" in result.lower()


def test_injection_does_not_execute(workspace, monkeypatch):
    _patch_manager(monkeypatch)
    result = _run_tool(workspace, f"{PY} main.py; rm -rf /")
    assert "[STATUS: blocked]" in result


def test_approval_required_does_not_execute(workspace, monkeypatch):
    _patch_manager(monkeypatch)
    result = _run_tool(workspace, "pip install requests")
    assert "[STATUS: approval_required]" in result
    assert "approved" in result


def test_approved_command_executes(workspace):
    result = _run_tool(workspace, "mkdir approved_dir", approved=True)
    assert "[STATUS: success]" in result
    assert (workspace / "approved_dir").is_dir()


def test_approval_without_flag_still_requires_approval(workspace, monkeypatch):
    _patch_manager(monkeypatch)
    result = _run_tool(workspace, "mkdir should_not_exist")
    assert "[STATUS: approval_required]" in result
    assert not (workspace / "should_not_exist").exists()


# -- sandbox method-level -------------------------------------------------------


def test_subprocess_exec_cwd_bounds_and_reports(workspace):
    from anzar.sandbox import get_manager

    result = get_manager()._subprocess_exec_cwd(
        f"{PY} main.py", cwd=str(workspace), timeout=10
    )
    assert result["exit_code"] == 0
    assert "hello world" in result["stdout"]
    assert result.get("executor") is None
