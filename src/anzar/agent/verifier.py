"""Mandatory post-modification verification for the Anzar agent.

After any workspace mutation (``write_file``, or a ``run_command`` that
changed source files) the graph runs the verifier before continuing. The
verifier is deterministic — it makes **no LLM calls** — and never executes
commands itself: every command goes through :class:`CommandPolicy` (the gate)
and ``SandboxManager.exec_workspace_command`` (the executor), so the existing
security layer cannot be bypassed.

Baseline awareness: the first verification of a turn records the number of
currently-failing tests as the baseline. Later runs are only considered a
*new* failure when the failure count rises above that baseline, so
pre-existing breakage is reported but never blamed on the agent (a documented
simplification — the first run cannot distinguish pre-existing failures from
ones the agent just introduced, so it is reported honestly as failed).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Hard cap on consecutive failed verifications, enforced by the graph routing
# (infrastructure). The model cannot change it.
MAX_VERIFICATION_ATTEMPTS = 3

# Per-command timeout for verification runs (tests may take longer than the
# normal 30s command policy timeout).
VERIFIER_TIMEOUT = 120

# Capped feedback the agent sees so verification output can never bloat context.
MAX_STDOUT_CHARS = 4000
MAX_STDERR_CHARS = 2000
MAX_FAILURE_LINES = 20

# Generated / cache / VCS directories excluded from mutation snapshots so that
# read-only operations (e.g. running pytest, which writes .pytest_cache) never
# look like a source mutation.
_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "dist", "build", "target", ".cache", "coverage",
}


class VerificationStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class VerificationResult:
    """Structured outcome of one verification run."""

    status: VerificationStatus
    command: str | None = None
    tests_passed: int = 0
    tests_failed: int = 0
    failures: list[str] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    baseline: bool = False


# --------------------------------------------------------------------------
# Mutation detection (infrastructure-level, does not trust the model)
# --------------------------------------------------------------------------


def snapshot_workspace(workspace_path: str) -> dict[str, tuple[int, int]]:
    """Map of relative path -> ``(size_bytes, mtime_ns)`` for source files.

    Generated/cache/VCS directories are excluded. Robust to missing roots and
    unreadable entries. Used to detect whether a ``run_command`` round actually
    mutated the workspace — regardless of what the model claimed.
    """
    root = Path(workspace_path)
    try:
        if not root.is_dir():
            return {}
    except OSError:
        return {}

    snap: dict[str, tuple[int, int]] = {}
    try:
        walker = os.walk(root)
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            base = Path(dirpath)
            for name in filenames:
                p = base / name
                try:
                    st = p.stat()
                except OSError:
                    continue
                try:
                    rel = p.relative_to(root)
                except ValueError:
                    continue
                snap[str(rel).replace("\\", "/")] = (st.st_size, st.st_mtime_ns)
    except OSError:
        pass
    return snap


def snapshot_changed(
    before: dict[str, tuple[int, int]],
    after: dict[str, tuple[int, int]],
) -> bool:
    """True when the workspace snapshot differs (add/edit/remove of source files)."""
    return before != after


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


class WorkspaceVerifier:
    """Deterministic project verifier bound to one workspace directory.

    Selects an existing test/lint/typecheck command for the project, gates it
    through ``CommandPolicy`` (must be ALLOW) and runs it via
    ``SandboxManager.exec_workspace_command`` — never ``subprocess`` directly.
    """

    def __init__(self, workspace_path: str, timeout: int = VERIFIER_TIMEOUT):
        self.workspace_path = workspace_path
        self.timeout = timeout

    def verify(self) -> VerificationResult:
        """Run the best available verification command, or UNAVAILABLE."""
        command = self._select_command()
        if command is None:
            return VerificationResult(status=VerificationStatus.UNAVAILABLE)

        from anzar.agent.policy import CommandPolicy, Verdict
        from anzar.sandbox import get_manager

        policy = CommandPolicy(self.workspace_path)
        decision = policy.evaluate(command)
        if decision.verdict is not Verdict.ALLOW:
            # Never execute a command the policy would not allow.
            return VerificationResult(
                status=VerificationStatus.ERROR,
                command=command,
                failures=[
                    f"verification command not allowed by policy: {decision.reason}"
                ],
            )

        manager = get_manager()
        start = time.monotonic()
        result = manager.exec_workspace_command(
            command,
            workspace_path=self.workspace_path,
            timeout=self.timeout,
        )
        duration_ms = int((time.monotonic() - start) * 1000)

        if result.get("timed_out"):
            return VerificationResult(
                status=VerificationStatus.TIMEOUT,
                command=command,
                stdout=str(result.get("stdout", ""))[:MAX_STDOUT_CHARS],
                stderr=str(result.get("stderr", ""))[:MAX_STDERR_CHARS],
                duration_ms=duration_ms,
            )

        return self._classify(
            command=command,
            exit_code=result.get("exit_code", -1),
            stdout=str(result.get("stdout", "")),
            stderr=str(result.get("stderr", "")),
            duration_ms=duration_ms,
        )

    # -- command selection -------------------------------------------------

    def _select_command(self) -> str | None:
        """Return a verification command, or None when none is meaningful."""
        root = Path(self.workspace_path)
        try:
            root.exists()
        except OSError:
            return None

        package_json = root / "package.json"
        if package_json.is_file():
            return self._node_script(package_json)

        if self._has_python_project(root):
            return self._python_test_command()
        return None

    def _has_python_project(self, root: Path) -> bool:
        markers = (
            "pyproject.toml", "pytest.ini", "setup.cfg", "tox.ini",
            "setup.py", "requirements.txt",
        )
        for name in markers:
            if (root / name).is_file():
                return True
        if (root / "tests").is_dir():
            return True
        return False

    def _python_test_command(self) -> str | None:
        # Prefer the Windows py launcher: a bare `python` may be the Microsoft
        # Store stub (exit 9009). Same probe order as the sandbox exec tests.
        interpreter = next(
            (c for c in ("py", "python", "python3", "pytest") if shutil.which(c)),
            None,
        )
        if interpreter is None:
            return None
        if interpreter == "pytest":
            return "pytest -q"
        return f"{interpreter} -m pytest -q"

    def _node_script(self, package_json: Path) -> str | None:
        """First existing script from test -> lint -> typecheck."""
        try:
            data = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
            scripts = data.get("scripts") or {}
        except (OSError, ValueError):
            return None
        for name in ("test", "lint", "typecheck"):
            if name in scripts:
                return f"npm run {name}"
        return None

    # -- output classification ----------------------------------------------

    def _classify(
        self,
        command: str,
        exit_code: int,
        stdout: str,
        stderr: str,
        duration_ms: int,
    ) -> VerificationResult:
        passed, failed, failures = self._parse_output(stdout, stderr)
        low = f"{stdout}\n{stderr}".lower()
        if failed == 0 and ("no tests ran" in low or "no test files found" in low):
            return VerificationResult(
                status=VerificationStatus.UNAVAILABLE,
                command=command,
                stdout=stdout[:MAX_STDOUT_CHARS],
                stderr=stderr[:MAX_STDERR_CHARS],
                duration_ms=duration_ms,
            )
        if failed > 0:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                command=command,
                tests_passed=passed,
                tests_failed=failed,
                failures=failures,
                stdout=stdout[:MAX_STDOUT_CHARS],
                stderr=stderr[:MAX_STDERR_CHARS],
                duration_ms=duration_ms,
            )
        if exit_code == 0:
            return VerificationResult(
                status=VerificationStatus.PASSED,
                command=command,
                tests_passed=passed,
                tests_failed=failed,
                stdout=stdout[:MAX_STDOUT_CHARS],
                stderr=stderr[:MAX_STDERR_CHARS],
                duration_ms=duration_ms,
            )
        return VerificationResult(
            status=VerificationStatus.ERROR,
            command=command,
            stdout=stdout[:MAX_STDOUT_CHARS],
            stderr=stderr[:MAX_STDERR_CHARS],
            duration_ms=duration_ms,
        )

    @staticmethod
    def _parse_output(stdout: str, stderr: str) -> tuple[int, int, list[str]]:
        """Best-effort parse of pytest/jest/vitest-style summaries.

        Returns ``(passed, failed, failure_lines)``. Simple and documented:
        counts are the maximum seen across summary lines, and failures are
        the ``FAILED ...`` lines (or traceback ``file:line:`` lines) capped
        at MAX_FAILURE_LINES.
        """
        text = f"{stdout}\n{stderr}"

        passed = 0
        for num in re.findall(r"(\d+)\s+passed", text):
            passed = max(passed, int(num))

        failed = 0
        for num in re.findall(r"(\d+)\s+failed", text):
            failed = max(failed, int(num))
        for num in re.findall(r"(\d+)\s+(?:errors?|failing)", text):
            failed = max(failed, int(num))

        failures: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("FAILED "):
                failures.append(stripped[:200])
            if len(failures) >= MAX_FAILURE_LINES:
                break
        if not failures:
            for line in text.splitlines():
                if re.match(r"^[^\s]+:\d+:", line):
                    failures.append(line.strip()[:200])
                if len(failures) >= MAX_FAILURE_LINES:
                    break
        return passed, failed, failures


# --------------------------------------------------------------------------
# Rendering (what the model and the user see)
# --------------------------------------------------------------------------


def verification_message(
    result: VerificationResult,
    attempt: int,
    baseline_failures: int | None,
) -> str:
    """Render a VerificationResult as compact tagged feedback for the agent."""
    lines = [
        f"[VERIFICATION: {result.status.value}] attempt {attempt}/{MAX_VERIFICATION_ATTEMPTS}"
    ]
    if result.command:
        lines.append(f"[COMMAND: {result.command}]")
    lines.append(f"[PASSED: {result.tests_passed}] [FAILED: {result.tests_failed}]")
    if baseline_failures:
        lines.append(
            f"[BASELINE: {baseline_failures} pre-existing failure(s) recorded before "
            "this change — the agent is not blamed for them; only NEW failures matter]"
        )
    if result.failures:
        lines.append("[FAILURES]")
        lines.extend(f"- {f}" for f in result.failures[:5])
        if len(result.failures) > 5:
            lines.append(f"... and {len(result.failures) - 5} more failure(s)")
    elif result.stderr:
        lines.append("[STDERR tail]")
        lines.append(result.stderr[-400:])

    if result.status is VerificationStatus.PASSED:
        lines.append("Verification passed — continue with the work.")
    elif result.status is VerificationStatus.UNAVAILABLE:
        lines.append(
            "No meaningful test/lint/typecheck command found for this project — "
            "not blocking."
        )
    elif result.status in (VerificationStatus.TIMEOUT, VerificationStatus.ERROR):
        lines.append(
            "Verification could not complete (timeout/error) — not blocking, but be cautious."
        )
    else:
        lines.append(
            "Fix the reported failures, then the graph re-runs verification automatically "
            "after the next workspace change."
        )
    return "\n".join(lines)


def verification_failed_summary(result: VerificationResult) -> str:
    """User-facing summary when the attempt cap is exhausted."""
    lines = [
        f"[VERIFICATION FAILED] The workspace change could not pass verification after "
        f"{MAX_VERIFICATION_ATTEMPTS} attempts. Last run: {result.tests_passed} passed, "
        f"{result.tests_failed} failed."
    ]
    if result.failures:
        lines.append("First failures:")
        lines.extend(f"- {f}" for f in result.failures[:10])
    if result.stderr:
        lines.append("Last stderr (tail):")
        lines.append(result.stderr[-500:])
    return "\n".join(lines)
