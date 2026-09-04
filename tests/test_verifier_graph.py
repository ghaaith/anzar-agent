"""Graph integration tests for mandatory post-modification verification.

Uses the real compiled tool graph with an injected fake verifier and a fake
SandboxManager, so routing/attempt-cap/baseline behavior is tested hermetically
without running real test suites. Mutation detection is real: ``write_file``
mutates the workspace on disk and ``run_command`` mutation is detected via the
workspace snapshot (the fake manager optionally writes a file).
"""

import os
import shutil
import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage

from anzar.agent.core import AnzarAgent, _CLIMemory
from anzar.agent.tools import create_tools
from anzar.agent.verifier import VerificationResult, VerificationStatus


def _tmp_workspace():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "main.py"), "w") as f:
        f.write('print("hello world")')
    return tmp


def _tool_call(name, args, tid):
    return {"name": name, "args": args, "id": tid, "type": "tool_call"}


class _MockLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self.model_name = "mock-model"
        self.invokes = 0
        self.tool_bindings = 0

    def bind_tools(self, tools, **kwargs):
        self.tool_bindings += 1
        return self

    def invoke(self, messages):
        self.invokes += 1
        if not self._responses:
            return AIMessage(content="done")
        return self._responses.pop(0)


class _FakeVerifier:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def verify(self):
        r = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return r


class _FakeManager:
    def __init__(self, stdout="ok", write_file=None):
        self.calls = []
        self.stdout = stdout
        self.write_file = write_file

    def exec_workspace_command(self, command, workspace_path, timeout=30):
        self.calls.append(command)
        if self.write_file:
            path = Path(workspace_path) / self.write_file
            path.write_text("mutated", encoding="utf-8")
        return {"exit_code": 0, "stdout": self.stdout, "stderr": "", "executor": "subprocess"}


def _new_agent(llm, workspace=None, verifier=None, **kwargs):
    ws = workspace or _tmp_workspace()
    return AnzarAgent(
        llm=llm,
        tools=create_tools(ws),
        system_prompt="You are the test agent.",
        memory=_CLIMemory(),
        workspace_path=ws,
        verifier=verifier,
        **kwargs,
    )


def _passed():
    return VerificationResult(status=VerificationStatus.PASSED, tests_passed=5)


def _failed(count, label="FAILED tests/test_x.py::test_x - boom"):
    return VerificationResult(
        status=VerificationStatus.FAILED,
        tests_failed=count,
        failures=[label] * count,
    )


# --------------------------------------------------------------------------
# Triggering: mutations verify, read-only rounds don't
# --------------------------------------------------------------------------


def test_write_file_triggers_verification_and_passes():
    verifier = _FakeVerifier(_passed())
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": "main.py", "content": "print('hi')"}, "w1")]),
            AIMessage(content="done writing"),
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("write main.py in the workspace")
        assert out == "done writing"
        assert verifier.calls == 1
        assert agent.last_metrics.verifications == 1
    finally:
        shutil.rmtree(agent.workspace_path)


def test_read_only_round_skips_verification():
    verifier = _FakeVerifier(_passed())
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("list_files", {}, "l1")]),
            AIMessage(content="listed"),
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("list the files in the workspace")
        assert out == "listed"
        assert verifier.calls == 0
        assert agent.last_metrics.verifications == 0
    finally:
        shutil.rmtree(agent.workspace_path)


def test_mutating_run_command_triggers_verification(monkeypatch):
    ws = _tmp_workspace()
    manager = _FakeManager(write_file="mutated.txt")
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: manager)
    verifier = _FakeVerifier(_passed())
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("run_command", {"command": "python helper.py"}, "r1")]),
            AIMessage(content="ran it"),
        ]
    )
    agent = _new_agent(llm, workspace=ws, verifier=verifier)
    try:
        out = agent.run("run the helper script in the workspace")
        assert out == "ran it"
        assert verifier.calls == 1  # snapshot diff caught the file the command wrote
    finally:
        shutil.rmtree(ws)


def test_readonly_run_command_skips_verification(monkeypatch):
    ws = _tmp_workspace()
    manager = _FakeManager(stdout="hi")
    monkeypatch.setattr("anzar.sandbox.get_manager", lambda: manager)
    verifier = _FakeVerifier(_passed())
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("run_command", {"command": "echo hi"}, "r1")]),
            AIMessage(content="echoed"),
        ]
    )
    agent = _new_agent(llm, workspace=ws, verifier=verifier)
    try:
        out = agent.run("echo hi in the workspace")
        assert out == "echoed"
        assert verifier.calls == 0  # read-only round: no snapshot diff
    finally:
        shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Failure loop + attempt cap (infrastructure-enforced)
# --------------------------------------------------------------------------


def test_verification_failure_loops_and_caps():
    ws = _tmp_workspace()
    verifier = _FakeVerifier(
        _failed(1),
        _failed(2),
        _failed(3),
    )
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": f"f{i}.py", "content": f"x={i}"}, f"w{i}")])
            for i in range(3)
        ]
    )
    agent = _new_agent(llm, workspace=ws, verifier=verifier)
    try:
        out = agent.run("write files in the workspace")
        assert "[VERIFICATION FAILED]" in out
        assert agent.last_metrics.ended_by == "verification_failed"
        assert agent.last_metrics.verifications == 3
        assert verifier.calls == 3
        assert llm.invokes == 3  # no 4th fix round: the graph ENDed, the model can't extend the cap
    finally:
        shutil.rmtree(ws)


def test_pending_failed_forces_reverify_before_end():
    # Agent mutates, verification fails, agent answers with no tools — the
    # graph must NOT end: it re-runs verification on the current workspace.
    verifier = _FakeVerifier(
        _failed(2),
        _passed(),
    )
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": "f0.py", "content": "x=0"}, "w0")]),
            AIMessage(content="fixed it"),
            AIMessage(content="fixed it"),
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("write and fix files in the workspace")
        assert out == "fixed it"
        assert verifier.calls == 2
        assert agent.last_metrics.verifications == 2
        assert agent.last_metrics.ended_by is None
        assert llm.invokes == 3
    finally:
        shutil.rmtree(agent.workspace_path)


def test_pre_existing_failures_not_blamed():
    # Same failures before and after the change: the first run establishes the
    # baseline, the second adds nothing new, so the agent proceeds normally.
    verifier = _FakeVerifier(
        _failed(2, "FAILED tests/test_pre.py::test_pre - broken baseline"),
        _failed(2, "FAILED tests/test_pre.py::test_pre - broken baseline"),
    )
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": "f0.py", "content": "x=0"}, "w0")]),
            AIMessage(content="done"),
            AIMessage(content="done"),
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("write a file in the workspace")
        assert out == "done"
        assert agent.last_metrics.ended_by is None
        assert agent.last_metrics.verifications == 2
    finally:
        shutil.rmtree(agent.workspace_path)


def test_new_failures_above_baseline_engage_loop():
    # First run passes (baseline 0); a later change adds failures -> blocked
    # and capped at 3 attempts like a fresh failure.
    verifier = _FakeVerifier(
        _passed(),
        _failed(3),
        _failed(4),
        _failed(5),
    )
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": f"f{i}.py", "content": f"x={i}"}, f"w{i}")])
            for i in range(4)
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("write files in the workspace")
        assert "[VERIFICATION FAILED]" in out
        assert agent.last_metrics.ended_by == "verification_failed"
        assert agent.last_metrics.verifications == 4  # 1 pass + 3 capped failures
    finally:
        shutil.rmtree(agent.workspace_path)


def test_verification_unavailable_nonblocking():
    verifier = _FakeVerifier(VerificationResult(status=VerificationStatus.UNAVAILABLE))
    llm = _MockLLM(
        [
            AIMessage(content="", tool_calls=[_tool_call("write_file", {"path": "f0.py", "content": "x=0"}, "w0")]),
            AIMessage(content="wrote"),
        ]
    )
    agent = _new_agent(llm, verifier=verifier)
    try:
        out = agent.run("write a file in the workspace")
        assert out == "wrote"
        assert agent.last_metrics.ended_by is None
        assert verifier.calls == 1
    finally:
        shutil.rmtree(agent.workspace_path)
