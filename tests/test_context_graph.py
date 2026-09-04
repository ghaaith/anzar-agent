"""Graph-level tests for context management: HTTP 413 recovery in
``AnzarAgent._invoke_llm_retry`` plus regression checks that normal LLM
requests, tool loops, and PLAN/STATE injection are unchanged.
"""

import os
import shutil
import tempfile

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from anzar.agent.context import (
    ContextBudget,
    PayloadTooLargeError,
    estimate_message_tokens,
    estimate_request_tokens,
)
from anzar.agent.core import AnzarAgent, _CLIMemory
from anzar.agent.llm import friendly_error_message
from anzar.agent.tools import create_tools


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _Http413(Exception):
    status_code = 413

    def __str__(self):
        return "Request body too large (HTTP 413)"


class RecordingLLM:
    """Canned-response LLM that records every message list sent to invoke()."""

    def __init__(self, responses=(), raise_413_on=()):
        self._responses = list(responses)
        self._raise_413_on = set(raise_413_on)
        self.invokes = 0
        self.seen = []
        self.model_name = "mock-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.invokes += 1
        self.seen.append(list(messages))
        if self.invokes in self._raise_413_on:
            raise _Http413()
        if not self._responses:
            return AIMessage(content="done")
        return self._responses.pop(0)


def _tmp_workspace():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "main.py"), "w") as f:
        f.write('print("hello world")')
    return tmp


def _read_result(path: str, lines: int = 80, total: int = 100) -> str:
    body = "\n".join(f"{i + 1:4d} | def fn_{i}() -> int: return {i}" for i in range(lines))
    return f"# {path}: lines 1-{lines} of {total}\n{body}"


def _big_messages():
    msgs = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect the project and fix the bug."),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "a.py"}, "id": "a"}]),
        ToolMessage(content=_read_result("a.py", total=300), tool_call_id="a"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "b.py"}, "id": "b"}]),
        ToolMessage(content=_read_result("b.py", total=400), tool_call_id="b"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "c.py"}, "id": "c"}]),
        ToolMessage(content=_read_result("c.py", total=500), tool_call_id="c"),
        HumanMessage(content="verify the work"),
    ]
    return msgs


def _budget(configured: int) -> ContextBudget:
    """Small-reserve budget for tests: max_tokens == configured - 128."""
    return ContextBudget(
        configured_max=configured,
        output_reserve_tokens=64,
        request_overhead_tokens=64,
        safety_margin_frac=0.0,
    )


def _new_agent(llm, workspace=None, **kwargs):
    ws = workspace or _tmp_workspace()
    return AnzarAgent(
        llm=llm,
        tools=create_tools(ws),
        system_prompt="You are the test agent.",
        memory=_CLIMemory(),
        workspace_path=ws,
        **kwargs,
    )


# --------------------------------------------------------------------------
# HTTP 413 recovery
# --------------------------------------------------------------------------

def test_413_detected_compacted_retried_succeeds():
    ws = _tmp_workspace()
    llm = RecordingLLM(responses=[AIMessage(content="final answer")], raise_413_on=(1,))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    messages = _big_messages()

    response, calls, retries = agent._invoke_llm_retry(llm, messages)

    assert response.content == "final answer"
    assert llm.invokes == 2  # one retry, no infinite loop
    assert retries == 1
    assert calls == 1
    # The retried request was compacted smaller than the one that hit 413.
    assert estimate_message_tokens(llm.seen[1]) < estimate_message_tokens(llm.seen[0])
    shutil.rmtree(ws)


def test_413_exhausted_raises_clean_error_no_infinite_retry():
    ws = _tmp_workspace()
    llm = RecordingLLM(raise_413_on=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))

    with pytest.raises(PayloadTooLargeError) as exc:
        agent._invoke_llm_retry(llm, _big_messages())

    assert llm.invokes == 1 + agent.context_budget.max_retries  # never more than retry cap
    assert "request limit" in str(exc.value)
    # The user-facing message is clean, not raw provider internals.
    message = friendly_error_message(exc.value)
    assert "too large" in message
    assert "413" not in message
    shutil.rmtree(ws)


def test_413_compaction_progress_guard_raises_clean_error():
    ws = _tmp_workspace()
    # Below threshold (identity) and emergency compaction cannot shrink further.
    llm = RecordingLLM(raise_413_on=(1, 2, 3, 4))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    messages = [
        SystemMessage(content="tiny"),
        HumanMessage(content="hello"),
    ]
    with pytest.raises(PayloadTooLargeError):
        agent._invoke_llm_retry(llm, messages)
    assert llm.invokes == 1  # no-progress guard stops before resending
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Regression: normal behavior unchanged
# --------------------------------------------------------------------------

def test_normal_requests_pass_through_unchanged():
    ws = _tmp_workspace()
    llm = RecordingLLM(responses=[AIMessage(content="hi")])
    agent = _new_agent(llm, ws)
    messages = _big_messages()

    response, calls, retries = agent._invoke_llm_retry(llm, messages)

    assert response.content == "hi"
    assert llm.invokes == 1
    assert retries == 0
    shutil.rmtree(ws)


def test_default_budget_is_large_enough_to_passthrough():
    ws = _tmp_workspace()
    llm = RecordingLLM(responses=[AIMessage(content="hi")])
    agent = _new_agent(llm, ws)
    messages = _big_messages()
    assert agent.context_budget.max_tokens == 9417
    assert agent.context_budget.compaction_limit == int(9417 * 0.8)
    assert estimate_request_tokens(messages, (), agent.context_budget) <= (
        agent.context_budget.compaction_limit
    )
    agent._invoke_llm_retry(llm, messages)
    assert llm.seen[0] == messages  # identical message list reached the LLM
    shutil.rmtree(ws)


def test_agent_tool_loop_and_plan_injection_still_work():
    ws = _tmp_workspace()
    with open(os.path.join(ws, "PLAN.md"), "w") as f:
        f.write("# User plan\nBuild the feature.")
    llm = RecordingLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "read_file", "args": {"path": "main.py"}, "id": "t1"}],
            ),
            AIMessage(content="All done."),
        ]
    )
    agent = _new_agent(llm, ws)

    out = agent.run("inspect main.py and report")

    assert out == "All done."
    assert llm.invokes == 2
    # The system message still carries the injected PLAN.md content.
    assert "Build the feature." in llm.seen[0][0].content
    assert isinstance(llm.seen[0][0], SystemMessage)
    shutil.rmtree(ws)


def test_context_budget_kwarg_is_honored():
    ws = _tmp_workspace()
    budget = ContextBudget(configured_max=1234, compaction_threshold=0.5)
    agent = _new_agent(RecordingLLM(), ws, context_budget=budget)
    assert agent.context_budget is budget
    shutil.rmtree(ws)
