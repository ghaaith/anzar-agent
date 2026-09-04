"""Phase 10 — regression test reproducing the original HTTP 413 scenario.

A realistic agent conversation accumulates several ``list_files`` /
``read_file`` / test outputs until it exceeds the context budget, then the
user asks to "verify the work". We verify that:

* the oversized conversation is compacted preemptively and completes, and
* if the provider still rejects the request with HTTP 413 after the tool
  results have accumulated, the agent compacts again and recovers.

This is the exact failure the feature exists to prevent, so it must keep
passing.
"""

import os
import shutil
import tempfile

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from anzar.agent.context import ContextBudget, estimate_message_tokens, estimate_request_tokens
from anzar.agent.core import AnzarAgent, _CLIMemory
from anzar.agent.tools import create_tools

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

BIG_FILE_LINES = 150


def _write_big_file(path: str, lines: int = BIG_FILE_LINES) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(f"def fn_{i}() -> int: return {i}" for i in range(lines)))
        f.write("\n")


def _read_result(path: str, total: int = BIG_FILE_LINES) -> str:
    body = "\n".join(
        f"{i + 1:4d} | def fn_{i}() -> int: return {i}" for i in range(total)
    )
    return f"# {path}: lines 1-{total} of {total}\n{body}"


def _budget(configured: int) -> ContextBudget:
    """Small-reserve budget for tests: max_tokens == configured - 128."""
    return ContextBudget(
        configured_max=configured,
        output_reserve_tokens=64,
        request_overhead_tokens=64,
        safety_margin_frac=0.0,
    )


def _realistic_conversation() -> list:
    """System + task + several assistant messages + accumulated tool outputs."""
    msgs = [
        SystemMessage(
            content="You are Anzar, an AI software engineer.\n\n"
            "Working directory: C:\\workspace"
        ),
        HumanMessage(
            content="Add rate limiting to the API client. Inspect the project first."
        ),
        AIMessage(content="I'll start by listing the project structure."),
        AIMessage(
            content="",
            tool_calls=[{"name": "list_files", "args": {"path": "."}, "id": "l1"}],
        ),
        ToolMessage(
            content="Contents of .:\n"
            + "\n".join(f"  [FILE] {n}.py ({10 + i}B)" for i, n in enumerate(["a", "b", "c", "d"]))
            + "\n  [DIR]  tests/",
            tool_call_id="l1",
        ),
        AIMessage(content="Now reading the relevant source files."),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "rate_limiter.py"}, "id": "r1"}],
        ),
        ToolMessage(content=_read_result("rate_limiter.py"), tool_call_id="r1"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "client.py"}, "id": "r2"}],
        ),
        ToolMessage(content=_read_result("client.py"), tool_call_id="r2"),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_file", "args": {"path": "config.py"}, "id": "r3"}],
        ),
        ToolMessage(content=_read_result("config.py"), tool_call_id="r3"),
        AIMessage(content="Files inspected. Running the test suite."),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "run_command", "args": {"command": "py -m pytest -q"}, "id": "t1"}
            ],
        ),
        ToolMessage(
            content="[STATUS: success]\n[EXIT: 0]\n[MS: 240]\n"
            + "\n".join(f"test_case_{i} ... ok" for i in range(60)),
            tool_call_id="t1",
        ),
        AIMessage(content="All tests pass. Ready to verify."),
        HumanMessage(content="verify the work"),
    ]
    return msgs


class _RecordingLLM:
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


class _Http413(Exception):
    status_code = 413


def _new_agent(llm, workspace=None, **kwargs):
    ws = workspace or tempfile.mkdtemp()
    return AnzarAgent(
        llm=llm,
        tools=create_tools(ws),
        system_prompt="You are Anzar, an AI software engineer.",
        memory=_CLIMemory(),
        workspace_path=ws,
        **kwargs,
    )


# --------------------------------------------------------------------------
# Scenario tests
# --------------------------------------------------------------------------

def test_oversized_context_compacts_and_completes():
    """The accumulated conversation exceeds the budget; the agent compacts
    preemptively and completes instead of failing with HTTP 413."""
    ws = tempfile.mkdtemp()
    llm = _RecordingLLM(responses=[AIMessage(content="Verified: all work is complete.")])
    agent = _new_agent(llm, ws, context_budget=_budget(6000))

    conversation = _realistic_conversation()
    assert estimate_request_tokens(conversation, (), agent.context_budget) > (
        agent.context_budget.compaction_limit
    )

    response, calls, retries = agent._invoke_llm_retry(llm, conversation)

    assert response.content == "Verified: all work is complete."
    assert llm.invokes == 1  # never even tried an oversized request
    assert estimate_request_tokens(llm.seen[0], (), agent.context_budget) <= (
        agent.context_budget.max_tokens
    )
    assert estimate_message_tokens(llm.seen[0]) < estimate_message_tokens(conversation)
    # Compaction preserved the current instruction and the recent tool rounds.
    assert llm.seen[0][-1].content == "verify the work"
    shutil.rmtree(ws)


def test_413_after_tool_accumulation_recovers_end_to_end():
    """Full agent loop: tool results accumulate, the next request is rejected
    with HTTP 413, and the agent compacts and retries successfully."""
    ws = tempfile.mkdtemp()
    _write_big_file(os.path.join(ws, "rate_limiter.py"))
    _write_big_file(os.path.join(ws, "client.py"))

    llm = _RecordingLLM(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "rate_limiter.py"}, "id": "r1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "read_file", "args": {"path": "client.py"}, "id": "r2"}
                ],
            ),
            AIMessage(content="Verified: rate limiting works correctly."),
        ],
        raise_413_on=(3,),  # the request carrying both accumulated results
    )
    agent = _new_agent(llm, ws, context_budget=_budget(6000))

    out = agent.run("verify the work")

    assert out == "Verified: rate limiting works correctly."
    assert llm.invokes == 4  # 2 tool rounds + 1 rejected + 1 compacted retry
    # The compacted retry was smaller than the request that hit 413.
    assert estimate_message_tokens(llm.seen[3]) < estimate_message_tokens(llm.seen[2])
    # Tool results were actually accumulated in the rejected request.
    assert any(isinstance(m, ToolMessage) for m in llm.seen[2])
    shutil.rmtree(ws)


def test_long_history_trims_but_completes():
    """A long conversation history is trimmed on replay and the task still
    completes with the latest instruction preserved."""
    ws = tempfile.mkdtemp()
    llm = _RecordingLLM(responses=[AIMessage(content="Finished.")])
    agent = _new_agent(llm, ws, context_budget=_budget(3500))

    history = [SystemMessage(content="You are Anzar, an AI software engineer.")]
    for i in range(20):
        history.append(HumanMessage(content=f"step {i}: inspect and report"))
        history.append(AIMessage(content="Done. " + "details " * 50))
    agent.memory.messages = history  # _CLIMemory in-memory storage

    out = agent.run("verify the work")

    assert out == "Finished."
    request = llm.seen[0]
    assert estimate_request_tokens(request, (), agent.context_budget) <= (
        agent.context_budget.max_tokens
    )
    assert request[-1].content == "verify the work"
    shutil.rmtree(ws)
