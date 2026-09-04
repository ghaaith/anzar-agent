"""Complete-request context tests.

Verifies the central guarantee behind HTTP 413 prevention: the budget covers
*everything* the provider receives (messages + tool definitions + serialization
overhead + output reserve), and ``AnzarAgent._invoke_llm_retry`` never knowingly
sends an oversized request. Covers preemptive compaction, bounded emergency
compaction on provider 413s, tool-schema accounting, huge single items, repeated
compaction stability, and the original reproduction (many mid-turn
verification SystemMessages).
"""

import os
import shutil
import tempfile

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from anzar.agent.context import (
    ContextBudget,
    PayloadTooLargeError,
    build_request_estimate,
    compact_messages,
    estimate_message_tokens,
    estimate_request_tokens,
    estimate_tool_tokens,
)
from anzar.agent.core import AnzarAgent, _CLIMemory
from anzar.agent.llm import friendly_error_message
from anzar.agent.tools import create_tools, make_ask_user_tool

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

class _Http413(Exception):
    status_code = 413

    def __str__(self):
        return "Request body too large (HTTP 413)"


class _RecordingLLM:
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


def _read_result(path: str, total: int = 100, lines: int = 40) -> str:
    body = "\n".join(f"{i + 1:4d} | def fn_{i}() -> int: return {i}" for i in range(lines))
    return f"# {path}: lines 1-{lines} of {total}\n{body}"


def _tool_round(name: str, args: dict, content: str, call_id: str) -> list[BaseMessage]:
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=content, tool_call_id=call_id),
    ]


def _big_conversation(n_rounds: int = 8) -> list[BaseMessage]:
    """A conversation whose complete request needs compaction at a 4000 budget."""
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect the project and fix the bug."),
    ]
    for i in range(n_rounds):
        msgs.extend(_tool_round("read_file", {"path": f"f{i}.py"}, _read_result(f"f{i}.py"), f"c{i}"))
    msgs.append(AIMessage(content="I inspected the files. Now verifying."))
    msgs.append(HumanMessage(content="verify the work"))
    return msgs


def _verification_heavy_conversation(n_rounds: int = 8) -> list[BaseMessage]:
    """The original failure shape: tool rounds interleaved with mid-turn
    verification SystemMessages that previously accumulated unboundedly."""
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect and fix the bug."),
    ]
    for i in range(n_rounds):
        msgs.extend(_tool_round("read_file", {"path": f"f{i}.py"}, _read_result(f"f{i}.py"), f"c{i}"))
        msgs.append(SystemMessage(content=f"verification check {i}: feedback for round {i}"))
    msgs.append(AIMessage(content="I inspected the files. Now verifying."))
    msgs.append(HumanMessage(content="verify the work"))
    return msgs


def _snapshot(messages):
    return [(type(m).__name__, str(m.content)[:40]) for m in messages]


# --------------------------------------------------------------------------
# Complete-request budgeting (never send over budget)
# --------------------------------------------------------------------------

def test_never_sends_over_budget_with_bound_tools():
    ws = _tmp_workspace()
    llm = _RecordingLLM(responses=[AIMessage(content="done")])
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    bound_tools = [*agent.tools, make_ask_user_tool()]
    msgs = _big_conversation()

    response, calls, retries = agent._invoke_llm_retry(llm, msgs, tools=bound_tools)

    assert response.content == "done"
    assert llm.invokes == 1  # preemptive compaction handled it; no 413 retries
    # Tool definitions made the complete request tip over the threshold, so the
    # sent view is smaller than the original conversation.
    assert estimate_message_tokens(llm.seen[0]) < estimate_message_tokens(msgs)
    for sent in llm.seen:
        est = build_request_estimate(sent, bound_tools, agent.context_budget)
        assert est.total <= est.max_tokens  # never knowingly oversized
    assert agent.last_request_estimate.fits
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Preemptive compaction + emergency compaction on provider 413
# --------------------------------------------------------------------------

def test_413_recovers_after_preemptive_compaction_with_tools():
    ws = _tmp_workspace()
    llm = _RecordingLLM(responses=[AIMessage(content="done")], raise_413_on=(1,))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    bound_tools = [*agent.tools, make_ask_user_tool()]
    msgs = _big_conversation()

    response, calls, retries = agent._invoke_llm_retry(llm, msgs, tools=bound_tools)

    assert response.content == "done"
    assert llm.invokes == 2  # one 413, one compacted retry
    assert retries == 1
    # The provider rejected the preemptively-compacted request, so the
    # emergency compaction went deeper and sent something smaller.
    assert estimate_message_tokens(llm.seen[1]) < estimate_message_tokens(llm.seen[0])
    # Every request that actually reached the provider fit the budget, with the
    # complete request accounted (tool schemas included).
    for sent in llm.seen:
        est = build_request_estimate(sent, bound_tools, agent.context_budget)
        assert est.total <= est.max_tokens
    shutil.rmtree(ws)


def test_413_twice_then_success_each_request_fits():
    ws = _tmp_workspace()
    llm = _RecordingLLM(responses=[AIMessage(content="ok")], raise_413_on=(1, 2))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    msgs = _big_conversation()

    response, calls, retries = agent._invoke_llm_retry(llm, msgs)

    assert response.content == "ok"
    assert llm.invokes == 3
    assert retries == 2
    # Each retried request is strictly smaller than the previous one.
    assert estimate_message_tokens(llm.seen[2]) < estimate_message_tokens(llm.seen[1])
    assert estimate_message_tokens(llm.seen[1]) < estimate_message_tokens(llm.seen[0])
    # Every request that was actually sent fit its budget — never an oversized
    # body. Emergency budgets are strictly tighter than the original, so fitting
    # the original budget also proves each retry was safe.
    for sent in llm.seen:
        est = build_request_estimate(sent, (), agent.context_budget)
        assert est.total <= est.max_tokens
    # The latest user instruction survived even the deepest emergency view.
    assert llm.seen[2][-1].content == "verify the work"
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Persistent 413: bounded retries + clean error + canonical state intact
# --------------------------------------------------------------------------

def test_persistent_413_bounded_clean_error_and_state_intact():
    ws = _tmp_workspace()
    llm = _RecordingLLM(raise_413_on=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    msgs = _big_conversation()
    before = _snapshot(msgs)

    with pytest.raises(PayloadTooLargeError) as exc:
        agent._invoke_llm_retry(llm, msgs)

    # Bounded: the retry cap caps total invocations.
    assert llm.invokes == 1 + agent.context_budget.max_retries
    assert _snapshot(msgs) == before  # canonical history never mutated
    message = friendly_error_message(exc.value)
    assert "too large" in message
    assert "413" not in message
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Huge single item: fail cleanly before sending, never resend identical
# --------------------------------------------------------------------------

def test_huge_single_message_fails_before_send():
    ws = _tmp_workspace()
    llm = _RecordingLLM(raise_413_on=(1, 2, 3))
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    # A huge *latest* HumanMessage cannot be compacted (huge tool results are
    # summarized and can fit), so the preflight fit-or-raise must reject it
    # before anything reaches the provider.
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect."),
        HumanMessage(content="x" * 60000),
    ]
    before = _snapshot(msgs)

    with pytest.raises(PayloadTooLargeError) as exc:
        agent._invoke_llm_retry(llm, msgs)

    assert llm.invokes == 0  # fit-or-raise prevented the oversized send entirely
    assert _snapshot(msgs) == before
    assert "too large" in friendly_error_message(exc.value)
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Tool-schema accounting + observability
# --------------------------------------------------------------------------

def test_tool_schema_accounting_reflected_in_metrics():
    ws = _tmp_workspace()
    llm = _RecordingLLM(responses=[AIMessage(content="done")])
    agent = _new_agent(llm, ws, context_budget=_budget(6000))
    bound_tools = [*agent.tools, make_ask_user_tool()]
    msgs = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="hello"),
    ]

    response, calls, retries = agent._invoke_llm_retry(llm, msgs, tools=bound_tools)

    assert response.content == "done"
    est = agent.last_request_estimate
    assert est.tool_tokens == estimate_tool_tokens(bound_tools)
    assert est.tool_tokens > 0  # real bound tools are accounted
    assert est.total == estimate_request_tokens(msgs, bound_tools, agent.context_budget)
    assert est.total > estimate_request_tokens(msgs, (), agent.context_budget)
    assert est.fits
    assert est.to_dict()["tools"] == est.tool_tokens
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# Repeated compaction: stays bounded, preserves task context, no note stacking
# --------------------------------------------------------------------------

def test_repeated_compaction_stays_bounded_preserving_task():
    msgs = _big_conversation()
    b1 = _budget(4000)
    once = compact_messages(msgs, b1)
    assert estimate_request_tokens(once, (), b1) <= b1.max_tokens

    b2 = b1.compacted(0.5)
    twice = compact_messages(once, b2)
    assert estimate_request_tokens(twice, (), b2) <= b2.max_tokens
    assert estimate_message_tokens(twice) <= estimate_message_tokens(once)
    # Base system prompt and latest user instruction survive repeated passes.
    assert twice[0].content == "You are the test agent."
    assert twice[-1].content == "verify the work"
    # The compaction note is deduplicated — never stacked across passes.
    notes = [
        m for m in twice
        if isinstance(m, SystemMessage) and "Context was compacted" in str(m.content)
    ]
    assert len(notes) <= 1


# --------------------------------------------------------------------------
# Original reproduction: verification/recovery SystemMessages accumulate
# --------------------------------------------------------------------------

def test_verification_heavy_conversation_compacts_and_recovers():
    ws = _tmp_workspace()
    llm = _RecordingLLM(
        responses=[AIMessage(content="all verified")], raise_413_on=(1,)
    )
    agent = _new_agent(llm, ws, context_budget=_budget(4000))
    msgs = _verification_heavy_conversation()

    response, calls, retries = agent._invoke_llm_retry(llm, msgs)

    assert response.content == "all verified"
    assert llm.invokes == 2  # preemptively compacted, then one 413 recovery
    sent = llm.seen[1]
    # The most recent verification feedback survives; stale feedback is stubbed
    # away instead of forming an uncompactable floor.
    contents = [str(m.content) for m in sent]
    assert any("verification check 7" in c for c in contents)
    assert not any("verification check 0" in c for c in contents)
    assert sent[0].content == "You are the test agent."
    assert any(m.content == "verify the work" for m in sent if isinstance(m, HumanMessage))
    assert estimate_request_tokens(sent, (), agent.context_budget) <= (
        agent.context_budget.max_tokens
    )
    shutil.rmtree(ws)
