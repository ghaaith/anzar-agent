"""V2 agent upgrade tests: fast path, loop caps, retry/backoff, recovery,
human-in-the-loop (ask_user) with on_choice, metrics, mermaid, ChoiceScreen."""

import os
import shutil
import tempfile

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from anzar.agent.core import CANCEL_SENTINEL, AnzarAgent, _CLIMemory
from anzar.agent.errors import (
    ErrorCategory,
    backoff_seconds,
    classify_error,
    intervention_message,
    is_retryable,
)
from anzar.agent.prompts import (
    CHOICE_CANCELLED_MESSAGE,
    FAST_PATH_SYSTEM_PROMPT,
    LOOP_LIMIT_MESSAGE,
)
from anzar.agent.tools import create_tools


def _tmp_workspace():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "main.py"), "w") as f:
        f.write('print("hello world")')
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write("# My Project\nA test project.")
    return tmp


def _mock_llm(*responses, error=None, error_every=False):
    """An LLM whose invoke() pops canned responses (or raises ``error``)."""

    class _LLM:
        def __init__(self, responses, error, error_every):
            self._responses = list(responses)
            self._error = error
            self._error_every = error_every
            self.model_name = "mock-model"
            self.invokes = 0
            self.tool_bindings = 0

        def bind_tools(self, tools, **kwargs):
            self.tool_bindings += 1
            return self

        def invoke(self, messages):
            self.invokes += 1
            if self._error is not None and (self._error_every or self.invokes == 1):
                raise self._error
            if not self._responses:
                return AIMessage(content="done")
            return self._responses.pop(0)

    return _LLM(responses, error, error_every)


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


class _RateLimitError(Exception):
    status_code = 429


# --------------------------------------------------------------------------
# Fast path
# --------------------------------------------------------------------------

def test_fast_path_looks_simple():
    assert AnzarAgent._looks_simple("hello there") is True
    assert AnzarAgent._looks_simple("what is 2+2?") is True
    assert AnzarAgent._looks_simple("run the tests") is False      # tooly verb
    assert AnzarAgent._looks_simple("open main.py") is False        # file extension
    assert AnzarAgent._looks_simple("x" * 400) is False             # too long
    assert AnzarAgent._looks_simple("a = 1; b = 2") is False        # code chars
    # Continuation / project-intent messages must NOT lose their tools.
    assert AnzarAgent._looks_simple("explain this project and continue the work") is False
    assert AnzarAgent._looks_simple("explain the project") is False
    assert AnzarAgent._looks_simple("continue the work") is False
    assert AnzarAgent._looks_simple("resume the task") is False
    print("  [PASS] _looks_simple routes simple chat to fast, tooly to tools")


def test_use_fast_path_requires_real_graph():
    llm = _mock_llm(AIMessage(content="hi"))
    agent = _new_agent(llm)
    assert agent._use_fast_path("hello") is True
    # Replacing graph externally disables fast path (honors stubs).
    from unittest.mock import MagicMock

    agent.graph = MagicMock()
    assert agent._use_fast_path("hello") is False
    shutil.rmtree(agent.workspace_path)


def test_looks_simple_catches_derivatives():
    """Common word derivatives must NOT be routed to the tool-free fast path."""
    assert AnzarAgent._looks_simple("give me full explanation") is False
    assert AnzarAgent._looks_simple("explanation of the code") is False
    assert AnzarAgent._looks_simple("give details") is False
    assert AnzarAgent._looks_simple("show the contents") is False
    assert AnzarAgent._looks_simple("list all the files") is False
    assert AnzarAgent._looks_simple("give me the full overview") is False
    assert AnzarAgent._looks_simple("every file in the project") is False
    assert AnzarAgent._looks_simple("describe the whole thing") is False
    # Still allows genuinely simple chat
    assert AnzarAgent._looks_simple("hello") is True
    assert AnzarAgent._looks_simple("thanks") is True
    assert AnzarAgent._looks_simple("what is rust") is True
    print("  [PASS] _looks_simple catches derivatives like explanation, details, contents")


def test_fast_path_disabled_when_tool_history_exists():
    """If the conversation already has tool results, stay on the tool graph."""
    llm = _mock_llm(AIMessage(content="hi"))
    agent = _new_agent(llm)
    # No tool history → fast path allowed
    assert agent._use_fast_path("hello") is True
    # Simulate a tool result from a prior interaction
    agent.memory.add_user_message("read all files")
    agent.memory.messages.append(
        ToolMessage(content="Contents of main.py: print('hello')", tool_call_id="tc-1")
    )
    # Now even a "simple" message should stay on the tool graph
    assert agent._use_fast_path("hello") is False
    shutil.rmtree(agent.workspace_path)


def test_invalid_tool_call_produces_error_message():
    """When the LLM emits malformed JSON args, the agent recovers gracefully."""
    from langchain_core.messages import InvalidToolCall

    def _llm():
        class _L:
            model_name = "mock"
            invokes = 0

            def bind_tools(self, tools, **kw):
                return self

            def invoke(self, messages):
                self.invokes += 1
                if self.invokes == 1:
                    # First call: one valid + one invalid (bad JSON args)
                    msg = AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call("list_files", {"path": "."}, "c1"),
                        ],
                        invalid_tool_calls=[
                            InvalidToolCall(
                                type="invalid_tool_call",
                                name="read_file",
                                args="{bad json",
                                id="c2",
                                error="Failed to parse tool call arguments as JSON",
                            ),
                        ],
                    )
                    return msg
                return AIMessage(content="Done — recovered from bad args.")

        return _L()

    llm = _llm()
    agent = _new_agent(llm, max_steps=6)
    out = agent.run("list files and read something")
    assert "recovered" in out.lower() or "done" in out.lower()
    assert agent.last_metrics.steps >= 2
    shutil.rmtree(agent.workspace_path)


def test_fast_path_run_uses_fast_graph():
    llm = _mock_llm(AIMessage(content="hi there"))
    agent = _new_agent(llm)
    out = agent.run("hello")
    assert out == "hi there"
    assert agent.last_path == "fast"
    assert agent.last_metrics.path == "fast"
    assert llm.invokes == 1
    # memory saved the exchange
    assert any(isinstance(m, HumanMessage) for m in agent.memory.load_messages())
    assert any(isinstance(m, AIMessage) for m in agent.memory.load_messages())
    shutil.rmtree(agent.workspace_path)


def test_fast_path_uses_fast_prompt():
    seen = {}
    orig = AnzarAgent._fast_messages

    def spy_fast(self, user_message):
        msgs = orig(self, user_message)
        seen["fast_prompt"] = msgs[0].content == FAST_PATH_SYSTEM_PROMPT
        return msgs

    AnzarAgent._fast_messages = spy_fast
    try:
        llm = _mock_llm(AIMessage(content="ok"))
        agent = _new_agent(llm)
        agent.run("hello")
        assert seen.get("fast_prompt") is True
    finally:
        AnzarAgent._fast_messages = orig
    shutil.rmtree(agent.workspace_path)


def test_tooly_message_takes_tool_path():
    llm = _mock_llm(AIMessage(content="listed"))
    agent = _new_agent(llm)
    out = agent.run("run list_files on the workspace please")
    assert out == "listed"
    assert agent.last_path == "tools"
    shutil.rmtree(agent.workspace_path)


# --------------------------------------------------------------------------
# Loop caps
# --------------------------------------------------------------------------

def _tool_call(name, args, tid):
    return {"name": name, "args": args, "id": tid, "type": "tool_call"}


def test_max_steps_loop_limit():
    # Model always requests a tool; no results ever satisfy it -> step cap.
    tc = _tool_call("run_command", {"command": "echo hi"}, "c1")
    llm = _mock_llm(*[AIMessage(content="", tool_calls=[tc]) for _ in range(20)])
    agent = _new_agent(llm, max_steps=3, max_repeats=3)
    out = agent.run("run something repeatedly in the workspace")
    assert LOOP_LIMIT_MESSAGE in out
    assert agent.last_metrics.ended_by == "step_limit"
    shutil.rmtree(agent.workspace_path)


def test_default_max_steps_is_50():
    agent = _new_agent(_mock_llm())
    assert agent.max_steps == 50
    shutil.rmtree(agent.workspace_path)


def test_create_agent_from_config_plumbs_max_steps():
    from anzar.agent.core import create_agent_from_config

    ws = _tmp_workspace()
    try:
        a = create_agent_from_config(
            provider="groq", model=None, api_key="no-key", workspace_path=ws, max_steps=7
        )
        assert a.max_steps == 7
        b = create_agent_from_config(provider="groq", model=None, api_key="no-key", workspace_path=ws)
        assert b.max_steps == 50
    finally:
        shutil.rmtree(ws)


def test_loop_limit_message_mentions_raise_budget():
    assert "/step_limit" in LOOP_LIMIT_MESSAGE


def test_max_repeats_stops_on_repeated_call():
    # Successful identical calls reset the counter — the guard only fires for
    # consecutive *failures*. A loop of successful calls hits step limit instead.
    def _llm():
        class _L:
            model_name = "mock"
            invokes = 0

            def bind_tools(self, tools, **kw):
                return self

            def invoke(self, messages):
                self.invokes += 1
                return AIMessage(
                    content="",
                    tool_calls=[_tool_call("list_files", {}, "c1")],
                )

        return _L()

    llm = _llm()
    agent = _new_agent(llm, max_repeats=2, max_steps=5)
    out = agent.run("list the workspace files, then list them again, and again")
    # Successful repeats don't trip the guard — the step limit stops the loop.
    assert LOOP_LIMIT_MESSAGE in out
    assert agent.last_metrics.ended_by == "step_limit"
    shutil.rmtree(agent.workspace_path)


# --------------------------------------------------------------------------
# Retry / error classification
# --------------------------------------------------------------------------

def test_retry_on_transient_error_then_succeeds():
    llm = _mock_llm(
        AIMessage(content="recovered"), error=_RateLimitError("slow down")
    )
    agent = _new_agent(llm)
    out = agent.run("hello and reply")
    assert out == "recovered"
    assert llm.invokes == 2  # initial + one retry
    assert agent.last_metrics.retries == 1
    shutil.rmtree(agent.workspace_path)


def test_retry_gives_up_after_max_retries():
    llm = _mock_llm(error=_RateLimitError("slow down"), error_every=True)
    agent = _new_agent(llm, max_retries=2)
    try:
        agent.run("hello")
        raise AssertionError("expected a raised error after retries exhausted")
    except _RateLimitError:
        pass
    assert llm.invokes == 3  # 1 initial + 2 retries
    shutil.rmtree(agent.workspace_path)


def test_classify_error_categories():
    class _Auth(Exception):
        status_code = 401

    class _NotFound(Exception):
        status_code = 404

    class _Server(Exception):
        status_code = 500

    class _Timeout(Exception):
        status_code = 524

    class _Conn(Exception):
        pass

    _Conn.__module__ = "httpx"
    _Conn.__name__ = "ConnectionError"

    assert classify_error(_Auth("x")) == ErrorCategory.AUTH
    assert classify_error(_NotFound("x")) == ErrorCategory.CONFIGURATION
    assert classify_error(_RateLimitError("x")) == ErrorCategory.RATE_LIMIT
    assert classify_error(_Server("x")) == ErrorCategory.SERVER
    assert classify_error(_Timeout("x")) == ErrorCategory.TIMEOUT
    assert classify_error(_Conn("x")) == ErrorCategory.CONNECTION
    assert classify_error(ValueError("tool_use_failed oh no")) == ErrorCategory.MALFORMED_TOOL
    assert classify_error(ValueError("bogus")) == ErrorCategory.UNKNOWN
    assert not is_retryable(_Auth("x"))
    assert is_retryable(_RateLimitError("x"))
    assert is_retryable(_Server("x"))
    print("  [PASS] error classification: transient retried, auth/config not")


def test_backoff_seconds():
    assert backoff_seconds(0) == 0.5
    assert backoff_seconds(1) == 1.0
    assert backoff_seconds(2) == 2.0
    assert backoff_seconds(10) == 2.0  # capped
    print("  [PASS] exponential backoff 0.5 -> 1.0 -> 2.0 (capped)")


def test_intervention_message_format():
    msg = intervention_message(
        what="x", why="y", where="z", example_env="K=v"
    )
    assert msg.startswith("I need your intervention.")
    assert "What: x" in msg and "Why: y" in msg
    print("  [PASS] intervention message format")


# --------------------------------------------------------------------------
# Recovery edge
# --------------------------------------------------------------------------

def test_recovery_after_failed_tool():
    ws = _tmp_workspace()

    def _llm():
        class _L:
            model_name = "mock"
            invokes = 0

            def bind_tools(self, tools, **kw):
                return self

            def invoke(self, messages):
                self.invokes += 1
                if self.invokes == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call("run_command", {"command": "exit 1"}, "c1")
                        ],
                    )
                return AIMessage(content="I corrected the approach.")

        return _L()

    llm = _llm()
    agent = _new_agent(llm, workspace=ws, max_steps=6)
    out = agent.run("run a failing command then fix it in the workspace")
    assert out == "I corrected the approach."
    assert agent.last_metrics.errors == 0  # recovery is not an error
    assert agent.last_metrics.steps >= 2
    shutil.rmtree(ws)


def test_hallucinated_tool_name_recovered():
    # Model emits a tool name that isn't registered (e.g. an agentic-skill
    # name like repo_browser.print_tree). The run must not crash; the error
    # feeds back into the loop and the model answers normally.
    def _llm():
        class _L:
            model_name = "mock"
            invokes = 0

            def bind_tools(self, tools, **kw):
                return self

            def invoke(self, messages):
                self.invokes += 1
                if self.invokes == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[
                            _tool_call("repo_browser.print_tree", {"path": "."}, "c1")
                        ],
                    )
                return AIMessage(content="I'll just use the real tools.")

        return _L()

    llm = _llm()
    agent = _new_agent(llm, max_steps=6)
    out = agent.run("list the files in the workspace please")
    assert out == "I'll just use the real tools."
    assert agent.last_metrics.steps >= 2
    assert agent.last_metrics.tool_calls == 1
    shutil.rmtree(agent.workspace_path)


def test_stream_handles_full_aimessage_chunks():
    # Newer langgraph can emit a full AIMessage (not AIMessageChunk) in
    # messages mode; iterating tool_call_chunks on it used to crash stream().
    from langchain_core.messages import AIMessageChunk

    from anzar.agent.core import _tool_call_slices

    chunk = AIMessageChunk(content="", tool_call_chunks=[])
    full = AIMessage(content="", tool_calls=[_tool_call("list_files", {}, "c1")])
    assert _tool_call_slices(chunk) == []
    assert [tc["name"] for tc in _tool_call_slices(full)] == ["list_files"]

    def _llm():
        class _L:
            model_name = "mock"
            invokes = 0

            def bind_tools(self, tools, **kw):
                return self

            def invoke(self, messages):
                self.invokes += 1
                if self.invokes == 1:
                    return AIMessage(
                        content="",
                        tool_calls=[_tool_call("list_files", {"path": "."}, "c1")],
                    )
                return AIMessage(content="streamed reply")

        return _L()

    agent = _new_agent(_llm(), max_steps=4)
    out = "".join(agent.stream("list the files in the workspace please"))
    assert "streamed reply" in out
    assert agent.last_metrics.steps >= 2
    shutil.rmtree(agent.workspace_path)


# --------------------------------------------------------------------------
# Human-in-the-loop (ask_user)
# --------------------------------------------------------------------------

def _ask_user_llm(after_choice):
    class _L:
        model_name = "mock"
        invokes = 0

        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages):
            self.invokes += 1
            if self.invokes == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call(
                            "ask_user",
                            {"question": "pick one", "options": ["a", "b"]},
                            "c1",
                        )
                    ],
                )
            return after_choice()

    return _L()


def test_ask_user_resume_with_choice():
    llm = _ask_user_llm(lambda: AIMessage(content="building option a"))
    agent = _new_agent(llm)
    got = {}

    def on_choice(payload):
        got.update(payload)
        return "a"

    out = agent.run("run this and pick for me", on_choice=on_choice)
    assert out == "building option a"
    assert got.get("question") == "pick one"
    assert got["options"] == ["a", "b"]
    assert agent.last_metrics.interrupts == 1
    assert agent.last_metrics.ended_by is None
    shutil.rmtree(agent.workspace_path)


def test_ask_user_cancel_stops_agent():
    # Without on_choice the choice resolves to the cancel sentinel and the
    # graph ends: the model's second response must never be produced.
    llm = _ask_user_llm(lambda: AIMessage(content="should NOT appear"))
    agent = _new_agent(llm)
    out = agent.run("run this and ask me")
    assert out == CHOICE_CANCELLED_MESSAGE
    assert agent.last_metrics.ended_by == "cancelled"
    assert llm.invokes == 1  # only the ask_user call, no continuation
    shutil.rmtree(agent.workspace_path)


def test_ask_user_empty_choice_is_cancel():
    llm = _ask_user_llm(lambda: AIMessage(content="nope"))
    agent = _new_agent(llm)

    def on_choice(payload):
        return ""  # blank = no confirmation

    out = agent.run("run this and ask me", on_choice=on_choice)
    assert out == CHOICE_CANCELLED_MESSAGE
    assert agent.last_metrics.ended_by == "cancelled"
    shutil.rmtree(agent.workspace_path)


def test_max_interrupts_guard():
    # Model keeps asking even after a choice; interrupt budget caps it.
    class _L:
        model_name = "mock"

        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[
                    _tool_call(
                        "ask_user",
                        {"question": "again?", "options": ["yes", "no"]},
                        "c1",
                    )
                ],
            )

    llm = _L()
    agent = _new_agent(llm, max_interrupts=2)
    out = agent.run("run this and keep asking", on_choice=lambda p: "yes")
    assert out == CHOICE_CANCELLED_MESSAGE
    assert agent.last_metrics.ended_by == "interrupt_limit"
    assert agent.last_metrics.interrupts == 3  # 2 consumed + the guard fire
    shutil.rmtree(agent.workspace_path)


def test_cancel_sentinel_is_not_none():
    # LangGraph 1.2.x crashes on Command(resume=None); our sentinel avoids it.
    assert CANCEL_SENTINEL is not None and CANCEL_SENTINEL != ""
    print("  [PASS] cancel sentinel avoids LangGraph resume=None crash")


# --------------------------------------------------------------------------
# Metrics + observability
# --------------------------------------------------------------------------

def test_last_metrics_and_mermaid():
    llm = _mock_llm(AIMessage(content="hello"))
    agent = _new_agent(llm)
    agent.run("hi")
    m = agent.last_metrics
    assert m is not None
    assert m.path == "fast"
    assert m.llm_calls == 1
    assert m.elapsed_ms >= 0
    assert m.tokens_est > 0
    md = agent.mermaid()
    assert "graph" in md or "flowchart" in md
    assert "fast" in agent.mermaid_fast()
    assert "tool graph" in agent.mermaid_all().lower()
    shutil.rmtree(agent.workspace_path)


def test_stream_events_yields_done_and_tool_events():
    from anzar.tui.events import AgentError, Done, TextDelta

    class _L:
        model_name = "mock"
        invokes = 0

        def bind_tools(self, tools, **kw):
            return self

        def invoke(self, messages):
            self.invokes += 1
            if self.invokes == 1:
                return AIMessage(
                    content="I'll look.",
                    tool_calls=[
                        _tool_call("list_files", {}, "c1"),
                    ],
                )
            return AIMessage(content="Found main.py")

    llm = _L()
    ws = _tmp_workspace()
    agent = _new_agent(llm, workspace=ws)
    events = list(agent.stream_events("list files in the workspace"))
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(texts) == "I'll look.Found main.py"
    assert any(isinstance(e, Done) for e in events)
    assert not any(isinstance(e, AgentError) for e in events)
    assert agent.last_metrics.tool_calls == 1
    shutil.rmtree(ws)


# --------------------------------------------------------------------------
# ChoiceScreen (headless)
# --------------------------------------------------------------------------

def test_choice_screen_headless():
    import asyncio

    from textual.app import App

    from anzar.tui.screens.picker import ChoiceScreen

    results = {}

    class Host(App):
        def compose(self):
            yield from []

        def on_mount(self):
            self.push_screen(
                ChoiceScreen("pick", ["alpha", "beta"]),
                lambda result: results.update({"value": result}),
            )

    async def _smoke():
        app = Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            screen = app.screen
            assert screen is not None
            assert screen.query_one("#choice-options") is not None
            # choose an option, then confirm
            await pilot.press("enter")
            await pilot.pause()
            btn = screen.query_one("#choice-confirm")
            assert screen._chosen == "alpha"
            btn.focus()
            await pilot.press("enter")
            await pilot.pause()

    asyncio.run(_smoke())
    assert results.get("value") == "alpha"
    print("  [PASS] ChoiceScreen: option select + Confirm dismisses with choice")


def test_choice_screen_escape_cancels():
    import asyncio

    from textual.app import App

    from anzar.tui.screens.picker import ChoiceScreen

    results = {}

    class Host(App):
        def on_mount(self):
            self.push_screen(
                ChoiceScreen("pick", ["alpha"]),
                lambda result: results.update({"value": result}),
            )

    async def _smoke():
        app = Host()
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

    asyncio.run(_smoke())
    assert results.get("value") is None  # cancel
    print("  [PASS] ChoiceScreen: Esc dismisses with None (agent stops)")
