"""TUI layer tests: event protocol, stream_events mapping, and headless UI smoke."""

import asyncio
import os
import shutil
import tempfile
import uuid
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from anzar.agent.core import (
    PLAN_FILE,
    PLAN_INJECTION_CAP,
    STATE_FILE,
    TASK_PLAN_NAME,
    TASKS_DIR,
    AnzarAgent,
    _CLIMemory,
)
from anzar.agent.prompts import CHAT_SYSTEM_PROMPT
from anzar.agent.tools import create_tools
from anzar.cli import (
    _cmd_list_keys,
    _mask_api_key,
    _provider_has_key,
    _set_api_key,
)
from anzar.config import AnzarConfig
from anzar.db.models import Settings
from anzar.tui.app import AnzarTui
from anzar.tui.events import (
    AgentError,
    Done,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from anzar.tui.widgets.chat import ChatView, ToolEntry
from anzar.tui.widgets.header import AnzarHeader
from anzar.tui.widgets.inputbar import COMMANDS, InputBar


def _tmp_workspace():
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "main.py"), "w") as f:
        f.write('print("hello")\n')
    return tmp


def _mock_agent(workspace: str) -> AnzarAgent:
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Hello!")
    agent = AnzarAgent(
        llm=mock_llm,
        tools=create_tools(workspace),
        system_prompt=CHAT_SYSTEM_PROMPT.format(
            workspace_path=workspace,
            plan_path=PLAN_FILE,
            state_path=STATE_FILE,
            tasks_dir=TASKS_DIR,
            task_plan_name=TASK_PLAN_NAME,
        ),
        memory=_CLIMemory(),
        workspace_path=workspace,
    )
    return agent


def _write_task(workspace: str, slug: str, plan_body: str = "## Next\n- step\n") -> str:
    """Create tasks/<slug>/task_plan.md and return the absolute folder path."""
    folder = os.path.join(workspace, TASKS_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, TASK_PLAN_NAME), "w", encoding="utf-8") as f:
        f.write(plan_body)
    return folder


def _fake_graph(chunks):
    """Return a fake compiled graph whose .stream yields (mode, chunk) tuples."""

    def _stream(inputs, stream_mode=None, **kwargs):
        return iter(chunks)

    return MagicMock(stream=_stream)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Persistent planning files (PLAN.md / STATE.md / tasks/<slug>/)
# --------------------------------------------------------------------------

def test_load_plan_missing_returns_empty():
    agent = _mock_agent(_tmp_workspace())
    assert agent._load_plan() == ""


def test_load_plan_reads_plan_file():
    tmp = _tmp_workspace()
    with open(os.path.join(tmp, "PLAN.md"), "w", encoding="utf-8") as f:
        f.write("# Plan\n\n## Done\n- scaffolding\n\n## Next\n- feature X\n")
    agent = _mock_agent(tmp)
    plan = agent._load_plan()
    assert "# Plan" in plan
    assert "feature X" in plan


def test_load_plan_truncates_at_cap():
    tmp = _tmp_workspace()
    with open(os.path.join(tmp, "PLAN.md"), "w", encoding="utf-8") as f:
        f.write("x" * 10_000)
    agent = _mock_agent(tmp)
    assert len(agent._load_plan()) == PLAN_INJECTION_CAP


def test_load_state_missing_returns_empty():
    agent = _mock_agent(_tmp_workspace())
    assert agent._load_state() == ""


def test_load_state_reads_pointer():
    tmp = _tmp_workspace()
    with open(os.path.join(tmp, "STATE.md"), "w", encoding="utf-8") as f:
        f.write("Current task: 001-auth-login\nPhase: 2\n")
    agent = _mock_agent(tmp)
    state = agent._load_state()
    assert "001-auth-login" in state
    assert "Phase: 2" in state


def test_active_task_dir_none_without_tasks():
    assert _mock_agent(_tmp_workspace())._active_task_dir() is None


def test_active_task_dir_uses_state_pointer():
    tmp = _tmp_workspace()
    _write_task(tmp, "001-auth-login")
    _write_task(tmp, "002-homepage-bug")
    with open(os.path.join(tmp, "STATE.md"), "w", encoding="utf-8") as f:
        f.write("Current task: 001-auth-login\n")
    agent = _mock_agent(tmp)
    assert agent._active_task_dir().name == "001-auth-login"


def test_active_task_dir_falls_back_to_newest():
    tmp = _tmp_workspace()
    folder = _write_task(tmp, "001-auth-login")
    import time

    time.sleep(0.01)
    _write_task(tmp, "002-homepage-bug")
    agent = _mock_agent(tmp)
    assert agent._active_task_dir().name == "002-homepage-bug"
    assert agent._active_task_dir() != folder


def test_load_task_missing_returns_empty():
    agent = _mock_agent(_tmp_workspace())
    assert agent._load_task() == ""


def test_load_task_reads_active_plan():
    tmp = _tmp_workspace()
    _write_task(tmp, "001-auth-login", "# Goal\n\n## Next\n- fix login\n")
    agent = _mock_agent(tmp)
    task = agent._load_task()
    assert "# Goal" in task
    assert "fix login" in task


def test_load_task_truncates_at_cap():
    tmp = _tmp_workspace()
    _write_task(tmp, "001-auth-login", "x" * 10_000)
    agent = _mock_agent(tmp)
    assert len(agent._load_task()) == PLAN_INJECTION_CAP


def test_build_system_message_includes_plan_only_when_present():
    tmp = _tmp_workspace()
    agent = _mock_agent(tmp)
    assert "fix login" not in agent._build_system_message().content

    with open(os.path.join(tmp, "PLAN.md"), "w", encoding="utf-8") as f:
        f.write("## Next\n- fix login\n")
    content = agent._build_system_message().content
    assert "\n\n## User's app plan\n" in content
    assert "- fix login" in content


def test_build_system_message_includes_task_and_state_blocks():
    tmp = _tmp_workspace()
    _write_task(tmp, "001-auth-login", "# Goal\n\n## Next\n- connect API\n")
    with open(os.path.join(tmp, "STATE.md"), "w", encoding="utf-8") as f:
        f.write("Current task: 001-auth-login\n")
    content = _mock_agent(tmp)._build_system_message().content
    assert "\n\n## Current task plan\n" in content
    assert "connect API" in content
    assert "\n\n## Session pointer\n" in content


# --------------------------------------------------------------------------
# Event dataclasses
# --------------------------------------------------------------------------

def test_event_dataclasses():
    t = TextDelta(text="hi")
    assert t.text == "hi"
    started = ToolStarted(call_id="t1", name="read_file", args={"path": "a.py"})
    assert started.call_id == "t1"
    assert started.args["path"] == "a.py"
    finished = ToolFinished(call_id="t1", name="read_file", ok=True, output="x")
    assert finished.ok
    err = AgentError(message="boom")
    assert "boom" in err.message
    done = Done(text="bye")
    assert done.text == "bye"
    print("  [PASS] Event dataclasses carry their fields")


# --------------------------------------------------------------------------
# stream_events mapping
# --------------------------------------------------------------------------

def test_stream_events_text_and_done():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)
    agent.graph = _fake_graph(
        [
            ("messages", (AIMessageChunk(content="Hel"), {"langgraph_node": "agent"})),
            ("messages", (AIMessageChunk(content="lo!"), {"langgraph_node": "agent"})),
        ]
    )
    events = list(agent.stream_events("hi"))
    texts = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(texts) == "Hello!"
    assert isinstance(events[-1], Done)
    assert events[-1].text == "Hello!"
    assert all(not isinstance(e, AgentError) for e in events)
    shutil.rmtree(ws)
    print("  [PASS] stream_events yields TextDelta chunks + trailing Done")


def test_stream_events_tool_flow():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)
    agent.graph = _fake_graph(
        [
            ("messages", (AIMessageChunk(content="Let me look"), {"langgraph_node": "agent"})),
            (
                "updates",
                {
                    "agent": {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": "t1",
                                        "name": "read_file",
                                        "args": {"path": "main.py"},
                                    }
                                ],
                            )
                        ]
                    }
                },
            ),
            (
                "updates",
                {
                    "tools": {
                        "messages": [
                            ToolMessage(
                                content='print("hello")', tool_call_id="t1", name="read_file"
                            )
                        ]
                    }
                },
            ),
            ("messages", (AIMessageChunk(content=" Done"), {"langgraph_node": "agent"})),
        ]
    )
    events = list(agent.stream_events("read main.py"))
    started = [e for e in events if isinstance(e, ToolStarted)]
    finished = [e for e in events if isinstance(e, ToolFinished)]
    assert len(started) == 1
    assert started[0].name == "read_file"
    assert started[0].args == {"path": "main.py"}
    assert len(finished) == 1
    assert finished[0].call_id == "t1"
    assert finished[0].ok
    assert "hello" in finished[0].output
    assert events[-1].text == "Let me look Done"
    shutil.rmtree(ws)
    print("  [PASS] stream_events announces and finishes tools")


def test_stream_events_error():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)

    def _boom(inputs, stream_mode=None, **kwargs):
        raise RuntimeError("kaboom")

    agent.graph = MagicMock(stream=_boom)
    events = list(agent.stream_events("hi"))
    assert isinstance(events[0], AgentError)
    assert "kaboom" in events[0].message
    assert isinstance(events[-1], Done)
    assert events[-1].text == ""
    shutil.rmtree(ws)
    print("  [PASS] stream_events yields AgentError on failure")


def test_stream_events_empty_response_yields_error():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)
    agent.graph = _fake_graph(
        [
            ("updates", {"agent": {"messages": [AIMessage(content="")]}}),
            ("messages", (AIMessageChunk(content=""), {"langgraph_node": "agent"})),
        ]
    )
    events = list(agent.stream_events("hi"))
    errors = [e for e in events if isinstance(e, AgentError)]
    assert len(errors) == 1
    assert "empty response" in errors[0].message
    assert isinstance(events[-1], Done)
    shutil.rmtree(ws)
    print("  [PASS] stream_events flags an empty response instead of silence")


def test_stream_empty_response_yields_notice():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)
    agent.graph = _fake_graph([("messages", (AIMessageChunk(content=""), {"langgraph_node": "agent"}))])
    chunks = list(agent.stream("hi"))
    assert any("empty response" in c for c in chunks)
    shutil.rmtree(ws)
    print("  [PASS] stream yields a notice on empty response")


def test_run_empty_response_returns_fallback():
    ws = _tmp_workspace()
    agent = _mock_agent(ws)

    def _invoke(inputs):
        return {"messages": [AIMessage(content="")]}

    agent.graph = MagicMock(invoke=_invoke)
    response = agent.run("hi")
    assert "empty response" in response
    shutil.rmtree(ws)
    print("  [PASS] run returns a fallback on empty response")


# --------------------------------------------------------------------------
# API key management (/apikey, /list-keys, _provider_has_key)
# --------------------------------------------------------------------------

def _collect():
    msgs: list[str] = []

    def _r(m: str) -> None:
        msgs.append(m)

    return msgs, _r


def test_mask_api_key():
    assert _mask_api_key(None) == "****"
    assert _mask_api_key("") == "****"
    assert _mask_api_key("skab") == "****"  # len <= 8 → all stars
    assert _mask_api_key("sk1234567890") == "sk12...7890"
    assert _mask_api_key("sk12345678") == "sk12...5678"  # len 9
    print("  [PASS] _mask_api_key masks keys")


def test_set_api_key_stores_and_rebuilds(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq", model=None, api_key=None)
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    msgs, reporter = _collect()

    toolbar = {"provider": config.provider, "model": config.model, "msgs": ""}
    agent, prov, model = _set_api_key(
        db, user, conv, ws, config, "openai", "sk-test-key-12345", toolbar, reporter=reporter
    )

    assert agent is not None
    assert prov == "openai"
    assert config.provider == "openai"
    s = db.query(Settings).filter(Settings.user_id == user.id).first()
    assert s is not None
    assert s.api_key_encrypted == "sk-test-key-12345"
    assert s.provider == "openai"
    assert any("Key stored" in m and "sk-t...2345" in m for m in msgs)
    print("  [PASS] /apikey stores key and rebuilds agent")


def test_provider_has_key_checks_db(monkeypatch):
    db, user, conv = _make_db_and_conv()
    assert _provider_has_key("openai", db, user) is False
    assert _provider_has_key("anthropic", db, user) is False

    config = AnzarConfig(provider="groq")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    _set_api_key(
        db, user, conv, _tmp_workspace(), config, "anthropic", "sk-ant-test-12345",
        {}, reporter=lambda m: None,
    )

    assert _provider_has_key("anthropic", db, user) is True
    assert _provider_has_key("gemini", db, user) is False
    print("  [PASS] _provider_has_key respects stored DB keys")


def test_list_keys_reports_stored(monkeypatch):
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    _set_api_key(
        db, user, conv, _tmp_workspace(), config, "openai", "sk-test-key-12345",
        {}, reporter=lambda m: None,
    )

    out: list[str] = []
    _cmd_list_keys(db, user, reporter=out.append)
    joined = "\n".join(out)
    assert "openai" in joined
    assert "sk-t...2345" in joined
    print("  [PASS] /list-keys reports stored keys")


def test_set_api_key_clear_deletes(monkeypatch):
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    _set_api_key(
        db, user, conv, _tmp_workspace(), config, "openai", "sk-test-key",
        {}, reporter=lambda m: None,
    )
    msgs, reporter = _collect()
    _set_api_key(
        db, user, conv, _tmp_workspace(), config, "openai", "clear", {}, reporter=reporter
    )

    s = db.query(Settings).filter(Settings.user_id == user.id).first()
    assert s.api_key_encrypted is None
    assert any("cleared" in m.lower() for m in msgs)
    print("  [PASS] /apikey clear deletes the key")


def test_set_api_key_unknown_provider_rejected(monkeypatch):
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    msgs, reporter = _collect()
    agent, prov, model = _set_api_key(
        db, user, conv, _tmp_workspace(), config, "nope", "sk-x", {}, reporter=reporter
    )
    assert agent is None
    assert prov == "groq"
    assert any("Unknown provider" in m for m in msgs)
    print("  [PASS] /apikey rejects unknown providers")


def test_set_api_key_ollama_needs_no_key(monkeypatch):
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    msgs, reporter = _collect()
    agent, prov, model = _set_api_key(
        db, user, conv, _tmp_workspace(), config, "ollama", "anything", {}, reporter=reporter
    )
    assert agent is None
    assert msgs, "expected the ollama notice message"
    assert "ollama runs locally" in msgs[0].lower()
    print("  [PASS] /apikey skips ollama")


def test_set_api_key_with_model_applies_chosen_model(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq", model=None, api_key=None)
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    msgs, reporter = _collect()

    toolbar = {"provider": config.provider, "model": config.model, "msgs": ""}
    agent, prov, model = _set_api_key(
        db, user, conv, ws, config, "openai", "sk-test-key-12345", toolbar,
        reporter=reporter, model="gpt-5.6-terra",
    )

    assert agent is not None
    assert prov == "openai"
    assert model == "gpt-5.6-terra"
    assert config.provider == "openai"
    assert config.model == "gpt-5.6-terra"
    assert toolbar["model"] == "gpt-5.6-terra"
    print("  [PASS] /model-first flow stores key and applies the chosen model")


def test_set_api_key_default_model_unchanged(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq", model=None, api_key=None)
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)

    toolbar = {"provider": config.provider, "model": config.model, "msgs": ""}
    agent, prov, model = _set_api_key(
        db, user, conv, ws, config, "openai", "sk-test-key-12345", toolbar,
        reporter=lambda m: None,
    )

    assert agent is not None
    assert model is None  # /apikey path still resolves the provider default
    assert toolbar["model"] == "gpt-5.6-sol"
    print("  [PASS] /apikey path still auto-applies the provider default model")


def test_catalog_each_cloud_provider_lists_latest_three():
    from anzar.agent.providers import SUPPORTED_PROVIDERS

    for pid in ("groq", "gemini", "openai", "anthropic"):
        models = SUPPORTED_PROVIDERS[pid]["models"]
        assert len(models) == 3, f"{pid} should list the latest 3 models, got {models}"

    assert SUPPORTED_PROVIDERS["anthropic"]["models"] == [
        "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8",
    ]
    assert SUPPORTED_PROVIDERS["openai"]["models"][0] == "gpt-5.6-sol"
    print("  [PASS] catalog reflects each company's latest 3 models")


# --------------------------------------------------------------------------
# Headless UI smoke
# --------------------------------------------------------------------------

def _make_db_and_conv():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    import anzar.db.models  # noqa: F401  (register tables)
    from anzar.db.base import Base

    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    user = anzar.db.models.User(
        email=f"tui-{uuid.uuid4().hex}@test.local", name="TUI Tester"
    )
    db.add(user)
    db.flush()
    conv = anzar.db.models.Conversation(user_id=user.id, title="TUI Test Chat")
    db.add(conv)
    db.flush()
    ws = anzar.db.models.Workspace(user_id=user.id, disk_path=os.getcwd())
    db.add(ws)
    db.commit()
    return db, user, conv


def test_tui_mount_and_submit(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    agent = _mock_agent(ws)

    def _fake_events(user_message):
        yield TextDelta(text="Hi")
        yield ToolStarted(call_id="t1", name="list_files", args={})
        yield ToolFinished(call_id="t1", name="list_files", ok=True, output="main.py")
        yield TextDelta(text="!")
        yield Done(text="Hi!")

    monkeypatch.setattr(agent, "stream_events", _fake_events)

    config = AnzarConfig(provider="openrouter", model="openai/gpt-4o", api_key="sk-test")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)

    app = AnzarTui(
        agent=agent,
        config=config,
        db=db,
        user=user,
        conv=conv,
        workspace_path=ws,
        version="0.0.0",
    )

    async def _smoke():
        async with app.run_test(size=(110, 30)) as pilot:
            header = app.query_one(AnzarHeader)
            assert "ready" in header.query_one("#status").classes

            area = app.query_one("#prompt-area")
            area.text = "hello"
            await pilot.press("enter")

            for _ in range(100):
                await pilot.pause()
                if not app._busy:
                    break

            assert app._busy is False
            pill = header.query_one("#status")
            assert "ready" in pill.classes

            chat = app.query_one(ChatView)
            assert chat._active_assistant is None  # finalized by Done
            entries = list(chat.query(ToolEntry))
            assert len(entries) == 1
            assert entries[0]._status == "ok"
            assert app.conv.id == conv.id

    _run(_smoke())
    db.close()
    shutil.rmtree(ws)
    print("  [PASS] TUI mounts, submits, renders events, returns to ready")


def test_step_limit_command_updates_config(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    agent = _mock_agent(ws)
    config = AnzarConfig(provider="groq", api_key=None)
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)
    app = AnzarTui(
        agent=agent,
        config=config,
        db=db,
        user=user,
        conv=conv,
        workspace_path=ws,
        version="0.0.0",
    )
    rebuilt = []
    monkeypatch.setattr(app, "_rebuild_agent", lambda: rebuilt.append(True))
    monkeypatch.setattr(app, "_update_status", lambda: None)
    monkeypatch.setattr(app, "notify", lambda *a, **k: None)

    app._cmd_step_limit("25")
    assert config.max_steps == 25
    assert rebuilt == [True]

    for bad in ("0", "501", "abc", "-3"):
        app._cmd_step_limit(bad)
        assert config.max_steps == 25, bad

    db.close()
    shutil.rmtree(ws)
    print("  [PASS] /step_limit updates config, validates range, rebuilds agent")


def test_config_load_max_steps(monkeypatch, tmp_path):
    from anzar import config as cfg

    home = tmp_path / "home"
    (home / ".anzar").mkdir(parents=True)
    conf = home / ".anzar" / "config.yaml"
    monkeypatch.setattr("pathlib.Path.home", classmethod(lambda cls: home))

    conf.write_text("provider: groq\n", encoding="utf-8")
    monkeypatch.setenv("ANZAR_MAX_STEPS", "17")
    assert cfg.AnzarConfig.load().max_steps == 17

    monkeypatch.delenv("ANZAR_MAX_STEPS")
    conf.write_text("provider: groq\nmax_steps: 33\n", encoding="utf-8")
    assert cfg.AnzarConfig.load().max_steps == 33

    conf.write_text("provider: groq\n", encoding="utf-8")
    assert cfg.AnzarConfig.load().max_steps is None

    monkeypatch.setenv("ANZAR_MAX_STEPS", "not-an-int")
    assert cfg.AnzarConfig.load().max_steps is None
    print("  [PASS] config max_steps loads from env and config.yaml (with fallback)")


def test_tui_search_loads_conversation(monkeypatch):
    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()

    from anzar.db.models import Conversation, Message

    target = Conversation(user_id=user.id, title="Search Target")
    db.add(target)
    db.flush()
    msg = Message(
        conversation_id=conv.id,
        role="user",
        content="find the needle in this haystack",
    )
    msg2 = Message(
        conversation_id=target.id,
        role="assistant",
        content="the needle is here",
    )
    db.add_all([msg, msg2])
    db.commit()

    agent = _mock_agent(ws)
    config = AnzarConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="sk-test")
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)

    app = AnzarTui(
        agent=agent,
        config=config,
        db=db,
        user=user,
        conv=conv,
        workspace_path=ws,
        version="0.0.0",
    )
    monkeypatch.setattr(app, "_rebuild_agent", lambda: None)

    async def _run_test():
        async with app.run_test(size=(110, 30)) as pilot:
            app._cmd_search("needle")
            await pilot.pause()
            app._on_search_row(1)  # row 0 is the current conversation
            await pilot.pause()
            assert app.conv.id == target.id

    _run(_run_test())
    db.close()
    shutil.rmtree(ws)
    print("  [PASS] /search row loads the matching conversation")


# --------------------------------------------------------------------------
# Input bar history + autocomplete
# --------------------------------------------------------------------------

def _inputbar_smoke():
    from textual.app import App

    class Host(App):
        CSS = "InputBar { width: 100%; }"

        def __init__(self):
            super().__init__()
            self.submitted = []

        def compose(self):
            yield InputBar(self._on_sub)

        def _on_sub(self, text):
            self.submitted.append(text)

    async def _smoke():
        app = Host()
        async with app.run_test(size=(80, 10)) as pilot:
            bar = app.query_one(InputBar)
            area = bar._area

            # history
            area.text = "hello"
            await pilot.press("enter")
            assert app.submitted == ["hello"]
            await pilot.press("up")
            assert area.text == "hello"
            await pilot.press("down")
            assert area.text == ""

            # autocomplete — unique match fills the command
            area.text = "/hel"
            await pilot.press("tab")
            assert area.text == "/help"

            # autocomplete — ambiguous match lists options
            area.text = "/m"
            await pilot.press("tab")
            assert area.text == "/m"
            assert len(COMMANDS) > 1

            # multiline: up/down move the cursor instead of cycling history
            area.text = "line1\nline2"
            bar._hpos = -1
            await pilot.press("up")
            assert area.text == "line1\nline2"

    _run(_smoke())


def test_inputbar_history_and_autocomplete():
    _inputbar_smoke()
    print("  [PASS] InputBar history + Tab autocomplete work headlessly")


def test_prompt_api_key_from_model_first_flow(monkeypatch):
    """Model-first flow: keyless provider + chosen model → masked prompt →
    store key → apply the chosen model (exercises the _prompt_api_key closure)."""
    from anzar.tui.screens.picker import KeyInputScreen

    ws = _tmp_workspace()
    db, user, conv = _make_db_and_conv()
    config = AnzarConfig(provider="groq", model=None, api_key=None)
    monkeypatch.setattr(AnzarConfig, "save", lambda self: None)

    app = AnzarTui(
        agent=_mock_agent(ws),
        config=config,
        db=db,
        user=user,
        conv=conv,
        workspace_path=ws,
        version="0.0.0",
    )

    async def _run_test():
        async with app.run_test(size=(110, 30)) as pilot:
            app._prompt_api_key("openai", "gpt-5.6-terra")
            await pilot.pause()
            assert isinstance(app.screen, KeyInputScreen)

            key_input = app.screen.query_one("#key-input")
            key_input.value = "sk-test-key-12345"
            await pilot.press("enter")

            for _ in range(50):
                await pilot.pause()

            assert app.provider == "openai"
            assert app.model == "gpt-5.6-terra"
            s = db.query(Settings).filter(Settings.user_id == user.id).first()
            assert s.api_key_encrypted == "sk-test-key-12345"

    _run(_run_test())
    db.close()
    shutil.rmtree(ws)
    print("  [PASS] model-first flow prompts for key and applies the chosen model")
