"""Phase 3: Agent Core tests."""

import json
import os
import tempfile

from langchain_core.messages import AIMessage, HumanMessage

from anzar.agent.core import AnzarAgent, _CLIMemory
from anzar.agent.prompts import AGENT_SYSTEM_PROMPT, CHAT_SYSTEM_PROMPT
from anzar.agent.tools import create_tools


def _tmp_workspace():
    """Create a temp workspace with a test file."""
    tmp = tempfile.mkdtemp()
    with open(os.path.join(tmp, "main.py"), "w") as f:
        f.write('print("hello world")')
    with open(os.path.join(tmp, "README.md"), "w") as f:
        f.write("# My Project\nA test project.")
    return tmp


def test_create_tools():
    """create_tools returns 6 StructuredTool instances."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    assert len(tools) == 6
    names = [t.name for t in tools]
    assert "run_command" in names
    assert "read_file" in names
    assert "write_file" in names
    assert "list_files" in names
    assert "search_code" in names
    assert "get_file_info" in names
    print("  [PASS] create_tools returns 6 tools")
    import shutil
    shutil.rmtree(ws)


def test_tool_schemas():
    """Tool schemas don't expose workspace_path."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    for t in tools:
        schema = t.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "workspace_path" not in props, f"{t.name} exposes workspace_path in schema"
    print("  [PASS] No tool exposes workspace_path in schema")
    import shutil
    shutil.rmtree(ws)


def test_run_command():
    """run_command executes a shell command."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    rc = next(t for t in tools if t.name == "run_command")
    result = rc.invoke({"command": "echo hello"})
    assert "hello" in result
    print("  [PASS] run_command executes command")
    import shutil
    shutil.rmtree(ws)


def test_read_file():
    """read_file returns numbered file contents."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    rf = next(t for t in tools if t.name == "read_file")
    result = rf.invoke({"path": "main.py"})
    assert "print" in result
    assert "1" in result  # line numbers
    print("  [PASS] read_file returns content with line numbers")
    import shutil
    shutil.rmtree(ws)


def test_read_file_not_found():
    """read_file returns error for missing file."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    rf = next(t for t in tools if t.name == "read_file")
    result = rf.invoke({"path": "nonexistent.py"})
    assert "Error" in result
    print("  [PASS] read_file returns error for missing file")
    import shutil
    shutil.rmtree(ws)


def test_write_file():
    """write_file creates a new file."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    wf = next(t for t in tools if t.name == "write_file")
    result = wf.invoke({"path": "new.py", "content": "x = 1"})
    assert "Successfully wrote" in result
    assert os.path.exists(os.path.join(ws, "new.py"))
    print("  [PASS] write_file creates file")
    import shutil
    shutil.rmtree(ws)


def test_list_files():
    """list_files shows directory contents."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    lf = next(t for t in tools if t.name == "list_files")
    result = lf.invoke({"path": "."})
    assert "main.py" in result
    assert "README.md" in result
    print("  [PASS] list_files shows directory contents")
    import shutil
    shutil.rmtree(ws)


def test_search_code():
    """search_code finds matching lines."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    sc = next(t for t in tools if t.name == "search_code")
    result = sc.invoke({"query": "hello"})
    assert "main.py" in result
    assert "hello" in result
    print("  [PASS] search_code finds matches")
    import shutil
    shutil.rmtree(ws)


def test_get_file_info():
    """get_file_info returns metadata."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    gi = next(t for t in tools if t.name == "get_file_info")
    result = gi.invoke({"path": "main.py"})
    data = json.loads(result)
    assert data["type"] == "file"
    assert data["extension"] == ".py"
    print("  [PASS] get_file_info returns metadata")
    import shutil
    shutil.rmtree(ws)


def test_tool_binds_workspace():
    """Each tool is bound to the correct workspace path."""
    ws = _tmp_workspace()
    tools = create_tools(ws)
    lf = next(t for t in tools if t.name == "list_files")
    # Listing from a subdirectory that exists in our workspace
    result = lf.invoke({"path": "."})
    assert "main.py" in result
    print("  [PASS] Tools are bound to correct workspace")
    import shutil
    shutil.rmtree(ws)


def test_cli_memory():
    """_CLIMemory stores and retrieves messages."""
    mem = _CLIMemory()
    assert mem.load_messages() == []

    mem.add_user_message("hello")
    assert len(mem.load_messages()) == 1
    assert isinstance(mem.load_messages()[0], HumanMessage)

    mem.add_assistant_message("hi there")
    assert len(mem.load_messages()) == 2
    assert isinstance(mem.load_messages()[1], AIMessage)

    mem.clear()
    assert mem.load_messages() == []
    print("  [PASS] _CLIMemory stores and clears messages")


def test_agent_run_mock():
    """AnzarAgent.run works with a mock LLM (no real API key needed)."""
    from unittest.mock import MagicMock

    ws = _tmp_workspace()

    # Create a mock LLM that returns a fixed response
    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="I can help with that!")

    tools = create_tools(ws)
    memory = _CLIMemory()

    agent = AnzarAgent(
        llm=mock_llm,
        tools=tools,
        system_prompt=CHAT_SYSTEM_PROMPT.format(
            workspace_path=ws,
            plan_path="PLAN.md",
            state_path="STATE.md",
            tasks_dir="tasks",
            task_plan_name="task_plan.md",
        ),
        memory=memory,
        workspace_path=ws,
    )

    response = agent.run("hello")
    assert response == "I can help with that!"
    assert len(memory.load_messages()) == 2  # user + assistant
    print("  [PASS] AnzarAgent.run works with mock LLM")

    import shutil
    shutil.rmtree(ws)


def test_agent_tool_call_mock():
    """AnzarAgent.run handles tool calls via the graph."""
    from unittest.mock import MagicMock

    ws = _tmp_workspace()

    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.bind_tools.return_value = mock_llm

    # First call: LLM requests a tool call
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "run_command", "id": "tc1", "args": {"command": "echo test"}}],
    )
    # Second call: LLM gives final response after tool execution
    final_msg = AIMessage(content="The command ran successfully.")

    mock_llm.invoke.side_effect = [tool_call_msg, final_msg]

    tools = create_tools(ws)
    memory = _CLIMemory()

    agent = AnzarAgent(
        llm=mock_llm,
        tools=tools,
        system_prompt=CHAT_SYSTEM_PROMPT.format(
            workspace_path=ws,
            plan_path="PLAN.md",
            state_path="STATE.md",
            tasks_dir="tasks",
            task_plan_name="task_plan.md",
        ),
        memory=memory,
        workspace_path=ws,
    )

    response = agent.run("run echo test")
    assert "successfully" in response.lower()
    print("  [PASS] AnzarAgent handles tool calls via graph")

    import shutil
    shutil.rmtree(ws)


def test_agent_stream_mock():
    """AnzarAgent.stream yields tokens from a mock LLM."""
    from unittest.mock import MagicMock

    from langchain_core.messages import AIMessageChunk

    ws = _tmp_workspace()

    mock_llm = MagicMock()
    mock_llm.model_name = "mock-model"
    mock_llm.bind_tools.return_value = mock_llm
    mock_llm.invoke.return_value = AIMessage(content="Hello!")

    tools = create_tools(ws)
    memory = _CLIMemory()

    agent = AnzarAgent(
        llm=mock_llm,
        tools=tools,
        system_prompt=CHAT_SYSTEM_PROMPT.format(
            workspace_path=ws,
            plan_path="PLAN.md",
            state_path="STATE.md",
            tasks_dir="tasks",
            task_plan_name="task_plan.md",
        ),
        memory=memory,
        workspace_path=ws,
    )

    # Stub the compiled graph to emit tokens the same way LangGraph's
    # stream_mode="messages" does: (AIMessageChunk, metadata) tuples, with
    # ToolMessages interleaved (which must NOT appear in the output).
    def _fake_stream(inputs, stream_mode=None, **kwargs):
        from langchain_core.messages import ToolMessage

        return iter(
            [
                (AIMessageChunk(content="Hello"), {"langgraph_node": "agent"}),
                (
                    ToolMessage(content="tool output must not leak", tool_call_id="x"),
                    {"langgraph_node": "tools"},
                ),
                (AIMessageChunk(content="!"), {"langgraph_node": "agent"}),
            ]
        )

    agent.graph.stream = _fake_stream

    chunks = list(agent.stream("hi"))
    assert len(chunks) > 0
    assert "".join(chunks) == "Hello!"
    print("  [PASS] AnzarAgent.stream yields tokens")

    import shutil
    shutil.rmtree(ws)


def test_prompts_format():
    """System prompts can be formatted with workspace_path and plan paths."""
    fmt = dict(
        workspace_path="/tmp/ws",
        plan_path="PLAN.md",
        state_path="STATE.md",
        tasks_dir="tasks",
        task_plan_name="task_plan.md",
    )
    agent_prompt = AGENT_SYSTEM_PROMPT.format(**fmt)
    assert "/tmp/ws" in agent_prompt
    assert "PLAN.md" in agent_prompt
    assert "STATE.md" in agent_prompt
    assert "task_plan.md" in agent_prompt

    chat_prompt = CHAT_SYSTEM_PROMPT.format(**fmt)
    assert "/tmp/ws" in chat_prompt
    assert "PLAN.md" in chat_prompt
    assert "STATE.md" in chat_prompt
    assert "tasks/<slug>/task_plan.md" in chat_prompt
    print("  [PASS] Prompts format correctly")


if __name__ == "__main__":
    print("\n=== PHASE 3: AGENT CORE ===\n")

    tests = [
        test_create_tools,
        test_tool_schemas,
        test_run_command,
        test_read_file,
        test_read_file_not_found,
        test_write_file,
        test_list_files,
        test_search_code,
        test_get_file_info,
        test_tool_binds_workspace,
        test_cli_memory,
        test_agent_run_mock,
        test_agent_tool_call_mock,
        test_agent_stream_mock,
        test_prompts_format,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'=' * 40}\n")

    if failed:
        exit(1)
