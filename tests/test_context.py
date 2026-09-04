"""Unit tests for the central context manager (anzar.agent.context).

Covers token estimation, the ContextBudget, deterministic tool-result
compaction, preemptive message compaction (preservation + pair-consistency),
the known-state digest, and HTTP 413 detection.
"""

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from anzar.agent.context import (
    DEFAULT_COMPACTION_THRESHOLD,
    DEFAULT_EMERGENCY_RATIO,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_OUTPUT_RESERVE_TOKENS,
    DEFAULT_REQUEST_OVERHEAD_TOKENS,
    DEFAULT_SAFETY_MARGIN_FRAC,
    _SYSTEM_STUB,
    ContextBudget,
    PayloadTooLargeError,
    build_request_estimate,
    build_state_digest,
    compact_messages,
    compact_tool_output,
    estimate_message_tokens,
    estimate_request_tokens,
    estimate_tool_tokens,
    estimate_tokens,
    is_payload_too_large,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

READ_BODY_LINES = 80


class _EmptySchema:
    @staticmethod
    def model_json_schema():
        return {"type": "object", "properties": {}}


class _FakeTool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.args_schema = _EmptySchema()


def _budget(configured: int, *, threshold: float = 0.8) -> ContextBudget:
    """Small-reserve budget for tests: max_tokens == configured - 128."""
    return ContextBudget(
        configured_max=configured,
        output_reserve_tokens=64,
        request_overhead_tokens=64,
        safety_margin_frac=0.0,
        compaction_threshold=threshold,
    )


def _read_result(path: str, total: int = 100, lines: int = READ_BODY_LINES) -> str:
    body = "\n".join(f"{i + 1:4d} | def fn_{i}() -> int: return {i}" for i in range(lines))
    return f"# {path}: lines 1-{lines} of {total}\n{body}"


def _run_result(out: str = "1 passed", status: str = "success", exit_code: int = 0) -> str:
    return f"[STATUS: {status}]\n[EXIT: {exit_code}]\n[MS: 12]\n{out}"


def _listing(path: str = ".", n: int = 40) -> str:
    entries = "\n".join(
        f"  [FILE] {name}.py ({10 + i}B)" for i, name in enumerate([f"f{i}" for i in range(n)])
    )
    return f"Contents of {path}:\n{entries}"


def _tool_round(name: str, args: dict, content: str, call_id: str) -> list[BaseMessage]:
    return [
        AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}]),
        ToolMessage(content=content, tool_call_id=call_id),
    ]


def _conversation(n_rounds: int = 3) -> list[BaseMessage]:
    msgs: list[BaseMessage] = [SystemMessage(content="You are the test agent.")]
    msgs.append(HumanMessage(content="Inspect the project and fix the bug."))
    for i in range(n_rounds):
        msgs.extend(_tool_round("read_file", {"path": f"f{i}.py"}, _read_result(f"f{i}.py"), f"c{i}"))
    msgs.append(AIMessage(content="I inspected the files. Now verifying."))
    msgs.append(HumanMessage(content="verify the work"))
    return msgs


def _mid_system_messages() -> list[BaseMessage]:
    """Base system + one large round + small rounds + mid-turn verification notes.

    The large early round is what compaction actually shrinks, so level 0
    (keep the two most recent notes verbatim) is strictly smaller than the
    input — without it, the digest note alone would make level 0 *larger*
    than the conversation and level 0 could never be selected.
    """
    return [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect the project."),
        *_tool_round("read_file", {"path": "big.py"}, _read_result("big.py", total=300), "big"),
        *_tool_round("read_file", {"path": "s1.py"}, "small result one", "s1"),
        *_tool_round("read_file", {"path": "s2.py"}, "small result two", "s2"),
        SystemMessage(content="verification note 1 (old)"),
        SystemMessage(content="verification note 2"),
        SystemMessage(content="verification note 3 (recent)"),
        HumanMessage(content="verify the work"),
    ]


def _is_paired(messages: list[BaseMessage]) -> bool:
    """No orphan ToolMessage (each has a preceding AIMessage with its id)."""
    known: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            known.update(tc.get("id") for tc in msg.tool_calls if tc.get("id"))
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id not in known:
                return False
    return True


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def test_estimate_tokens_is_char_based():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("x" * 1000) == 333
    assert estimate_tokens("x" * 1001) == 333


def test_estimate_message_tokens_counts_content_and_tool_args():
    msgs = [
        SystemMessage(content="abc"),
        AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"path": "x.py"}, "id": "1"}]),
        ToolMessage(content="a" * 400, tool_call_id="1"),
    ]
    expected = (len("abc") + len('{"path": "x.py"}') + 400) // 3
    assert estimate_message_tokens(msgs) == expected


def test_estimate_tool_tokens_counts_schema():
    tool = _FakeTool(name="read_file", description="Read a file")
    expected = (
        len("read_file")
        + len("Read a file")
        + len('{"properties": {}, "type": "object"}')
    ) // 3
    assert estimate_tool_tokens([tool]) == expected
    assert estimate_tool_tokens([tool]) == estimate_tool_tokens([tool])  # cached/stable


def test_estimate_request_tokens_includes_tools_and_reserves():
    msgs = [HumanMessage(content="x" * 900)]
    budget = ContextBudget(
        configured_max=4000,
        output_reserve_tokens=100,
        request_overhead_tokens=50,
        safety_margin_frac=0.0,
    )
    tool = _FakeTool(name="t", description="y" * 300)
    assert estimate_request_tokens(msgs, (), budget) == 300 + 100 + 50
    assert estimate_request_tokens(msgs, [tool], budget) == (
        estimate_request_tokens(msgs, (), budget) + estimate_tool_tokens([tool])
    )


def test_build_request_estimate_breakdown():
    msgs = [HumanMessage(content="abc")]
    budget = ContextBudget(
        configured_max=4000,
        output_reserve_tokens=100,
        request_overhead_tokens=50,
        safety_margin_frac=0.0,
    )
    est = build_request_estimate(msgs, (), budget)
    assert est.messages_tokens == 1
    assert est.tool_tokens == 0
    assert est.overhead_tokens == 50
    assert est.output_reserve_tokens == 100
    assert est.total == 151
    assert est.fits  # 151 <= max_tokens 3850
    assert est.to_dict()["total"] == 151
    assert est.to_dict()["safe"] == 3850
    assert est.to_dict()["tools"] == 0


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

def test_budget_defaults():
    budget = ContextBudget()
    assert budget.cap == DEFAULT_MAX_CONTEXT_TOKENS == 12000
    assert budget.max_tokens == 9417  # (12000 - 1024 - 512) * 0.9
    assert budget.output_reserve_tokens == DEFAULT_OUTPUT_RESERVE_TOKENS == 1024
    assert budget.request_overhead_tokens == DEFAULT_REQUEST_OVERHEAD_TOKENS == 512
    assert budget.safety_margin_frac == DEFAULT_SAFETY_MARGIN_FRAC == 0.10
    assert budget.compaction_threshold == DEFAULT_COMPACTION_THRESHOLD == 0.8
    assert budget.emergency_ratio == DEFAULT_EMERGENCY_RATIO == 0.5
    assert budget.compaction_limit == int(9417 * 0.8)


def test_budget_compacted_ratios():
    budget = ContextBudget(configured_max=4000)
    assert budget.max_tokens == 2217
    assert budget.compacted().max_tokens == 1108
    assert budget.compacted(0.25).max_tokens == 554
    assert budget.max_tokens == 2217  # original unchanged


def test_budget_for_provider_caps():
    assert ContextBudget.for_provider("groq").max_tokens == 9417  # capped by configured 12000
    assert ContextBudget.for_provider("openai").max_tokens == 9417
    assert ContextBudget.for_provider("ollama").max_tokens == 5990  # min(12000, 8192)
    assert ContextBudget.for_provider(None).max_tokens == 9417
    assert ContextBudget.for_provider("unknown-provider").max_tokens == 9417


def test_budget_from_env(monkeypatch):
    monkeypatch.setenv("ANZAR_MAX_CONTEXT_TOKENS", "4000")
    monkeypatch.setenv("ANZAR_CONTEXT_COMPACTION_THRESHOLD", "0.5")
    budget = ContextBudget.from_env()
    assert budget.cap == 4000
    assert budget.max_tokens == 2217  # (4000 - 1536) * 0.9
    assert budget.compaction_threshold == 0.5
    assert budget.compaction_limit == int(2217 * 0.5)
    monkeypatch.delenv("ANZAR_MAX_CONTEXT_TOKENS")
    monkeypatch.delenv("ANZAR_CONTEXT_COMPACTION_THRESHOLD")
    assert ContextBudget.from_env().max_tokens == 9417


def test_budget_from_env_overrides_reserves_and_margin(monkeypatch):
    monkeypatch.setenv("ANZAR_CONTEXT_OUTPUT_RESERVE_TOKENS", "200")
    monkeypatch.setenv("ANZAR_CONTEXT_OVERHEAD_TOKENS", "100")
    monkeypatch.setenv("ANZAR_CONTEXT_SAFETY_MARGIN", "0.2")
    budget = ContextBudget.from_env()
    assert budget.output_reserve_tokens == 200
    assert budget.request_overhead_tokens == 100
    assert budget.safety_margin_frac == 0.2
    assert budget.max_tokens == int((12000 - 300) * 0.8)


# --------------------------------------------------------------------------
# Tool-result compaction
# --------------------------------------------------------------------------

def test_compact_read_file_preserves_header_and_lines():
    result = _read_result("rate_limiter.py", total=78)
    out = compact_tool_output(result, limit=300)
    assert out.startswith("# rate_limiter.py: lines 1-80 of 78")
    assert "[compacted" in out
    assert "def fn_0" in out  # real first line preserved
    assert len(out) <= 300
    assert len(out) < len(result)


def test_compact_run_command_preserves_tags():
    out = compact_tool_output(_run_result(out="x" * 5000), limit=600)
    assert out.startswith("[STATUS: success]")
    assert "[EXIT: 0]" in out
    assert "[MS: 12]" in out
    assert "[COMPACTED" in out
    assert len(out) <= 600


def test_compact_list_files_preserves_header_and_count():
    out = compact_tool_output(_listing(n=40), limit=300)
    assert out.startswith("Contents of .:")
    assert "40" in out and "more entries" in out
    assert len(out) <= 300


def test_compact_generic_and_small_content_untouched():
    generic = "line one\n" + "x" * 2000 + "\nline last"
    out = compact_tool_output(generic, limit=500)
    assert len(out) <= 500
    assert out.startswith("line one")
    assert out.endswith("line last")
    assert "[compacted]" in out
    small = "tiny result"
    assert compact_tool_output(small, limit=100) == small


def test_compact_empty():
    assert compact_tool_output("") == ""


# --------------------------------------------------------------------------
# State digest
# --------------------------------------------------------------------------

def test_build_state_digest_from_real_results():
    msgs = [
        *_tool_round("read_file", {"path": "rate_limiter.py"}, _read_result("rate_limiter.py", total=78), "a"),
        *_tool_round("read_file", {"path": "client.py"}, _read_result("client.py", total=120), "b"),
        *_tool_round("run_command", {"command": "py -m pytest -q"}, _run_result("5 passed"), "c"),
        *_tool_round("list_files", {"path": "."}, _listing(n=10), "d"),
    ]
    digest = build_state_digest(msgs)
    assert "read_file(rate_limiter.py): 78 lines" in digest
    assert "read_file(client.py): 120 lines" in digest
    assert "run_command: status=success exit=0" in digest
    assert "list_files(.): 10 entries" in digest
    assert build_state_digest([HumanMessage(content="hi")]) == ""


# --------------------------------------------------------------------------
# Message compaction
# --------------------------------------------------------------------------

def test_compact_messages_is_noop_below_threshold():
    msgs = _conversation(n_rounds=1)
    budget = _budget(100000)
    result = compact_messages(msgs, budget)
    assert result is msgs  # identical object: zero behavior change


def test_compact_messages_preserves_system_and_latest_instruction():
    msgs = _conversation(n_rounds=3)
    budget = _budget(3000)
    result = compact_messages(msgs, budget)
    assert result is not msgs
    assert estimate_message_tokens(result) < estimate_message_tokens(msgs)
    assert any(isinstance(m, SystemMessage) for m in result)
    system_first = next(m for m in result if isinstance(m, SystemMessage))
    assert system_first.content == "You are the test agent."
    humans = [m for m in result if isinstance(m, HumanMessage)]
    assert humans and humans[-1].content == "verify the work"
    assert _is_paired(result)


def test_compact_messages_recent_round_stays_detailed():
    msgs = _conversation(n_rounds=3)
    budget = _budget(3000)
    result = compact_messages(msgs, budget)
    # The most recent round's tool result must be preserved verbatim.
    last_tool = next(
        (m for m in reversed(result) if isinstance(m, ToolMessage) and m.tool_call_id == "c2"), None
    )
    original = next(m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert last_tool is not None
    assert last_tool.content == original.content


def test_compact_messages_summarizes_old_tool_output():
    # Old round is large; the two recent rounds are small, so compaction stops
    # at level 0 and the old result is summarized (not dropped).
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect the project."),
        *_tool_round("read_file", {"path": "big.py"}, _read_result("big.py", total=300), "old"),
        *_tool_round("read_file", {"path": "small1.py"}, "small result one", "c1"),
        *_tool_round("read_file", {"path": "small2.py"}, "small result two", "c2"),
        HumanMessage(content="verify the work"),
    ]
    budget = _budget(1000)
    result = compact_messages(msgs, budget)
    old = next((m for m in result if isinstance(m, ToolMessage) and m.tool_call_id == "old"), None)
    original = next(m for m in msgs if isinstance(m, ToolMessage) and m.tool_call_id == "old")
    assert old is not None  # old round retained as a compact summary
    assert len(old.content) < len(original.content)
    assert "[compacted" in str(old.content)
    recent = next(m for m in result if isinstance(m, ToolMessage) and m.tool_call_id == "c2")
    assert recent.content == "small result two"  # recent result stays verbatim


def test_compact_messages_drops_old_rounds_in_pairs():
    msgs = _conversation(n_rounds=4)
    budget = _budget(2000)
    result = compact_messages(msgs, budget)
    assert estimate_message_tokens(result) <= budget.compaction_limit
    ids_present = {m.tool_call_id for m in result if isinstance(m, ToolMessage)}
    assert ids_present  # at least the current round remains
    assert _is_paired(result)


def test_compact_messages_adds_digest_note():
    msgs = _conversation(n_rounds=3)
    budget = _budget(3000)
    result = compact_messages(msgs, budget)
    note = next(
        (m for m in result if isinstance(m, SystemMessage) and "Context was compacted" in m.content),
        None,
    )
    assert note is not None
    assert "Known workspace state" in note.content
    assert "read_file(f0.py)" in note.content


def test_compact_messages_never_mutates_input():
    msgs = _conversation(n_rounds=3)
    before = [(type(m).__name__, str(m.content)[:40]) for m in msgs]
    compact_messages(msgs, _budget(3000))
    after = [(type(m).__name__, str(m.content)[:40]) for m in msgs]
    assert before == after


def test_compact_messages_compacts_mid_turn_system_messages():
    msgs = _mid_system_messages()
    budget = _budget(1500)
    result = compact_messages(msgs, budget)
    mid = [m for m in result if isinstance(m, SystemMessage)][1:]
    contents = [m.content for m in mid]
    assert "verification note 3 (recent)" in contents  # latest feedback verbatim
    assert "verification note 2" in contents
    assert "verification note 1 (old)" not in contents  # older feedback stubbed
    assert _SYSTEM_STUB in contents
    assert result[0].content == "You are the test agent."  # base system verbatim
    assert _is_paired(result)


def test_compact_messages_deep_level_drops_stale_mid_system():
    # A larger conversation (four 80-line rounds) forces compaction past the
    # level that keeps two recent notes verbatim, so the mid-turn notes are
    # stubbed away instead of forming an uncompactable floor.
    conv = _conversation(n_rounds=4)
    msgs: list[BaseMessage] = list(conv[:-2])
    msgs.extend(
        [
            SystemMessage(content="verification note 1 (old)"),
            SystemMessage(content="verification note 2"),
            SystemMessage(content="verification note 3 (recent)"),
        ]
    )
    msgs.extend(conv[-2:])
    budget = _budget(1000)
    result = compact_messages(msgs, budget)
    assert estimate_request_tokens(result, (), budget) <= budget.max_tokens
    contents = [str(m.content) for m in result]
    assert "verification note 3 (recent)" in contents  # most recent feedback survives
    assert "verification note 1 (old)" not in contents  # stale feedback dropped entirely
    assert "verification note 2" not in contents
    assert result[0].content == "You are the test agent."
    assert any(m.content == "verify the work" for m in result if isinstance(m, HumanMessage))


def test_compact_messages_fit_or_raise_when_uncompactable():
    # A huge *latest* HumanMessage cannot be stubbed, trimmed, or dropped at
    # any level — unlike huge tool results (summarized to a compact form), it
    # is preserved verbatim everywhere, so fit-or-raise must raise.
    msgs: list[BaseMessage] = [
        SystemMessage(content="You are the test agent."),
        HumanMessage(content="Inspect."),
        HumanMessage(content="x" * 50000),
    ]
    before = [(type(m).__name__, str(m.content)[:40]) for m in msgs]
    with pytest.raises(PayloadTooLargeError):
        compact_messages(msgs, _budget(800))
    after = [(type(m).__name__, str(m.content)[:40]) for m in msgs]
    assert before == after


def test_compact_messages_tools_can_tip_budget():
    msgs = _conversation(n_rounds=1)
    budget = _budget(4000)
    assert compact_messages(msgs, budget) is msgs  # fits without tools
    huge = _FakeTool(name="huge_tool", description="d" * 30000)
    with pytest.raises(PayloadTooLargeError):
        compact_messages(msgs, budget, tools=[huge])  # tools counted -> cannot fit


# --------------------------------------------------------------------------
# HTTP 413 detection
# --------------------------------------------------------------------------

class _HttpError(Exception):
    def __init__(self, status=None, message=""):
        super().__init__(message)
        if status is not None:
            self.status_code = status


def test_is_payload_too_large_detects_status_413():
    assert is_payload_too_large(_HttpError(status=413))
    assert not is_payload_too_large(_HttpError(status=429))
    assert not is_payload_too_large(_HttpError(status=500))


def test_is_payload_too_large_detects_message_variants():
    assert is_payload_too_large(RuntimeError("Request body too large (413)"))
    assert is_payload_too_large(RuntimeError("HTTPError 413 Payload Too Large"))
    assert is_payload_too_large(RuntimeError("content too large"))
    assert not is_payload_too_large(RuntimeError("Connection reset by peer"))


def test_payload_too_large_error_is_clean():
    err = PayloadTooLargeError("boom")
    assert isinstance(err, Exception)
    assert str(err) == "boom"
