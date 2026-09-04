"""Central context management for the Anzar agent.

One component decides what is allowed into an LLM request. It owns:

* **Complete-request budgeting** — the budget covers *everything* the provider
  actually receives: the system message (base prompt + injected PLAN/STATE/task
  sections), the conversation messages, the tool definitions bound to the LLM,
  request serialization overhead, and a reserved share for the model's output.
  ``estimate_request_tokens()`` measures this complete request; ``ContextBudget``
  derives a *safe input budget* from the provider cap (when known) minus output
  reservation, request overhead, and a safety margin.
* **Tool-result management** — deterministic (never LLM-generated) compaction
  of old tool outputs into short, honest summaries so large results do not
  accumulate verbatim across turns.
* **Preemptive compaction** — when the estimated complete request crosses the
  compaction threshold, ``compact_messages`` returns a smaller *view* of the
  conversation that still preserves the current task, the latest user
  instruction, the base system prompt, recent tool rounds, the most recent
  system feedback, important decisions, and a deterministic "known state"
  digest parsed from actual tool results.
* **Fit-or-fail guarantee** — ``compact_messages`` never returns a request that
  exceeds the safe budget: if even the deepest compaction cannot fit, it raises
  :class:`PayloadTooLargeError` instead of sending an oversized body.
* **HTTP 413 recovery** — ``is_payload_too_large`` detects Payload Too Large
  errors so the caller can compact more aggressively and retry a bounded
  number of times before raising :class:`PayloadTooLargeError`.

Token estimation is deliberately cheap, deterministic, and provider-agnostic:
roughly ``len(text) // 3`` characters per token. ``chars / 3`` is conservative
for code-heavy, JSON-heavy, and non-ASCII content, and because HTTP 413 is a
*payload byte* limit, character count is a closer proxy than an exact
tokenizer — and it costs a single linear pass instead of a heavy dependency.
The exact token count is never relied on: the safety margin absorbs estimation
error so a normal request cannot be rejected by a small miscount.

Configuration (environment variables, all optional):

* ``ANZAR_MAX_CONTEXT_TOKENS`` — raw request cap before reserves/margin (default 12000).
* ``ANZAR_CONTEXT_OUTPUT_RESERVE_TOKENS`` — tokens reserved for the model's reply (default 1024).
* ``ANZAR_CONTEXT_OVERHEAD_TOKENS`` — tokens reserved for serialization/envelope overhead (default 512).
* ``ANZAR_CONTEXT_SAFETY_MARGIN`` — fractional safety margin (default 0.10).
* ``ANZAR_CONTEXT_COMPACTION_THRESHOLD`` — fraction of the safe budget at which preemptive compaction starts (default 0.8).
* ``ANZAR_CONTEXT_EMERGENCY_RATIO`` — fraction of the safe budget targeted by the first 413 retry (default 0.5).
* ``ANZAR_CONTEXT_MAX_RETRIES`` — bounded 413 retry count (default 2).
* ``ANZAR_CONTEXT_DEBUG`` — set to 1/true to emit request-size breakdown logs.

When a provider's request cap is unknown, no cap is invented: the configured
default is used and the budget is documented as an estimate, not a
provider-verified limit.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

# Raw request cap (tokens) before output/overhead reserves and the safety
# margin are subtracted. 12k gives ~48KB of text, which stays comfortably
# under the payload caps that trigger HTTP 413 while keeping normal
# small-conversation behavior untouched (below the threshold nothing changes).
DEFAULT_MAX_CONTEXT_TOKENS = 12000
# Tokens reserved for the model's reply inside the same request window.
DEFAULT_OUTPUT_RESERVE_TOKENS = 1024
# Tokens reserved for request serialization / provider envelope overhead.
DEFAULT_REQUEST_OVERHEAD_TOKENS = 512
# Fractional safety margin absorbed from the cap after reserves.
DEFAULT_SAFETY_MARGIN_FRAC = 0.10
# Fraction of ``max_tokens`` at which preemptive compaction kicks in.
DEFAULT_COMPACTION_THRESHOLD = 0.8
# Fraction of ``max_tokens`` targeted by the first emergency 413-recovery retry.
DEFAULT_EMERGENCY_RATIO = 0.5

# Bounded retry budget for HTTP 413 recovery (overridable via
# ANZAR_CONTEXT_MAX_RETRIES). The caller may invoke the LLM at most
# ``1 + max_retries`` times per request; after that it must fail cleanly
# instead of retrying forever.
MAX_CONTEXT_RETRIES = 2

# Never allow the effective safe budget to collapse to a degenerate value.
MIN_SAFE_BUDGET = 512

# Compaction budgets for a single summarized tool result / history message.
_TOOL_SUMMARY_CHARS = 800
_READ_FILE_SUMMARY_CHARS = 2400
_HISTORY_SUMMARY_CHARS = 600

# Deep-level cap for the injected PLAN / task / STATE sections inside the base
# system message. These are re-readable from disk, so truncating them when the
# request still cannot fit is safe.
_DEEP_SECTION_CAP = 1200

# How many of the most recent tool rounds stay fully detailed during
# preemptive compaction (older rounds become summaries).
_KEEP_DETAILED_ROUNDS = 2

# How many of the most recent mid-turn SystemMessages (verification feedback,
# recovery hints, compaction digests) stay verbatim; older ones are stubbed or
# dropped so they cannot accumulate into an uncompactable floor.
_KEEP_RECENT_SYSTEM = 2

_COMPACT_MARKER = "[COMPACTED: content truncated — re-read the file or re-run the command for full detail]"
_COMPACTION_NOTE = (
    "Context was compacted to fit the request budget. Older tool outputs were "
    "shortened; re-read files or re-run commands for full detail."
)
_SYSTEM_STUB = "[earlier system note]"

_READ_FILE_HEADER = re.compile(r"^# (.+?): lines (\d+)-(\d+) of (\d+)\b")
_LIST_FILES_HEADER = re.compile(r"^Contents of (.+):$", re.MULTILINE)

_SLICE_MARKER = "\n…[compacted]…\n"


# --------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------

def _content_text(content: object) -> str:
    """Flatten LangChain message content (str or content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                for key in ("text", "content"):
                    value = block.get(key)
                    if isinstance(value, str):
                        parts.append(value)
                    elif isinstance(value, (list, tuple)):
                        parts.append(_content_text(value))
        return "".join(parts)
    return str(content)


def estimate_tokens(text: str) -> int:
    """Cheap conservative token estimate for a text string (~3 chars/token).

    ``chars / 3`` is deliberately conservative: it absorbs JSON escaping,
    code-heavy content, and multibyte non-ASCII text that ``chars / 4`` would
    underestimate. Deterministic, O(1) per character, no tokenizer dependency.
    """
    return len(text) // 3


def estimate_message_tokens(messages: Sequence[BaseMessage]) -> int:
    """Estimate the token size of a message list (content + tool-call args)."""
    total_chars = 0
    for msg in messages:
        total_chars += len(_content_text(msg.content))
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                total_chars += len(json.dumps(tc.get("args", {}), sort_keys=True))
    return total_chars // 3


def _tool_schema_key(tool: Any) -> tuple[str, str, str]:
    """Stable ``(name, description, schema)`` key for a tool definition."""
    name = getattr(tool, "name", "") or ""
    description = getattr(tool, "description", "") or ""
    schema = "{}"
    try:
        args_schema = getattr(tool, "args_schema", None)
        if args_schema is not None and hasattr(args_schema, "model_json_schema"):
            schema = json.dumps(args_schema.model_json_schema(), sort_keys=True)
    except Exception:
        schema = "{}"
    return (name, description, schema)


@lru_cache(maxsize=128)
def _estimate_tool_tokens_keyed(key: tuple[tuple[str, str, str], ...]) -> int:
    total_chars = sum(len(n) + len(d) + len(s) for n, d, s in key)
    return total_chars // 3


def estimate_tool_tokens(tools: Sequence[Any]) -> int:
    """Estimate the token size of the tool definitions bound to the request.

    Tool schemas are static per invocation, so the estimate is cached by a
    stable ``(name, description, schema)`` key — cheap and never recomputed
    for the same tool set.
    """
    return _estimate_tool_tokens_keyed(tuple(_tool_schema_key(t) for t in tools))


@dataclass(frozen=True)
class ContextEstimate:
    """Size breakdown of a complete LLM request."""

    messages_tokens: int
    tool_tokens: int
    overhead_tokens: int
    output_reserve_tokens: int
    max_tokens: int

    @property
    def total(self) -> int:
        """Estimated complete-request size (tokens)."""
        return (
            self.messages_tokens
            + self.tool_tokens
            + self.overhead_tokens
            + self.output_reserve_tokens
        )

    @property
    def fits(self) -> bool:
        """True when the complete request stays within the safe budget."""
        return self.total <= self.max_tokens

    def to_dict(self) -> dict[str, int]:
        """Observability snapshot (no message content)."""
        return {
            "messages": self.messages_tokens,
            "tools": self.tool_tokens,
            "overhead": self.overhead_tokens,
            "output_reserve": self.output_reserve_tokens,
            "total": self.total,
            "safe": self.max_tokens,
        }


def build_request_estimate(
    messages: Sequence[BaseMessage],
    tools: Sequence[Any] = (),
    budget: "ContextBudget | None" = None,
) -> ContextEstimate:
    """Measure the complete request: messages + tools + overhead + output reserve."""
    budget = budget if budget is not None else ContextBudget()
    return ContextEstimate(
        messages_tokens=estimate_message_tokens(messages),
        tool_tokens=estimate_tool_tokens(tools),
        overhead_tokens=budget.request_overhead_tokens,
        output_reserve_tokens=budget.output_reserve_tokens,
        max_tokens=budget.max_tokens,
    )


def estimate_request_tokens(
    messages: Sequence[BaseMessage],
    tools: Sequence[Any] = (),
    budget: "ContextBudget | None" = None,
) -> int:
    """Estimated total size (tokens) of the complete request."""
    return build_request_estimate(messages, tools, budget).total


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ContextBudget:
    """Safe input budget for the *complete* LLM request.

    The effective budget (``max_tokens``) is derived from the request cap —
    ``min(provider_limit, configured_max)`` — minus output/overhead reserves
    and a safety margin:

        cap -> - output_reserve - overhead -> x (1 - safety_margin) -> max_tokens

    Attributes:
        configured_max: Raw request cap before reserves (default or env override).
        provider_limit: Known provider request cap; ``None`` when unknown
            (then ``configured_max`` alone bounds the request — no invented cap).
        output_reserve_tokens: Tokens reserved for the model's reply.
        request_overhead_tokens: Tokens reserved for serialization overhead.
        safety_margin_frac: Fraction of the cap held back as a safety margin.
        compaction_threshold: Fraction of ``max_tokens`` at which preemptive
            compaction starts (0.0-1.0).
        emergency_ratio: Fraction of ``max_tokens`` targeted by the first HTTP
            413 recovery retry.
        max_retries: Bounded number of HTTP 413 compaction retries.
        debug: When True, request-size breakdowns are logged.
    """

    configured_max: int = DEFAULT_MAX_CONTEXT_TOKENS
    provider_limit: int | None = None
    output_reserve_tokens: int = DEFAULT_OUTPUT_RESERVE_TOKENS
    request_overhead_tokens: int = DEFAULT_REQUEST_OVERHEAD_TOKENS
    safety_margin_frac: float = DEFAULT_SAFETY_MARGIN_FRAC
    compaction_threshold: float = DEFAULT_COMPACTION_THRESHOLD
    emergency_ratio: float = DEFAULT_EMERGENCY_RATIO
    max_retries: int = MAX_CONTEXT_RETRIES
    debug: bool = False

    @property
    def cap(self) -> int:
        """Effective request cap: the provider limit if known, capped by config."""
        if self.provider_limit is None:
            return self.configured_max
        return min(self.provider_limit, self.configured_max)

    @property
    def max_tokens(self) -> int:
        """Safe input budget for the complete request (tokens)."""
        after_reserves = self.cap - self.output_reserve_tokens - self.request_overhead_tokens
        return max(MIN_SAFE_BUDGET, int(after_reserves * (1.0 - self.safety_margin_frac)))

    @property
    def compaction_limit(self) -> int:
        """Complete-request size above which ``compact_messages`` must act."""
        return int(self.max_tokens * self.compaction_threshold)

    @property
    def describe(self) -> str:
        """One-line budget summary for observability logs."""
        return (
            f"configured={self.configured_max} provider_limit={self.provider_limit} "
            f"cap={self.cap} output_reserve={self.output_reserve_tokens} "
            f"overhead={self.request_overhead_tokens} margin={self.safety_margin_frac:.0%} "
            f"max_tokens={self.max_tokens} threshold={self.compaction_threshold:.0%}"
        )

    def compacted(self, ratio: float | None = None) -> "ContextBudget":
        """Return a genuinely tighter budget for emergency compaction.

        The target is an exact fraction of the current *safe* budget, so the
        emergency request really is smaller (reserves/margin are stripped to
        make ``max_tokens`` equal the target).
        """
        ratio = self.emergency_ratio if ratio is None else ratio
        target = max(MIN_SAFE_BUDGET, int(self.max_tokens * ratio))
        return replace(
            self,
            configured_max=target,
            provider_limit=None,
            output_reserve_tokens=0,
            request_overhead_tokens=0,
            safety_margin_frac=0.0,
        )

    @classmethod
    def for_provider(cls, provider: str | None = None, *, base: "ContextBudget | None" = None) -> "ContextBudget":
        """Budget capped by the provider's known request limit (if any)."""
        from anzar.agent.providers import context_limit

        budget = base if base is not None else cls()
        limit = context_limit(provider) if provider else None
        if limit is None:
            return budget
        return replace(budget, provider_limit=limit)

    @classmethod
    def from_env(cls, provider: str | None = None) -> "ContextBudget":
        """Default budget from env overrides (see module docstring)."""

        def _int(name: str, default: int, minimum: int) -> int:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return max(minimum, int(raw))
            except ValueError:
                return default

        def _float(name: str, default: float, lo: float, hi: float) -> float:
            raw = os.environ.get(name)
            if raw is None:
                return default
            try:
                return min(hi, max(lo, float(raw)))
            except ValueError:
                return default

        kwargs: dict[str, object] = {
            "configured_max": _int("ANZAR_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS, MIN_SAFE_BUDGET),
            "output_reserve_tokens": _int("ANZAR_CONTEXT_OUTPUT_RESERVE_TOKENS", DEFAULT_OUTPUT_RESERVE_TOKENS, 0),
            "request_overhead_tokens": _int("ANZAR_CONTEXT_OVERHEAD_TOKENS", DEFAULT_REQUEST_OVERHEAD_TOKENS, 0),
            "safety_margin_frac": _float("ANZAR_CONTEXT_SAFETY_MARGIN", DEFAULT_SAFETY_MARGIN_FRAC, 0.0, 0.9),
            "compaction_threshold": _float("ANZAR_CONTEXT_COMPACTION_THRESHOLD", DEFAULT_COMPACTION_THRESHOLD, 0.1, 1.0),
            "emergency_ratio": _float("ANZAR_CONTEXT_EMERGENCY_RATIO", DEFAULT_EMERGENCY_RATIO, 0.05, 1.0),
            "max_retries": _int("ANZAR_CONTEXT_MAX_RETRIES", MAX_CONTEXT_RETRIES, 0),
        }
        raw_debug = os.environ.get("ANZAR_CONTEXT_DEBUG")
        if raw_debug is not None:
            kwargs["debug"] = raw_debug.strip().lower() in ("1", "true", "yes", "on")
        return cls.for_provider(provider, base=cls(**kwargs))


# --------------------------------------------------------------------------
# Tool-result compaction
# --------------------------------------------------------------------------

def _head_tail(text: str, budget: int) -> str:
    """Keep the first and last parts of ``text`` within ``budget`` chars."""
    if len(text) <= budget:
        return text
    avail = max(0, budget - len(_SLICE_MARKER))
    half = avail // 2
    return text[:half] + _SLICE_MARKER + text[-half:]


def compact_tool_output(content: str, *, limit: int = _TOOL_SUMMARY_CHARS) -> str:
    """Deterministically compact a tool result to at most ``limit`` chars.

    Content-aware: the real ``read_file`` / ``list_files`` / ``run_command``
    output formats are recognized so the summary preserves the genuinely
    useful signal (file header, entry count, status/exit tags). Everything in
    the result comes from slicing/parsing the actual content — nothing is
    invented. Generic output falls back to a head/tail slice.
    """
    if not content or len(content) <= limit:
        return content

    m = _READ_FILE_HEADER.match(content)
    if m:
        path, start, end, total = m.groups()
        lines = content.splitlines()
        snippet = lines[1] if len(lines) > 1 else ""
        out = f"# {path}: lines {start}-{end} of {total} [compacted — re-read for full contents]"
        if snippet:
            out += "\n" + snippet
        return out[:_READ_FILE_SUMMARY_CHARS]

    m = _LIST_FILES_HEADER.match(content)
    if m:
        path = m.group(1)
        lines = content.splitlines()
        entries = [ln for ln in lines[1:] if ln.lstrip().startswith("[")]
        out = "\n".join(lines[:3])
        if len(entries) > 2:
            out += f"\n… and {len(entries) - 2} more entries ({len(entries)} total) [compacted]"
        return out[:limit]

    if content.startswith("["):
        # Tagged output (run_command / approval blocks): keep the leading
        # [STATUS]/[EXIT]/[MS]/[NOTE]/[REASON] lines intact, then head/tail
        # of the command body.
        lines = content.splitlines()
        i = 0
        while i < len(lines) and lines[i].startswith("["):
            i += 1
        tags = "\n".join(lines[:i])
        body = "\n".join(lines[i:])
        remaining = max(1, limit - len(tags) - len(_COMPACT_MARKER) - 2)
        body_c = _head_tail(body, remaining)
        out = tags + ("\n" + body_c if body_c else "") + "\n" + _COMPACT_MARKER
        return out[:limit]

    return _head_tail(content, limit)


# --------------------------------------------------------------------------
# Known-state digest (state awareness without re-reading)
# --------------------------------------------------------------------------

_MAX_DIGEST_FILES = 12
_MAX_DIGEST_COMMANDS = 4


def build_state_digest(messages: Sequence[BaseMessage]) -> str:
    """Deterministic summary of what the agent already inspected / ran.

    Built from actual tool results in ``messages`` — files read (with line
    counts), directories listed, and the last command outcomes. Never
    fabricated: entries are only produced when the corresponding result is
    present. Empty when there is nothing worth reporting.
    """
    files: list[tuple[str, int]] = []
    dirs: list[tuple[str, int]] = []
    commands: list[str] = []
    seen: set[str] = set()

    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = str(msg.content or "")
        m = _READ_FILE_HEADER.match(content)
        if m:
            path = m.group(1)
            if path not in seen:
                seen.add(path)
                files.append((path, int(m.group(4))))
            continue
        m = _LIST_FILES_HEADER.match(content)
        if m:
            path = m.group(1)
            if path not in seen:
                seen.add(path)
                entries = [
                    ln for ln in content.splitlines()[1:] if ln.lstrip().startswith("[")
                ]
                dirs.append((path, len(entries)))
            continue
        if content.startswith("[STATUS:"):
            status = exit_code = None
            for line in content.splitlines()[:8]:
                if line.startswith("[STATUS:"):
                    status = line[len("[STATUS:") : -1].strip()
                elif line.startswith("[EXIT:"):
                    exit_code = line[len("[EXIT:") : -1].strip()
            commands.append(f"run_command: status={status} exit={exit_code}")

    lines: list[str] = []
    for path, total in files[:_MAX_DIGEST_FILES]:
        lines.append(f"- read_file({path}): {total} lines")
    for path, count in dirs[:_MAX_DIGEST_FILES]:
        lines.append(f"- list_files({path}): {count} entries")
    for cmd in commands[-_MAX_DIGEST_COMMANDS:]:
        lines.append(f"- {cmd}")
    if not lines:
        return ""
    return "Known workspace state (compiled from earlier tool results; re-run tools for updated data):\n" + "\n".join(lines)


# --------------------------------------------------------------------------
# Message compaction
# --------------------------------------------------------------------------

def _summarize_tool_message(msg: ToolMessage) -> BaseMessage:
    return msg.model_copy(
        update={"content": compact_tool_output(str(msg.content or ""))}
    )


def _trim_history_message(msg: BaseMessage, limit: int = _HISTORY_SUMMARY_CHARS) -> BaseMessage:
    content = _content_text(msg.content)
    if len(content) <= limit:
        return msg
    return msg.model_copy(
        update={"content": content[:limit] + "\n…[older conversation compacted]…"}
    )


def _stub_history_message(msg: BaseMessage) -> BaseMessage:
    """Replace an old conversation message with a short role placeholder.

    Used only by the deepest compaction levels, where trimming alone cannot
    free enough budget. The latest user instruction and the compaction digest
    remain in the view, so the agent keeps its task context.
    """
    if isinstance(msg, HumanMessage):
        label = "user"
    elif isinstance(msg, AIMessage):
        label = "assistant"
    else:
        label = "message"
    return msg.model_copy(update={"content": f"[earlier {label} message]"})


_SECTION_HEADERS = ("## User's app plan", "## Current task plan", "## Session pointer")


def _shrink_system_sections(content: str, cap: int = _DEEP_SECTION_CAP) -> str:
    """Truncate the injected PLAN/task/STATE sections inside a system message.

    Only the injected sections are shortened (each to ``cap`` chars); the base
    system prompt is untouched. Safe because those sections are re-readable
    from disk. Returns the input unchanged when it is already small.
    """
    if len(content) <= cap * 4 + 256:
        return content
    result = content
    for header in _SECTION_HEADERS:
        marker = f"\n{header}\n"
        start = result.find(marker)
        if start == -1:
            marker = f"{header}\n"
            start = result.find(marker)
            if start == -1:
                continue
        body_start = start + len(marker)
        end = len(result)
        for other in _SECTION_HEADERS:
            nxt = result.find(f"\n{other}\n", body_start)
            if nxt != -1 and nxt < end:
                end = nxt
        body = result[body_start:end]
        if len(body) <= cap:
            continue
        result = (
            result[:body_start]
            + body[:cap]
            + "\n…[section truncated — re-read the source file]…"
            + result[end:]
        )
    return result


def _find_rounds(messages: Sequence[BaseMessage]) -> list[tuple[int, int]]:
    """Index ranges ``(start, end)`` of tool-call rounds (AIMessage + ToolMessages)."""
    rounds: list[tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        msg = messages[i]
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            j = i + 1
            while j < n and isinstance(messages[j], ToolMessage):
                j += 1
            rounds.append((i, j))
            i = j
        else:
            i += 1
    return rounds


# Escalating compaction levels. ``drop_beyond`` keeps only the N most recent
# rounds (older rounds are removed whole — AIMessage + ToolMessages, preserving
# valid provider pairing). ``keep_detailed`` keeps that many rounds verbatim.
# ``keep_recent_history`` is how many pre-turn conversation messages stay full;
# older history is trimmed ("trim") or replaced with placeholders ("stub").
# ``keep_recent_system`` is how many of the most recent mid-turn SystemMessages
# stay verbatim (older ones are stubbed, or dropped at the deepest level so
# verification/recovery feedback cannot accumulate into an uncompactable
# floor). ``trim_system`` shortens the injected PLAN/task/STATE sections at
# deep levels. ``digest`` appends the compaction note + known-state digest.
_LEVELS = (
    {"keep_detailed": 2, "drop_beyond": None, "keep_recent_history": 8, "history": "trim", "digest": True, "keep_recent_system": 2, "drop_old_system": False, "trim_system": False},
    {"keep_detailed": 1, "drop_beyond": 2, "keep_recent_history": 6, "history": "trim", "digest": True, "keep_recent_system": 1, "drop_old_system": False, "trim_system": False},
    {"keep_detailed": 0, "drop_beyond": 2, "keep_recent_history": 4, "history": "stub", "digest": True, "keep_recent_system": 1, "drop_old_system": False, "trim_system": True},
    {"keep_detailed": 0, "drop_beyond": 1, "keep_recent_history": 2, "history": "stub", "digest": False, "keep_recent_system": 1, "drop_old_system": True, "trim_system": True},
)


def _compact_once(messages: Sequence[BaseMessage], level: int) -> list[BaseMessage]:
    """Compress ``messages`` at one escalation ``level`` (0-based)."""
    cfg = _LEVELS[min(level, len(_LEVELS) - 1)]
    keep_detailed = cfg["keep_detailed"]
    drop_beyond = cfg["drop_beyond"]
    keep_recent_history = cfg["keep_recent_history"]
    history_mode = cfg["history"]
    include_digest = cfg["digest"]
    keep_recent_system = cfg["keep_recent_system"]
    drop_old_system = cfg["drop_old_system"]
    trim_system = cfg["trim_system"]

    n = len(messages)
    rounds = _find_rounds(messages)
    detailed: set[tuple[int, int]] = set(rounds[-keep_detailed:]) if keep_detailed else set()
    dropped: set[tuple[int, int]] = set()
    if drop_beyond is not None:
        keep = set(rounds[-drop_beyond:]) if drop_beyond > 0 else set()
        dropped = {r for r in rounds if r not in keep}

    # The latest human message is the current instruction and must survive.
    last_human = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, HumanMessage):
            last_human = i

    # The first SystemMessage is the base system prompt and must survive;
    # mid-turn SystemMessages (verification/recovery/digest) are kept only
    # when recent enough, so they cannot accumulate unboundedly.
    system_indices = [i for i, msg in enumerate(messages) if isinstance(msg, SystemMessage)]
    base_system_index = system_indices[0] if system_indices else -1
    mid_system_indices = [i for i in system_indices[1:]]
    keep_mid_system: set[int] = set()
    if keep_recent_system and mid_system_indices:
        keep_mid_system = set(mid_system_indices[-keep_recent_system:])

    out: list[BaseMessage] = []
    for i, msg in enumerate(messages):
        round_span = next((r for r in rounds if r[0] <= i < r[1]), None)
        if round_span is not None and round_span in dropped:
            continue
        if isinstance(msg, SystemMessage):
            if i == base_system_index:
                content = _content_text(msg.content)
                if trim_system:
                    content = _shrink_system_sections(content)
                if content != _content_text(msg.content):
                    out.append(msg.model_copy(update={"content": content}))
                else:
                    out.append(msg)
                continue
            if i in keep_mid_system:
                out.append(msg)
                continue
            if drop_old_system:
                continue
            out.append(msg.model_copy(update={"content": _SYSTEM_STUB}))
            continue
        if isinstance(msg, ToolMessage) and round_span is not None and round_span not in detailed:
            out.append(_summarize_tool_message(msg))
        elif (
            isinstance(msg, (HumanMessage, AIMessage))
            and i < last_human
            and round_span is None
            and i < last_human - keep_recent_history
        ):
            out.append(_stub_history_message(msg) if history_mode == "stub" else _trim_history_message(msg))
        else:
            out.append(msg)

    if include_digest:
        digest = build_state_digest(messages)
        already_has_note = any(
            isinstance(m, SystemMessage) and str(m.content).startswith(_COMPACTION_NOTE)
            for m in out
        )
        if digest and not already_has_note:
            note = SystemMessage(content=_COMPACTION_NOTE + "\n\n" + digest)
            insert_at = len(out)
            for i, m in enumerate(out):
                if isinstance(m, HumanMessage):
                    insert_at = i
                    break
            out.insert(insert_at, note)
    return out


def compact_messages(
    messages: Sequence[BaseMessage],
    budget: ContextBudget,
    tools: Sequence[Any] = (),
) -> list[BaseMessage]:
    """Return a view of ``messages`` that fits ``budget`` (never mutates input).

    The *complete* request is measured (messages + tool definitions + request
    overhead + output reserve). Below the compaction threshold this is a no-op
    and returns the *same* list, so normal requests behave exactly as before.
    Otherwise compaction escalates through the levels in :data:`_LEVELS` and
    returns the first view that fits the safe budget.

    If even the deepest level cannot fit the safe budget, :class:`PayloadTooLargeError`
    is raised — an oversized request is never returned.
    """
    if estimate_request_tokens(messages, tools, budget) <= budget.compaction_limit:
        return messages

    results = [_compact_once(messages, level) for level in range(len(_LEVELS))]
    # Prefer the least aggressive level that fits the soft (preemptive) limit…
    for result in results:
        if estimate_request_tokens(result, tools, budget) <= budget.compaction_limit:
            return result
    # …otherwise the least aggressive level that fits the hard safe budget.
    for result in results:
        if estimate_request_tokens(result, tools, budget) <= budget.max_tokens:
            return result
    raise PayloadTooLargeError(
        "The conversation is too large for the configured context budget, even "
        "after automatic compaction."
    )


def next_deeper_view(
    messages: Sequence[BaseMessage],
    budget: ContextBudget,
    current_estimate: int,
    tools: Sequence[Any] = (),
) -> list[BaseMessage] | None:
    """Least-aggressive compaction level strictly smaller than a given size.

    Used by the HTTP 413 recovery path: ``current_estimate`` is the complete
    estimate of the request the provider just rejected, and the returned view
    is the first (least aggressive) compaction level whose complete estimate is
    strictly smaller. Because the levels in :data:`_LEVELS` are strictly
    decreasing, this guarantees every retry is genuinely stronger than the view
    that failed and never resends an identical body — unlike a size-fraction
    target, which can land between levels (skipping one) and then hit the
    minimum-budget floor on the next escalation.

    Returns ``None`` when no level is strictly smaller (the deepest view was
    already rejected), so the caller can stop bounded retries cleanly.
    """
    for level in range(len(_LEVELS)):
        result = _compact_once(messages, level)
        if estimate_request_tokens(result, tools, budget) < current_estimate:
            return result
    return None


# --------------------------------------------------------------------------
# HTTP 413 detection
# --------------------------------------------------------------------------

class PayloadTooLargeError(Exception):
    """Raised when the conversation cannot be made to fit the request budget.

    Only raised after ``max_retries`` aggressive compaction retries (or when
    even the deepest compaction cannot fit the safe budget), so callers surface
    a clean, actionable error to the user instead of raw provider internals.
    """


def is_payload_too_large(exc: Exception) -> bool:
    """Detect a Payload Too Large (HTTP 413) failure from any provider."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status == 413:
        return True
    text = str(exc)
    if "413" in text:
        return True
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "payload too large",
            "request too large",
            "request body too large",
            "content too large",
            "message too long",
        )
    )
