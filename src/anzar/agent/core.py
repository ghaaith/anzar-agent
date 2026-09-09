"""Agent core using LangGraph for the tool-calling loop.

The agent runs one of two compiled graphs:

* **Tool graph** — the full loop for tasks that need the workspace:
  ``START -> agent -> route -> human | tools -> agent -> END`` with a
  recovery edge back into ``agent`` when a tool fails, and a ``human``
  node that surfaces choices to the user via LangGraph ``interrupt``
  (human-in-the-loop).
* **Fast graph** — a single unbound LLM call for plain chat: no tool
  schemas, no plan/task file injection. This is the main latency win.

Both graphs share ``AnzarState`` (a superset of ``MessagesState``) and
collect per-turn metrics that are exposed on ``agent.last_metrics``.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from sqlalchemy.orm import Session

from anzar.agent.context import (
    ContextBudget,
    PayloadTooLargeError,
    build_request_estimate,
    compact_messages,
    is_payload_too_large,
    next_deeper_view,
)
from anzar.agent.errors import (
    ErrorCategory,
    backoff_seconds,
    classify_error,
    intervention_message,
    is_retryable,
)
from anzar.agent.llm import LLMProvider, friendly_error_message
from anzar.agent.memory import ConversationMemory
from anzar.agent.prompts import (
    CHAT_SYSTEM_PROMPT,
    CHOICE_CANCELLED_MESSAGE,
    FAST_PATH_SYSTEM_PROMPT,
    LOOP_LIMIT_MESSAGE,
)
from anzar.agent.tools import create_tools, make_ask_user_tool
from anzar.agent.verifier import (
    MAX_VERIFICATION_ATTEMPTS,
    VerificationStatus,
    WorkspaceVerifier,
    snapshot_changed,
    snapshot_workspace,
    verification_failed_summary,
    verification_message,
)
from anzar.checkpoint import (
    compute_diff_from_result,
    create_checkpoint,
    record_task,
)
from anzar.db.models import Settings, Usage, Workspace

logger = logging.getLogger("anzar.agent")

# Relative paths (from the workspace root) of the persistent planning files the
# agent reads every turn and maintains as it completes multi-step work:
#   PLAN.md  - the user's product plan (read-only context; agent never edits it)
#   STATE.md - the session pointer (current task, phase, last completed)
#   tasks/   - one folder per dev task with task_plan.md / findings.md / progress.md
PLAN_FILE = "PLAN.md"
STATE_FILE = "STATE.md"
TASKS_DIR = "tasks"
TASK_PLAN_NAME = "task_plan.md"

# Cap on how much of each planning file is injected into the prompt per turn.
PLAN_INJECTION_CAP = 4000

# Shown when a turn completes with no text and no error (free-tier models
# sometimes return an empty completion instead of raising).
EMPTY_RESPONSE_MESSAGE = (
    "The model returned an empty response. Try again, or switch to a faster model with /model."
)

# LangGraph 1.2.x crashes on ``Command(resume=None)`` (an unbound ``resume_is_map``
# in the pregel loop), so "user cancelled" is signalled with a sentinel string
# instead of ``None``. The human node treats it the same as no answer.
CANCEL_SENTINEL = "<anzar-user-cancelled>"

# Bounded so the agent can never spin forever on a single turn. 50 rounds is
# enough for multi-file dev tasks while remaining interruptible via the
# repeated-tool guard, the interrupt budget, and the user pressing Esc.
DEFAULT_MAX_STEPS = 50
DEFAULT_MAX_REPEATS = 3
DEFAULT_MAX_INTERRUPTS = 3
DEFAULT_MAX_RETRIES = 2


class AnzarState(MessagesState):
    """Graph state: MessagesState plus loop/observability bookkeeping."""

    steps: int
    tool_attempts: dict[str, int]
    metrics: dict[str, Any]
    fast_path: bool
    done: bool
    mutated: bool
    cp_id: Optional[str] = None
    checkpoint_result: Optional[Any] = None


@dataclass
class AgentMetrics:
    """Per-turn observability snapshot, set on ``agent.last_metrics``."""

    path: str = "tools"
    llm_calls: int = 0
    tool_calls: int = 0
    interrupts: int = 0
    retries: int = 0
    steps: int = 0
    tokens_est: int = 0
    elapsed_ms: int = 0
    ended_by: Optional[str] = None
    errors: int = 0
    verifications: int = 0


# Words that strongly hint the message needs the workspace/tools.
_TOOLY_RE = re.compile(
    r"\b(run|read|write|list|search|edit|fix|install|build|test|create|delete|move|find|"
    r"grep|update|refactor|debug|compile|lint|format|git|pip|npm|open|check|verify|"
    r"analy[sz]e|error|bug|file|code|workspace|explain|explanation|explained|explaining|"
    r"describe|description|described|describing|summarize|summary|summarised|summarizing|"
    r"overview|contents|listing|listed|listing|detail|details|detailed|"
    r"project|repo|repository|directory|structure|task|continue|resume|implement|improve|"
    r"full|all|every|each|whole|entire)\b",
    re.IGNORECASE,
)
_FILE_EXT_RE = re.compile(
    r"\b\w+\.(py|js|ts|tsx|jsx|css|html|md|json|toml|yaml|yml|txt|env|lock)\b", re.IGNORECASE
)


def _tool_call_slices(msg: BaseMessage) -> list[dict]:
    """Return tool-call dicts from a messages-mode stream chunk.

    Newer langgraph (1.2.x) can emit a full ``AIMessage`` in messages mode,
    which carries ``tool_calls`` but not the chunk-only ``tool_call_chunks``
    attribute; ``AIMessageChunk`` exposes the reverse. Accept both.
    """
    if isinstance(msg, AIMessageChunk):
        chunks = msg.tool_call_chunks
        if chunks:
            return list(chunks)
        return msg.tool_calls or []
    return list(getattr(msg, "tool_calls", None) or [])


class AnzarAgent:
    """Main agent that processes messages using LLM + tools in a loop.

    The agent uses LangGraph to manage the tool-calling loop:
    1. LLM receives the message and decides whether to use tools
    2. If tools are called, they execute and results feed back to the LLM
    3. This repeats until the LLM produces a final response
    """

    def __init__(
        self,
        llm: Any,
        tools: list[BaseTool],
        system_prompt: str,
        memory: ConversationMemory,
        workspace_path: str,
        user_id: uuid.UUID | None = None,
        db: Session | None = None,
        *,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_repeats: int = DEFAULT_MAX_REPEATS,
        max_interrupts: int = DEFAULT_MAX_INTERRUPTS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        fast_path: bool = True,
        verifier: WorkspaceVerifier | None = None,
        context_budget: ContextBudget | None = None,
    ):
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.memory = memory
        self.workspace_path = workspace_path
        self.user_id = user_id
        self.db = db

        self.max_steps = max_steps
        self.max_repeats = max_repeats
        self.max_interrupts = max_interrupts
        self.max_retries = max_retries
        self._fast_enabled = fast_path

        self.context_budget = context_budget or ContextBudget.from_env()

        self.verifier = verifier if verifier is not None else WorkspaceVerifier(workspace_path)

        self.last_metrics: AgentMetrics | None = None
        self.last_path: str | None = None
        self.last_request_estimate: Any | None = None

        # Build the agent graphs
        self.graph, self.fast_graph = self._build_graph()
        self._compiled_tool_graph = self.graph

    # ------------------------------------------------------------------
    # Fast-path detection
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_simple(text: str) -> bool:
        """Conservative heuristic: is this plain chat, not a tool task?

        Biased toward False (use the tool graph) whenever in doubt so the
        agent never loses its tools on a borderline request.
        """
        t = text.strip()
        if not t or len(t) > 300:
            return False
        if "```" in t or "`" in t:
            return False
        if any(c in t for c in "{};=<>|"):
            return False
        if "\\" in t:
            return False
        if _TOOLY_RE.search(t):
            return False
        if _FILE_EXT_RE.search(t):
            return False
        return True

    def _use_fast_path(self, user_message: str) -> bool:
        """Only route to the fast graph when it is intact and the message is simple.

        If ``graph`` was replaced externally (e.g. tests stub it), fall back to
        the tool graph so the replacement is honored.  Also stays on the tool
        graph when the recent conversation already contains tool results — the
        user is likely mid-task and a tool-free reply would be unhelpful.
        """
        if not (
            self._fast_enabled
            and self.fast_graph is not None
            and self.graph is self._compiled_tool_graph
            and self._looks_simple(user_message)
        ):
            return False
        # If the conversation already has tool results, the user is mid-task
        # — stay on the tool graph even for "simple" follow-up messages.
        recent = self.memory.load_messages(limit=6)
        if any(isinstance(m, ToolMessage) for m in recent):
            return False
        return True

    # ------------------------------------------------------------------
    # Credentials gate
    # ------------------------------------------------------------------

    def _credentials_error(self) -> str | None:
        """Return an intervention message if a cloud provider lacks a key."""
        name = type(self.llm).__name__
        if "Ollama" in name:
            return None
        keys = [
            getattr(self.llm, "api_key", None),
            getattr(self.llm, "anthropic_api_key", None),
            getattr(self.llm, "google_api_key", None),
            # langchain-openai's ChatOpenAI (also used for groq via base_url)
            # stores the key on ``openai_api_key``.
            getattr(self.llm, "openai_api_key", None),
        ]
        if any(keys):
            return None
        is_cloud = any(k in name for k in ("OpenAI", "Anthropic", "Google", "GenAI"))
        if not is_cloud:
            return None
        return intervention_message(
            what="the configured LLM provider requires an API key before the agent can call it.",
            why="no API key is set for this provider.",
            where="run /apikey in the app, or add the provider key to the project's .env file.",
            example_env="GROQ_API_KEY=sk-...  (or OPENAI_API_KEY / ANTHROPIC_API_KEY / GOOGLE_API_KEY)",
        )

    # ------------------------------------------------------------------
    # Plan / task context
    # ------------------------------------------------------------------

    def _read_file(self, name: str) -> str:
        """Read a file from the workspace root ('' if missing/unreadable)."""
        path = Path(self.workspace_path) / name
        try:
            if not path.is_file():
                return ""
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:PLAN_INJECTION_CAP]
        except OSError:
            return ""

    def _load_plan(self) -> str:
        """Read the user's product plan from the workspace ('' if none)."""
        return self._read_file(PLAN_FILE)

    def _load_state(self) -> str:
        """Read the session pointer ('' if none)."""
        return self._read_file(STATE_FILE)

    def _active_task_dir(self) -> Path | None:
        """Resolve the current task folder ('' if none).

        Preference order: the slug named in STATE.md, then the newest
        tasks/<NNN>-<slug>/ folder that contains a task_plan.md.
        """
        root = Path(self.workspace_path)
        tasks_root = root / TASKS_DIR
        try:
            if tasks_root.is_dir():
                pass
            else:
                return None
        except OSError:
            return None

        # 1) Pointer from STATE.md
        for line in self._load_state().splitlines():
            line = line.strip().lower()
            if line.startswith("current task:"):
                slug = line.split(":", 1)[1].strip()
                candidate = tasks_root / slug
                if candidate.is_dir() and (candidate / TASK_PLAN_NAME).is_file():
                    return candidate
                break

        # 2) Newest task folder containing a task_plan.md
        try:
            candidates = [
                p
                for p in tasks_root.iterdir()
                if p.is_dir() and (p / TASK_PLAN_NAME).is_file()
            ]
        except OSError:
            return None
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def _load_task(self) -> str:
        """Read the active task plan ('' if none)."""
        task_dir = self._active_task_dir()
        if task_dir is None:
            return ""
        path = task_dir / TASK_PLAN_NAME
        try:
            if not path.is_file():
                return ""
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:PLAN_INJECTION_CAP]
        except OSError:
            return ""

    def _build_system_message(self) -> SystemMessage:
        """System message including the user's plan, active task, and pointer."""
        content = self.system_prompt
        plan = self._load_plan()
        if plan:
            content += f"\n\n## User's app plan\n{plan}"
        task = self._load_task()
        if task:
            content += f"\n\n## Current task plan\n{task}"
        state = self._load_state()
        if state:
            content += f"\n\n## Session pointer\n{state}"
        return SystemMessage(content=content)

    # ------------------------------------------------------------------
    # LLM invocation with retry
    # ------------------------------------------------------------------

    def _invoke_llm_retry(
        self,
        llm: Any,
        messages: list[BaseMessage],
        *,
        tools: Any = (),
        budget: ContextBudget | None = None,
    ) -> tuple[Any, int, int]:
        """Invoke the LLM with complete-request budgeting and bounded retry.

        The complete request is measured — conversation (``messages``) plus the
        tool definitions bound to ``llm`` (``tools``) plus reserved overhead and
        output space. ``compact_messages`` guarantees the view fits the safe
        budget or raises; this method never knowingly sends an oversized body.

        The budgeted (possibly compacted) view is what reaches the provider —
        the graph's message state is never mutated, so streaming/metrics and the
        checkpointer see the original history.

        Transient failures retry with backoff (up to ``max_retries``). HTTP 413
        payload-too-large failures compact the *original* conversation one
        compaction level deeper on each retry (bounded by ``budget.max_retries``
        and re-measuring the complete request before every send), so every
        retry is genuinely stronger than the view that just failed and the
        identical request is never resent. When the conversation still cannot
        fit, a clean :class:`PayloadTooLargeError` is raised instead of
        exposing raw provider internals.

        Returns ``(response, llm_calls, retries)``.
        """
        budget = budget or self.context_budget
        retries = budget.max_retries
        request_messages = compact_messages(messages, budget, tools=tools)
        attempts = 0
        while True:
            estimate = build_request_estimate(request_messages, tools, budget)
            self._debug_context_estimate("send", estimate, budget)
            if not estimate.fits:
                raise PayloadTooLargeError(
                    "The conversation cannot be made to fit within the configured "
                    f"context budget ({budget.max_tokens} tokens) even after compaction."
                )
            try:
                response = llm.invoke(request_messages)
                self.last_request_estimate = estimate
                return response, 1, attempts
            except Exception as e:
                if is_payload_too_large(e):
                    if attempts >= retries:
                        raise PayloadTooLargeError(
                            "The conversation could not be made to fit within the "
                            f"model's request limit after {retries} compaction retries."
                        ) from e
                    attempts += 1
                    # Escalate one compaction *level* deeper than the view the
                    # provider rejected. Compaction levels are strictly
                    # decreasing, so every retry is genuinely stronger than the
                    # request that just failed and can never resend the identical
                    # body. (A size-fraction target — e.g. half of max_tokens —
                    # is too coarse: it can land between levels and then hit the
                    # minimum-budget floor, making the second retry a no-op.)
                    tighter = next_deeper_view(messages, budget, estimate.total, tools=tools)
                    if tighter is None:
                        logger.warning(
                            "HTTP 413 compaction made no progress; giving up: %s", e
                        )
                        raise PayloadTooLargeError(
                            "The conversation could not be made to fit within the "
                            "model's request limit, even after automatic compaction."
                        ) from e
                    tighter_estimate = build_request_estimate(tighter, tools, budget)
                    self._debug_context_estimate(f"emergency{attempts}", tighter_estimate, budget)
                    request_messages = tighter
                    logger.warning(
                        "HTTP 413 detected; compacted context (attempt %d) and retrying",
                        attempts,
                    )
                    continue
                if is_retryable(e) and attempts < self.max_retries:
                    attempts += 1
                    delay = backoff_seconds(attempts - 1)
                    logger.warning("LLM retry %d after %s: %s", attempts, type(e).__name__, e)
                    time.sleep(delay)
                    continue
                raise

    def _debug_context_estimate(
        self,
        stage: str,
        estimate: Any,
        budget: ContextBudget,
    ) -> None:
        """Log a request-size breakdown when ``ANZAR_CONTEXT_DEBUG`` is on.

        Only counts are logged — never prompts, message content, or secrets.
        """
        if not budget.debug:
            return
        logger.info(
            "[context:%s] %s | messages=%d tools=%d overhead=%d output_reserve=%d "
            "total=%d safe=%d fits=%s",
            stage,
            budget.describe,
            estimate.messages_tokens,
            estimate.tool_tokens,
            estimate.overhead_tokens,
            estimate.output_reserve_tokens,
            estimate.total,
            estimate.max_tokens,
            estimate.fits,
        )

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self, compile_graphs: bool = True) -> tuple[Any, Any]:
        """Build the tool graph and the fast graph.

        Args:
            compile_graphs: When True (default), return compiled graphs with
                checkpointer/recursion-limit applied. When False, return the raw
                ``StateGraph`` builders (used by LangGraph Studio so the API
                server can attach its own managed checkpointer).
        """
        # Bind tools to the LLM so it knows what's available.
        # tool_choice="auto" is required — some providers (Groq, OpenRouter)
        # return the old <function=...> format without it, causing tool_use_failed.
        # parallel_tool_calls lets the model batch independent calls (e.g.
        # list_files + read_file) in a single round-trip for fewer LLM calls.
        bound_tools = [*self.tools, make_ask_user_tool()]
        llm_with_tools = self.llm.bind_tools(
            bound_tools, tool_choice="auto", parallel_tool_calls=True
        )
        tool_node = ToolNode(bound_tools)
        tool_names = {t.name for t in bound_tools}

        def call_llm(state: AnzarState) -> dict:
            """Call the LLM with the current message history (loop-protected)."""
            messages = state["messages"]
            steps = state.get("steps", 0)
            metrics = dict(state.get("metrics") or {})

            if steps >= self.max_steps:
                metrics["ended_by"] = "step_limit"
                return {
                    "messages": [AIMessage(content=LOOP_LIMIT_MESSAGE)],
                    "metrics": metrics,
                }

            try:
                response, calls, retries = self._invoke_llm_retry(
                    llm_with_tools, messages, tools=bound_tools
                )
            except Exception as e:
                if classify_error(e) == ErrorCategory.MALFORMED_TOOL:
                    logger.warning("Tool call failed, retrying with correction: %s", e)
                    corrected = messages + [
                        SystemMessage(
                            content=(
                                "Your previous tool call was rejected because it was "
                                "malformed. Retry using a proper JSON tool call — do "
                                "not wrap arguments in <function> tags."
                            )
                        )
                    ]
                    try:
                        response, calls, retries = self._invoke_llm_retry(
                            llm_with_tools, corrected, tools=bound_tools
                        )
                    except Exception:
                        # Tool-less fallback: let the model answer without tools.
                        response, calls, retries = self._invoke_llm_retry(
                            self.llm, messages, tools=()
                        )
                else:
                    metrics["errors"] = metrics.get("errors", 0) + 1
                    raise

            metrics["llm_calls"] = metrics.get("llm_calls", 0) + calls
            metrics["retries"] = metrics.get("retries", 0) + retries
            metrics["steps"] = steps + 1

            updates: dict[str, Any] = {
                "messages": [response],
                "steps": steps + 1,
                "metrics": metrics,
            }

            if isinstance(response, AIMessage) and response.tool_calls:
                attempts = dict(state.get("tool_attempts") or {})

                # Build id→key map across full history for error lookup.
                tc_map_all: dict[str, str] = {}
                for msg in state.get("messages") or []:
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tid = tc.get("id", "")
                            if tid:
                                tc_map_all[tid] = (
                                    f"{tc.get('name', '')}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                                )

                blocked_key: str | None = None
                for tc in response.tool_calls:
                    key = (
                        f"{tc.get('name', '')}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                    )
                    attempts[key] = attempts.get(key, 0) + 1
                    if attempts[key] > self.max_repeats and blocked_key is None:
                        blocked_key = key

                if blocked_key is not None:
                    # Find the most recent error text for the blocked call.
                    last_error = ""
                    for msg in reversed(state.get("messages") or []):
                        if isinstance(msg, ToolMessage):
                            tid = getattr(msg, "tool_call_id", None) or ""
                            if tc_map_all.get(tid) == blocked_key:
                                last_error = str(msg.content or "")[:300]
                                break

                    if attempts[blocked_key] > self.max_repeats * 2:
                        # Hard cap: END with an informative message.
                        tool_part = blocked_key.split(":", 1)
                        tool_name = tool_part[0] if tool_part else "unknown"
                        tool_args = tool_part[1] if len(tool_part) > 1 else ""
                        alt_list = sorted(tool_names - {tool_name})
                        alt_str = ", ".join(alt_list[:8]) if alt_list else "(none)"
                        err_line = (
                            f"\nLast error: {last_error}" if last_error else ""
                        )
                        metrics["ended_by"] = "repeated_tool"
                        metrics["repeat_block_key"] = blocked_key
                        metrics["repeat_block_error"] = last_error
                        updates["messages"] = [
                            AIMessage(
                                content=(
                                    f"Reached the hard repeat limit for "
                                    f"`{tool_name}({tool_args})` after "
                                    f"{self.max_repeats * 2 + 1} identical "
                                    f"calls.{err_line}\n\n"
                                    f"Available tools: {alt_str}\n\n"
                                    f"Try a different tool, or answer the "
                                    f"user directly with what you know."
                                )
                            )
                        ]
                        updates["metrics"] = metrics
                        return updates

                    # Soft cap: route to recover (drop the tool-call attempt).
                    metrics["repeat_block"] = True
                    metrics["repeat_block_key"] = blocked_key
                    metrics["repeat_block_error"] = last_error
                    return {
                        "metrics": metrics,
                        "tool_attempts": attempts,
                        "steps": steps + 1,
                    }

                updates["tool_attempts"] = attempts
            return updates

        def route(state: AnzarState) -> str:
            """Route to tools, the human-in-the-loop node, verification, or END."""
            # Repeat-blocked: route to recover for corrective feedback.
            metrics = state.get("metrics") or {}
            if metrics.get("repeat_block"):
                return "recover"
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                if any(tc.get("name") == "ask_user" for tc in last_message.tool_calls):
                    return "human"
                return "tools"
            # Mandatory verification: the graph must not END on a failed
            # verification until it passes or the attempt cap is exhausted.
            if (
                metrics.get("verification_pending_failed")
                and metrics.get("verification_attempts", 0) < MAX_VERIFICATION_ATTEMPTS
            ):
                return "verify"
            return END

        def human_node(state: AnzarState) -> dict:
            """Pause for user input when the model calls ``ask_user``.

            Runs on the main graph thread (never inside ToolNode), so the
            LangGraph ``interrupt`` mechanism is always safe to use.

            A cancelled choice (or exhausting the interrupt budget) ends the
            graph here — no confirmation, no further tool work.
            """
            last = state["messages"][-1]
            tc = None
            if isinstance(last, AIMessage) and last.tool_calls:
                tc = next((t for t in last.tool_calls if t.get("name") == "ask_user"), None)
            if tc is None:
                return {"messages": [SystemMessage(content="No pending choice.")]}

            metrics = dict(state.get("metrics") or {})
            count = metrics.get("interrupts", 0)
            if count >= self.max_interrupts:
                metrics["interrupts"] = count + 1
                metrics["ended_by"] = "interrupt_limit"
                return {
                    "messages": [AIMessage(content=CHOICE_CANCELLED_MESSAGE)],
                    "metrics": metrics,
                    "done": True,
                }

            payload = {
                "type": "choice",
                "question": str(tc["args"].get("question", "")),
                "options": [str(o) for o in (tc["args"].get("options") or [])],
            }
            choice = interrupt(payload)
            metrics["interrupts"] = count + 1
            cancelled = choice in (None, "", CANCEL_SENTINEL)
            if cancelled:
                metrics["ended_by"] = "cancelled"
                return {
                    "messages": [AIMessage(content=CHOICE_CANCELLED_MESSAGE)],
                    "metrics": metrics,
                    "done": True,
                }
            return {
                "messages": [
                    ToolMessage(
                        content=f"User selected: {choice}",
                        tool_call_id=tc["id"],
                    )
                ],
                "metrics": metrics,
            }

        def route_human(state: AnzarState) -> str:
            """After a choice, route to the agent (answered) or END (cancelled)."""
            return END if state.get("done") else "agent"

        def tools_node(state: AnzarState) -> dict:
            """Execute tool calls, counting them for metrics.

            Some models occasionally hallucinate a tool name (e.g. emit
            ``repo_browser.print_tree``). Newer langgraph converts that into
            an error ToolMessage; older versions raise instead. Convert the
            rejection into error ToolMessages here so the recovery loop can
            correct the model either way.

            Mutation detection is infrastructure-level, never trusted to the
            model: ``write_file`` always counts, and a ``run_command`` round
            counts when the workspace snapshot changed (cache dirs excluded).
            """
            last = state["messages"][-1]
            calls = list(last.tool_calls or []) if isinstance(last, AIMessage) else []
            names = [tc.get("name") for tc in calls]
            has_write = "write_file" in names
            has_run = "run_command" in names

            # Create a checkpoint before the first mutation of a turn so the
            # user can diff/rollback the agent's changes later. Best-effort:
            # a failed checkpoint must NOT fail the round, but the attempt is
            # made before tools run so the save gets the pre-mutation state.
            cp_id = state.get("cp_id")
            if has_write or has_run:
                if not cp_id:
                    try:
                        cp_result = create_checkpoint(
                            self.workspace_path,
                            db=self.db,
                            description="agent turn",
                        )
                        cp_id = str(cp_result.id)
                        state["checkpoint_result"] = cp_result
                        self.last_checkpoint = cp_result
                    except Exception:  # noqa: BLE001
                        cp_id = None

            snapshot_before = (
                snapshot_workspace(self.workspace_path) if has_run else None
            )
            try:
                out = tool_node.invoke(state)
            except Exception:
                last = state["messages"][-1]
                bad = []
                if isinstance(last, AIMessage) and last.tool_calls:
                    bad = [
                        tc
                        for tc in last.tool_calls
                        if tc.get("name") not in tool_names
                    ]
                if not bad:
                    raise
                return {
                    "messages": [
                        ToolMessage(
                            content=(
                                "Error: tool '%s' is not available. "
                                "Available tools: %s"
                            )
                            % (tc.get("name"), ", ".join(sorted(tool_names))),
                            tool_call_id=tc.get("id") or f"unknown-{i}",
                            status="error",
                        )
                        for i, tc in enumerate(bad)
                    ],
                    "mutated": False,
                }
            metrics = dict(state.get("metrics") or {})
            msgs = out.get("messages") or []

            # Convert any invalid tool calls (malformed JSON args) into error
            # ToolMessages so the recovery loop can correct the model.
            invalid_tcs = (
                getattr(last, "invalid_tool_calls", [])
                if isinstance(last, AIMessage)
                else []
            )
            if invalid_tcs:
                extra_msgs = []
                for i, itc in enumerate(invalid_tcs):
                    extra_msgs.append(
                        ToolMessage(
                            content=(
                                "Error: tool '%s' received invalid arguments"
                                " — %s. Fix the JSON and retry."
                            )
                            % (
                                itc.get("name", "?"),
                                itc.get("error", "bad JSON"),
                            ),
                            tool_call_id=itc.get("id") or f"invalid-{i}",
                            status="error",
                        )
                    )
                msgs = list(msgs) + extra_msgs
                out["messages"] = msgs

            metrics["tool_calls"] = metrics.get("tool_calls", 0) + sum(
                1 for m in msgs if isinstance(m, ToolMessage)
            )

            # Reset consecutive-failure counter for calls that succeeded.
            tool_attempts = dict(state.get("tool_attempts") or {})
            tc_by_id = {tc.get("id"): tc for tc in calls if tc.get("id")}
            for msg in msgs:
                if isinstance(msg, ToolMessage) and not (
                    msg.content or ""
                ).lstrip().startswith("Error"):
                    tc = tc_by_id.get(msg.tool_call_id)
                    if tc:
                        key = f"{tc.get('name', '')}:{json.dumps(tc.get('args', {}), sort_keys=True)}"
                        tool_attempts[key] = 0

            mutated = has_write
            if has_run:
                mutated = mutated or snapshot_changed(
                    snapshot_before, snapshot_workspace(self.workspace_path)
                )
            out["metrics"] = metrics
            out["mutated"] = mutated
            out["tool_attempts"] = tool_attempts
            if cp_id:
                out["cp_id"] = cp_id
            return out

        def should_recover(state: AnzarState) -> str:
            """After tools run, route to recovery, verification, or the agent.

            Recovery wins on an errored result (a failed write is not a
            mutation). Otherwise a mutated round goes to the verifier; pure
            read-only rounds go straight back to the agent.
            """
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage):
                    content = msg.content or ""
                    if isinstance(content, str) and content.startswith("Error"):
                        return "recover"
                    break
            if state.get("mutated"):
                return "verify"
            return "agent"

        def recover(state: AnzarState) -> dict:
            """Bounded corrective hint after a failed or repeated tool call."""
            metrics = dict(state.get("metrics") or {})
            metrics["recoveries"] = metrics.get("recoveries", 0) + 1

            # Repeat-block: specific corrective hint with error + alternatives.
            block_key = metrics.pop("repeat_block_key", None)
            block_error = metrics.pop("repeat_block_error", "")
            metrics.pop("repeat_block", None)

            if block_key:
                parts = block_key.split(":", 1)
                tool_name = parts[0] if parts else "unknown"
                alternatives = sorted(tool_names - {tool_name})
                alt_str = ", ".join(alternatives[:8]) if alternatives else "(none)"
                hint = (
                    f"You have called `{tool_name}` with the same arguments "
                    f"{self.max_repeats + 1} times without making progress.\n"
                    f"Last result: {block_error[:300] or '(no error)'}.\n"
                    f"Do not call `{tool_name}` with these exact same arguments "
                    f"again.\n"
                    f"Available tools: {alt_str}.\n"
                    f"If you need more information, pick a different tool or "
                    f"answer the user directly."
                )
            else:
                hint = (
                    "A previous tool call failed. Inspect the error message and "
                    "try a corrected approach. Do not repeat the exact same call."
                )

            return {
                "messages": [SystemMessage(content=hint)],
                "metrics": metrics,
            }

        def verify_node(state: AnzarState) -> dict:
            """Run deterministic post-modification verification (no LLM calls).

            Baseline: the first run of a turn records the current failure count;
            later runs are only blocked (and the 3-attempt cap engaged) when the
            failure count rises above that baseline, so pre-existing breakage is
            reported but not blamed on the agent.
            """
            result = self.verifier.verify()
            metrics = dict(state.get("metrics") or {})

            baseline = metrics.get("verification_baseline")
            if baseline is None:
                baseline = result.tests_failed
                metrics["verification_baseline"] = baseline
                result = replace(result, baseline=True)

            new_failures = result.tests_failed - (baseline or 0)
            first_run = metrics.get("verification_attempts", 0) == 0
            blocked = (
                result.status is VerificationStatus.FAILED
                and (new_failures > 0 or first_run)
            )

            if blocked:
                attempts = metrics.get("verification_attempts", 0) + 1
                metrics["verification_attempts"] = attempts
            elif result.status is VerificationStatus.PASSED:
                metrics["verification_attempts"] = 0

            metrics["verifications"] = metrics.get("verifications", 0) + 1
            metrics["verification_pending_failed"] = bool(blocked)

            if blocked and metrics["verification_attempts"] >= MAX_VERIFICATION_ATTEMPTS:
                metrics["ended_by"] = "verification_failed"
                return {
                    "messages": [AIMessage(content=verification_failed_summary(result))],
                    "metrics": metrics,
                }

            message = verification_message(
                result,
                attempt=metrics.get("verification_attempts", 0) + 1,
                baseline_failures=baseline,
            )
            return {
                "messages": [SystemMessage(content=message)],
                "metrics": metrics,
            }

        def after_verify(state: AnzarState) -> str:
            """After verification: END once the attempt cap is exhausted."""
            metrics = state.get("metrics") or {}
            if (
                metrics.get("verification_pending_failed")
                and metrics.get("verification_attempts", 0) >= MAX_VERIFICATION_ATTEMPTS
            ):
                return END
            return "agent"

        def call_fast(state: AnzarState) -> dict:
            """Single unbound LLM call — no tools, no plan injection."""
            metrics = dict(state.get("metrics") or {})
            response, calls, retries = self._invoke_llm_retry(
                self.llm, state["messages"], tools=()
            )
            metrics["llm_calls"] = metrics.get("llm_calls", 0) + calls
            metrics["retries"] = metrics.get("retries", 0) + retries
            metrics["steps"] = 1
            metrics["fast_path"] = True
            return {"messages": [response], "steps": 1, "metrics": metrics}

        # ---- Tool graph -------------------------------------------------
        graph = StateGraph(AnzarState)
        graph.add_node("agent", call_llm)
        graph.add_node("tools", tools_node)
        graph.add_node("human", human_node)
        graph.add_node("recover", recover)
        graph.add_node("verify", verify_node)
        graph.add_edge(START, "agent")
        graph.add_conditional_edges("agent", route, {"tools": "tools", "human": "human", "verify": "verify", "recover": "recover", END: END})
        graph.add_conditional_edges("human", route_human, {END: END, "agent": "agent"})
        graph.add_conditional_edges("tools", should_recover, {"recover": "recover", "verify": "verify", "agent": "agent"})
        graph.add_edge("recover", "agent")
        graph.add_conditional_edges("verify", after_verify, {"agent": "agent", END: END})

        # ---- Fast graph -------------------------------------------------
        fast = StateGraph(AnzarState)
        fast.add_node("fast", call_fast)
        fast.add_edge(START, "fast")
        fast.add_edge("fast", END)

        if not compile_graphs:
            return graph, fast

        compiled = graph.compile(checkpointer=InMemorySaver())
        fast_compiled = fast.compile()

        return compiled, fast_compiled

    # ------------------------------------------------------------------
    # Helpers shared by the entry points
    # ------------------------------------------------------------------

    def _fast_messages(self, user_message: str) -> list[BaseMessage]:
        return [
            SystemMessage(content=FAST_PATH_SYSTEM_PROMPT),
            *self.memory.load_messages(limit=10),
            HumanMessage(content=user_message),
        ]

    def _new_thread_config(self) -> dict:
        thread_id = f"{getattr(self.memory, 'conversation_id', 'cli')}:{uuid.uuid4().hex}"
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": max(self.max_steps * 4, 16),
        }

    @staticmethod
    def _unpack_interrupt(data: Any) -> Any:
        """Normalize the ``__interrupt__`` payload to the interrupt value."""
        if isinstance(data, (tuple, list)) and data:
            data = data[0]
        return getattr(data, "value", data)

    def _resolve_choice(self, payload: Any, on_choice: Any) -> Any:
        """Resolve a pending choice via the callback, or cancel by default.

        Cancelling (via the sentinel) makes ``ask_user`` report 'user
        cancelled' and the agent stops working for this request — no
        confirmation, no work. The sentinel (not ``None``) is used because
        LangGraph 1.2.x crashes on ``Command(resume=None)``.
        """
        if callable(on_choice):
            choice = on_choice(payload)
            if choice not in (None, ""):
                return choice
        return CANCEL_SENTINEL

    @staticmethod
    def _extract_final(result: dict) -> str:
        messages = result.get("messages") or []
        final = messages[-1] if messages else None
        if isinstance(final, AIMessage):
            return final.content or ""
        if final is not None:
            return str(final.content) if hasattr(final, "content") else str(final)
        return ""

    def _finalize_metrics(
        self,
        metrics: dict,
        start: float,
        path: str,
        response: str,
        user_message: str,
    ) -> None:
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        tokens_est = (len(user_message) + len(response)) // 4
        if self.last_request_estimate is not None:
            tokens_est = self.last_request_estimate.total
        self.last_metrics = AgentMetrics(
            path=path,
            llm_calls=metrics.get("llm_calls", 0),
            tool_calls=metrics.get("tool_calls", 0),
            interrupts=metrics.get("interrupts", 0),
            retries=metrics.get("retries", 0),
            steps=metrics.get("steps", 0),
            tokens_est=metrics.get("tokens_est", tokens_est),
            elapsed_ms=elapsed_ms,
            ended_by=metrics.get("ended_by"),
            errors=metrics.get("errors", 0),
            verifications=metrics.get("verifications", 0),
        )
        self.last_path = path
        m = self.last_metrics
        logger.info(
            "Anzar turn: path=%s elapsed_ms=%d llm_calls=%d tool_calls=%d interrupts=%d steps=%d",
            path,
            elapsed_ms,
            m.llm_calls,
            m.tool_calls,
            m.interrupts,
            m.steps,
        )
        block_key = metrics.get("repeat_block_key")
        if block_key:
            block_error = str(metrics.get("repeat_block_error", ""))[:200]
            logger.warning(
                "Anzar repeated-tool block: key=%s error=%s", block_key, block_error
            )

    def _save_assistant(self, response: str) -> None:
        self.memory.add_assistant_message(
            response,
            metadata={"model": getattr(self.llm, "model_name", "unknown")},
        )

    def _record_task_from_turn(self, metrics: dict[str, Any]) -> None:
        """Record a task row after a mutated turn (best-effort, DB-gated).

        Uses :func:`anzar.checkpoint.create_checkpoint` if no checkpoint was
        created during the turn (e.g. the earlier attempt failed) so the
        checkpoint and task stay consistent. When no database is configured,
        the task is only stashed on ``self.last_checkpoint`` for tests.
        """
        if not metrics.get("mutated"):
            return
        ended_by = metrics.get("ended_by")
        if ended_by and ended_by in ("error", "user_cancelled", "verification_failed"):
            status = "failed"
        else:
            status = "completed"

        if self.db is None and not getattr(self, "last_checkpoint", None):
            # No DB and nothing was saved earlier — nothing to persist.
            return

        try:
            cp = self.last_checkpoint if hasattr(self, "last_checkpoint") else None
            cp_id = None
            if cp is None:
                cp = create_checkpoint(self.workspace_path, db=self.db)
                self.last_checkpoint = cp
            cp_id = cp.id if cp is not None else None

            if self.db is not None:
                diff = compute_diff_from_result(cp, self.workspace_path) if cp else None
                record_task(
                    db=self.db,
                    workspace_path=self.workspace_path,
                    description=metrics.get("task_description"),
                    status=status,
                    checkpoint_id=cp_id,
                    files_created=diff.added if diff else None,
                    files_modified=diff.modified if diff else None,
                    files_deleted=diff.deleted if diff else None,
                    verification_status=(
                        "passed"
                        if metrics.get("verifications") and not metrics.get("verification_pending_failed")
                        else None
                    ),
                )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to record task/checkpoint", exc_info=True)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self, user_message: str, on_choice: Any = None) -> str:
        """Run the agent on a user message and return the final response.

        This is the synchronous entry point. For streaming, use `stream()`.
        ``on_choice`` is an optional callable(payload) -> chosen option for
        human-in-the-loop; if omitted, a pending choice cancels the request.
        """
        start = time.perf_counter()
        credentials = self._credentials_error()
        if credentials:
            self._finalize_metrics({"errors": 1}, start, "error", "", user_message)
            return credentials

        # Save to memory
        self.memory.add_user_message(user_message)

        if self._use_fast_path(user_message):
            messages = self._fast_messages(user_message)
            result = self.fast_graph.invoke({"messages": messages, "metrics": {}})
            response = self._extract_final(result)
            metrics = dict(result.get("metrics") or {})
        else:
            system = self._build_system_message()
            history = self.memory.load_messages()
            human = HumanMessage(content=user_message)
            messages = [system] + history + [human]
            config = self._new_thread_config()

            if self.graph is self._compiled_tool_graph:
                # Real compiled graph: needs the checkpointer config for the
                # interrupt/resume loop. Replaced graphs (test stubs) are
                # invoked plain so they don't need a ``config`` argument.
                result = self.graph.invoke(
                    {"messages": messages, "metrics": {}}, config=config
                )
                while result.get("__interrupt__"):
                    payload = self._unpack_interrupt(result["__interrupt__"])
                    choice = self._resolve_choice(payload, on_choice)
                    result = self.graph.invoke(Command(resume=choice), config=config)
            else:
                result = self.graph.invoke({"messages": messages, "metrics": {}})

            response = self._extract_final(result)
            metrics = dict(result.get("metrics") or {})

        if not response.strip():
            response = f"[{EMPTY_RESPONSE_MESSAGE}]"

        self._save_assistant(response)
        self._track_usage(user_message, response)
        self._record_task_from_turn(metrics)
        self.memory.commit()
        path = "fast" if metrics.get("fast_path") else "tools"
        self._finalize_metrics(metrics, start, path, response, user_message)
        return response

    def stream(self, user_message: str, on_choice: Any = None):
        """Stream the agent's response token by token (synchronous).

        Uses LangGraph's ``messages`` stream mode so tokens arrive as they
        are generated. Also yields tool execution status messages so the
        user can see what the agent is doing.
        """
        start = time.perf_counter()
        credentials = self._credentials_error()
        if credentials:
            self._finalize_metrics({"errors": 1}, start, "error", "", user_message)
            yield credentials
            return

        self.memory.add_user_message(user_message)

        if self._use_fast_path(user_message):
            yield from self._stream_fast(user_message, start)
            return

        system = self._build_system_message()
        history = self.memory.load_messages()
        human = HumanMessage(content=user_message)
        messages = [system] + history + [human]
        config = self._new_thread_config()

        final_response = ""
        announced: set[str] = set()
        metrics: dict[str, Any] = {}
        pending: Any = {"messages": messages, "metrics": {}}

        while True:
            interrupt_payload: Any = None
            for mode, chunk in self.graph.stream(
                pending, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        interrupt_payload = self._unpack_interrupt(chunk["__interrupt__"])
                        break
                    for node, update in chunk.items():
                        msgs = (
                            update.get("messages")
                            if isinstance(update, dict)
                            else [update]
                        )
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        for msg in msgs:
                            if node == "agent" and isinstance(msg, AIMessage):
                                metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
                                metrics["steps"] = metrics.get("steps", 0) + 1
                            elif node == "tools" and isinstance(msg, ToolMessage):
                                metrics["tool_calls"] = metrics.get("tool_calls", 0) + 1
                    continue
                msg, _meta = chunk
                if not isinstance(msg, (AIMessageChunk, AIMessage)):
                    continue
                for tc in _tool_call_slices(msg):
                    tc_id = tc.get("id") or ""
                    name = tc.get("name") or ""
                    if name and tc_id not in announced:
                        announced.add(tc_id)
                        status = self._tool_status(name)
                        if status:
                            yield status
                if msg.content:
                    final_response += msg.content
                    yield msg.content

            if interrupt_payload is None:
                break
            metrics["interrupts"] = metrics.get("interrupts", 0) + 1
            choice = self._resolve_choice(interrupt_payload, on_choice)
            pending = Command(resume=choice)

        if not final_response:
            yield f"\n\n_[{EMPTY_RESPONSE_MESSAGE}]_\n\n"

        if final_response:
            self._save_assistant(final_response)

        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._record_task_from_turn(metrics)
        self._finalize_metrics(metrics, start, "tools", final_response, user_message)

    def _stream_fast(self, user_message: str, start: float):
        """Stream a simple chat reply through the fast graph."""
        messages = self._fast_messages(user_message)
        final_response = ""
        try:
            for mode, chunk in self.fast_graph.stream(
                {"messages": messages, "metrics": {}}, stream_mode=["messages", "updates"]
            ):
                if mode != "messages":
                    continue
                msg, _meta = chunk
                if isinstance(msg, (AIMessageChunk, AIMessage)) and msg.content:
                    final_response += msg.content
                    yield msg.content
        except Exception as e:
            yield f"\n\n_[Error: {friendly_error_message(e)}]_\n\n"
            final_response = ""

        if not final_response:
            yield f"\n\n_[{EMPTY_RESPONSE_MESSAGE}]_\n\n"

        if final_response:
            self._save_assistant(final_response)
        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._finalize_metrics({"llm_calls": 1, "steps": 1}, start, "fast", final_response, user_message)

    def stream_events(self, user_message: str, on_choice: Any = None):
        """Stream structured events for the TUI (synchronous generator).

        Yields ``AgentEvent`` objects (TextDelta, ToolStarted, ToolFinished,
        AgentError, Done). Human-in-the-loop choices are resolved through
        ``on_choice(payload)`` (the TUI shows a popup and blocks on the user);
        when no callback is given a pending choice cancels the request.
        """
        from anzar.tui.events import AgentError, Done, TextDelta, ToolFinished, ToolStarted

        start = time.perf_counter()
        credentials = self._credentials_error()
        if credentials:
            yield AgentError(message=credentials)
            yield Done(text="")
            return

        self.memory.add_user_message(user_message)

        if self._use_fast_path(user_message):
            yield from self._stream_events_fast(user_message, start)
            return

        system = self._build_system_message()
        history = self.memory.load_messages()
        human = HumanMessage(content=user_message)
        messages = [system] + history + [human]
        config = self._new_thread_config()

        final_response = ""
        pending_calls: dict[str, str] = {}
        announced: set[str] = set()
        error_yielded = False
        metrics: dict[str, Any] = {}
        pending: Any = {"messages": messages, "metrics": {}}

        try:
            while True:
                interrupt_payload: Any = None
                for mode, chunk in self.graph.stream(
                    pending, config=config, stream_mode=["messages", "updates"]
                ):
                    if mode == "updates":
                        if "__interrupt__" in chunk:
                            interrupt_payload = self._unpack_interrupt(chunk["__interrupt__"])
                            break
                        for node, update in chunk.items():
                            msgs = (
                                update.get("messages")
                                if isinstance(update, dict)
                                else [update]
                            )
                            if not isinstance(msgs, list):
                                msgs = [msgs]
                            for msg in msgs:
                                if node == "agent":
                                    if isinstance(msg, AIMessage):
                                        metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
                                        metrics["steps"] = metrics.get("steps", 0) + 1
                                        if msg.tool_calls:
                                            for tc in msg.tool_calls:
                                                tc_id = tc.get("id") or ""
                                                if tc_id and tc_id not in announced:
                                                    announced.add(tc_id)
                                                    pending_calls[tc_id] = tc.get("name") or "tool"
                                                    yield ToolStarted(
                                                        call_id=tc_id,
                                                        name=pending_calls[tc_id],
                                                        args=tc.get("args") or {},
                                                    )
                                elif node == "tools":
                                    if isinstance(msg, ToolMessage):
                                        metrics["tool_calls"] = metrics.get("tool_calls", 0) + 1
                                        call_id = msg.tool_call_id or ""
                                        name = pending_calls.pop(call_id, getattr(msg, "name", "tool"))
                                        content = str(msg.content or "")
                                        yield ToolFinished(
                                            call_id=call_id,
                                            name=name,
                                            ok=not content.startswith("Error"),
                                            output=content,
                                        )
                                elif node == "verify":
                                    if (
                                        isinstance(msg, SystemMessage)
                                        and isinstance(msg.content, str)
                                        and msg.content.startswith("[VERIFICATION:")
                                    ):
                                        yield ToolStarted(
                                            call_id="verify",
                                            name="verify",
                                            args={},
                                        )
                                        yield ToolFinished(
                                            call_id="verify",
                                            name="verify",
                                            ok="[VERIFICATION: passed]" in msg.content
                                            or "[VERIFICATION: unavailable]" in msg.content,
                                            output=msg.content,
                                        )
                        continue
                    msg, _meta = chunk
                    if isinstance(msg, (AIMessageChunk, AIMessage)) and msg.content:
                        final_response += msg.content
                        yield TextDelta(text=msg.content)

                if interrupt_payload is None:
                    break
                metrics["interrupts"] = metrics.get("interrupts", 0) + 1
                choice = self._resolve_choice(interrupt_payload, on_choice)
                pending = Command(resume=choice)
        except Exception as e:
            logger.error("Agent stream_events error: %s", e)
            yield AgentError(message=friendly_error_message(e))
            error_yielded = True
            final_response = ""

        if not final_response and not error_yielded:
            yield AgentError(message=EMPTY_RESPONSE_MESSAGE)

        if final_response:
            self._save_assistant(final_response)

        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._finalize_metrics(metrics, start, "tools", final_response, user_message)
        yield Done(text=final_response)

    def _stream_events_fast(self, user_message: str, start: float):
        from anzar.tui.events import AgentError, Done, TextDelta

        messages = self._fast_messages(user_message)
        final_response = ""
        error_yielded = False
        try:
            for mode, chunk in self.fast_graph.stream(
                {"messages": messages, "metrics": {}}, stream_mode=["messages", "updates"]
            ):
                if mode != "messages":
                    continue
                msg, _meta = chunk
                if isinstance(msg, (AIMessageChunk, AIMessage)) and msg.content:
                    final_response += msg.content
                    yield TextDelta(text=msg.content)
        except Exception as e:
            logger.error("Fast path stream_events error: %s", e)
            yield AgentError(message=friendly_error_message(e))
            error_yielded = True
            final_response = ""

        if not final_response and not error_yielded:
            yield AgentError(message=EMPTY_RESPONSE_MESSAGE)

        if final_response:
            self._save_assistant(final_response)
        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._finalize_metrics({"llm_calls": 1, "steps": 1}, start, "fast", final_response, user_message)
        yield Done(text=final_response)

    async def astream(self, user_message: str, on_choice: Any = None) -> AsyncGenerator[str, None]:
        """Stream the agent's response token by token (async).

        Yields text chunks as they're generated. Also yields tool execution
        status messages so the user can see what the agent is doing.
        """
        start = time.perf_counter()
        credentials = self._credentials_error()
        if credentials:
            self._finalize_metrics({"errors": 1}, start, "error", "", user_message)
            yield credentials
            return

        self.memory.add_user_message(user_message)

        if self._use_fast_path(user_message):
            async for chunk in self._astream_fast(user_message, start):
                yield chunk
            return

        system = self._build_system_message()
        history = self.memory.load_messages()
        human = HumanMessage(content=user_message)
        messages = [system] + history + [human]
        config = self._new_thread_config()

        final_response = ""
        announced: set[str] = set()
        metrics: dict[str, Any] = {}
        pending: Any = {"messages": messages, "metrics": {}}

        while True:
            interrupt_payload: Any = None
            async for mode, chunk in self.graph.astream(
                pending, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "updates":
                    if "__interrupt__" in chunk:
                        interrupt_payload = self._unpack_interrupt(chunk["__interrupt__"])
                        break
                    for node, update in chunk.items():
                        msgs = (
                            update.get("messages")
                            if isinstance(update, dict)
                            else [update]
                        )
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        for msg in msgs:
                            if node == "agent" and isinstance(msg, AIMessage):
                                metrics["llm_calls"] = metrics.get("llm_calls", 0) + 1
                                metrics["steps"] = metrics.get("steps", 0) + 1
                            elif node == "tools" and isinstance(msg, ToolMessage):
                                metrics["tool_calls"] = metrics.get("tool_calls", 0) + 1
                    continue
                msg, _meta = chunk
                if not isinstance(msg, (AIMessageChunk, AIMessage)):
                    continue
                for tc in _tool_call_slices(msg):
                    tc_id = tc.get("id") or ""
                    name = tc.get("name") or ""
                    if name and tc_id not in announced:
                        announced.add(tc_id)
                        status = self._tool_status(name)
                        if status:
                            yield status
                if msg.content:
                    final_response += msg.content
                    yield msg.content

            if interrupt_payload is None:
                break
            metrics["interrupts"] = metrics.get("interrupts", 0) + 1
            choice = self._resolve_choice(interrupt_payload, on_choice)
            pending = Command(resume=choice)

        if not final_response:
            yield f"\n\n_[{EMPTY_RESPONSE_MESSAGE}]_\n\n"

        if final_response:
            self._save_assistant(final_response)

        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._finalize_metrics(metrics, start, "tools", final_response, user_message)

    async def _astream_fast(self, user_message: str, start: float):
        messages = self._fast_messages(user_message)
        final_response = ""
        try:
            async for mode, chunk in self.fast_graph.astream(
                {"messages": messages, "metrics": {}}, stream_mode=["messages", "updates"]
            ):
                if mode != "messages":
                    continue
                msg, _meta = chunk
                if isinstance(msg, (AIMessageChunk, AIMessage)) and msg.content:
                    final_response += msg.content
                    yield msg.content
        except Exception as e:
            yield f"\n\n_[Error: {friendly_error_message(e)}]_\n\n"
            final_response = ""

        if not final_response:
            yield f"\n\n_[{EMPTY_RESPONSE_MESSAGE}]_\n\n"

        if final_response:
            self._save_assistant(final_response)
        self._track_usage(user_message, final_response)
        self.memory.commit()
        self._finalize_metrics({"llm_calls": 1, "steps": 1}, start, "fast", final_response, user_message)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def mermaid(self) -> str:
        """Return the tool graph as a Mermaid diagram (for ``anzar graph``)."""
        return self.graph.get_graph().draw_mermaid()

    def mermaid_fast(self) -> str:
        """Return the fast-path graph as a Mermaid diagram."""
        return self.fast_graph.get_graph().draw_mermaid()

    def mermaid_all(self) -> str:
        """Return both graphs as a single Mermaid document."""
        return f"## Tool graph\n\n{self.mermaid()}\n\n## Fast path\n\n{self.mermaid_fast()}"

    @staticmethod
    def _tool_status(tool_name: str) -> str:
        """Return a human-readable status line for a tool call."""
        if tool_name == "run_command":
            return "\n\n> Running command...\n\n"
        elif tool_name == "read_file":
            return "\n\n> Reading file...\n\n"
        elif tool_name == "write_file":
            return "\n\n> Writing file...\n\n"
        elif tool_name == "list_files":
            return "\n\n> Listing files...\n\n"
        elif tool_name == "search_code":
            return "\n\n> Searching...\n\n"
        return ""

    def _count_tool_calls(self, messages: list) -> int:
        """Count how many tool calls were made in the message history."""
        count = 0
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                count += len(msg.tool_calls)
        return count

    def _track_usage(self, user_message: str, response: str):
        """Track token usage in the database."""
        if not self.user_id or not self.db:
            return
        try:
            tokens = (len(user_message) + len(response)) // 4
            usage = Usage(
                user_id=self.user_id,
                action="chat",
                tokens_used=tokens,
            )
            self.db.add(usage)
            # No commit here — caller batches via memory.commit()
        except Exception:
            pass


def create_agent(
    db: Session,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    workspace_id: uuid.UUID | None = None,
    provider_override: str | None = None,
    model_override: str | None = None,
    api_key_override: str | None = None,
    max_steps: int | None = None,
) -> AnzarAgent:
    """Factory function to create an AnzarAgent from database state.

    This reads the user's settings from the DB and creates a fully configured agent.

    Args:
        db: Database session
        user_id: The user's UUID
        conversation_id: The conversation UUID
        workspace_id: Optional workspace UUID (for workspace path)
        provider_override: Override the user's provider setting
        model_override: Override the user's model setting
        api_key_override: Override the user's API key
        max_steps: Max tool rounds per turn (falls back to the user's stored
            setting, then the module default)

    Returns:
        A configured AnzarAgent instance.
    """
    from anzar.db.models import User

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise ValueError(f"User not found: {user_id}")

    # Read user's LLM settings
    settings = db.query(Settings).filter(Settings.user_id == user_id).first()
    provider = provider_override or (settings.provider if settings else "groq")
    model = model_override or (settings.model if settings else None)
    api_key = api_key_override or (settings.api_key_encrypted if settings else None)
    # User's stored step limit (column may be absent on older DBs).
    if max_steps is None:
        max_steps = getattr(settings, "max_steps", None)

    # Get workspace path
    workspace = db.query(Workspace).filter(Workspace.user_id == user_id).first()
    workspace_path = workspace.disk_path if workspace else f"/workspaces/{user_id}"

    # Create LLM — fall back to server-level API key from .env if user has none
    if not api_key:
        from anzar.config import settings as server_settings
        provider_key_map = {
            "openrouter": server_settings.openrouter_api_key,
            "groq": server_settings.groq_api_key,
            "gemini": server_settings.google_api_key,
            "openai": server_settings.openai_api_key,
            "anthropic": server_settings.anthropic_api_key,
        }
        api_key = provider_key_map.get(provider)

    llm = LLMProvider.create(provider=provider, model=model, api_key=api_key)

    # Create tools bound to this workspace
    tools = create_tools(workspace_path)

    # Build system prompt
    system_prompt = CHAT_SYSTEM_PROMPT.format(
        workspace_path=workspace_path,
        plan_path=PLAN_FILE,
        state_path=STATE_FILE,
        tasks_dir=TASKS_DIR,
        task_plan_name=TASK_PLAN_NAME,
    )

    # Create memory
    memory = ConversationMemory(db, conversation_id, workspace_id)

    return AnzarAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        memory=memory,
        workspace_path=workspace_path,
        user_id=user_id,
        db=db,
        max_steps=max_steps or DEFAULT_MAX_STEPS,
        context_budget=ContextBudget.from_env(provider),
    )


def create_agent_from_config(
    provider: str = "groq",
    model: str | None = None,
    api_key: str | None = None,
    workspace_path: str = ".",
    max_steps: int | None = None,
) -> AnzarAgent:
    """Create an agent from direct config (for CLI mode, no database needed).

    Args:
        provider: LLM provider name
        model: Model name
        api_key: API key
        workspace_path: Path to the workspace
        max_steps: Max tool rounds per turn (defaults to the module default)

    Returns:
        A configured AnzarAgent instance.
    """
    # Create LLM
    llm = LLMProvider.create(provider=provider, model=model, api_key=api_key)

    # Create tools
    tools = create_tools(workspace_path)

    # Build system prompt
    system_prompt = CHAT_SYSTEM_PROMPT.format(
        workspace_path=workspace_path,
        plan_path=PLAN_FILE,
        state_path=STATE_FILE,
        tasks_dir=TASKS_DIR,
        task_plan_name=TASK_PLAN_NAME,
    )

    # Create a mock memory (no DB in CLI mode)
    memory = _CLIMemory()

    return AnzarAgent(
        llm=llm,
        tools=tools,
        system_prompt=system_prompt,
        memory=memory,
        workspace_path=workspace_path,
        max_steps=max_steps or DEFAULT_MAX_STEPS,
        context_budget=ContextBudget.from_env(provider),
    )


class _CLIMemory:
    """In-memory conversation storage for CLI mode (no database)."""

    def __init__(self):
        self.messages: list[BaseMessage] = []

    def load_messages(self, limit: int = 30) -> list[BaseMessage]:
        return list(self.messages)[-limit:]

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content=content))

    def add_assistant_message(self, content: str, metadata: dict | None = None):
        self.messages.append(AIMessage(content=content))

    def clear(self):
        self.messages.clear()

    def commit(self):
        pass  # No-op for in-memory mode
