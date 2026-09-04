"""AnzarTui — full-screen Textual interface for the Anzar CLI.

Layout: header (logo, model, status) / sidebar (files, tasks, status) /
chat transcript / bottom input bar / hint footer.
"""

from __future__ import annotations

import os
import queue
import sys

from textual import events
from textual.app import App
from textual.binding import Binding
from textual.geometry import Size

from anzar.db.models import Conversation, Message, Workspace
from anzar.tui.events import (
    AgentError,
    ChoiceRequired,
    Done,
    TextDelta,
    ToolFinished,
    ToolStarted,
)
from anzar.tui.logo import splash_markup
from anzar.tui.messages import AgentEventMessage
from anzar.tui.render import tool_verb
from anzar.tui.screens.help import HelpScreen
from anzar.tui.screens.picker import ChoiceScreen, ModelPicker
from anzar.tui.screens.table import TableScreen
from anzar.tui.theme import CSS
from anzar.tui.widgets.chat import ChatView
from anzar.tui.widgets.footer import AnzarFooter
from anzar.tui.widgets.header import AnzarHeader
from anzar.tui.widgets.inputbar import InputBar
from anzar.tui.widgets.sidebar import Sidebar

_HINT = "[dim]Enter send · Shift+Enter newline · ↑/↓ history · Tab complete · /help[/dim]"


def _visible_console_height() -> int | None:
    """Height of the visible console window on Windows, or None if unavailable.

    `os.get_terminal_size()` reports the console *buffer* height on Windows
    (cmd defaults to ~9001 rows), which makes Textual lay the app out below
    the visible window. The visible height comes from `srWindow` instead.
    """
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _COORD(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class _SMALL_RECT(ctypes.Structure):
            _fields_ = [
                ("Left", wintypes.SHORT),
                ("Top", wintypes.SHORT),
                ("Right", wintypes.SHORT),
                ("Bottom", wintypes.SHORT),
            ]

        class _CONSOLE_SCREEN_BUFFER_INFO(ctypes.Structure):
            _fields_ = [
                ("dwSize", _COORD),
                ("dwCursorPosition", _COORD),
                ("wAttributes", wintypes.WORD),
                ("srWindow", _SMALL_RECT),
                ("dwMaximumWindowSize", _COORD),
            ]

        STD_OUTPUT_HANDLE = -11
        handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        info = _CONSOLE_SCREEN_BUFFER_INFO()
        ok = ctypes.windll.kernel32.GetConsoleScreenBufferInfo(
            handle, ctypes.byref(info)
        )
        if not ok:
            return None
        return int(info.srWindow.Bottom - info.srWindow.Top) + 1
    except Exception:
        return None


class AnzarTui(App):
    """The Anzar terminal UI."""

    CSS = CSS
    TITLE = "anzar"
    BINDINGS = [
        Binding("ctrl+b", "toggle_sidebar", "Sidebar"),
        Binding("f5", "refresh_files", "Files"),
        Binding("ctrl+c", "interrupt_or_quit", "Stop/Quit"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        agent,
        config,
        db,
        user,
        conv,
        workspace_path: str,
        version: str,
    ):
        super().__init__()
        self.agent = agent
        self.config = config
        self.db = db
        self.user = user
        self.conv = conv
        self.workspace_path = workspace_path
        self.version = version
        self.provider = config.provider
        self.model = config.model or config.get_default_model()
        self._busy = False
        self._worker = None
        self._visible_height: int | None = None
        self._pending_choice_queue: queue.Queue | None = None

    # --- Windows size clamp ----------------------------------------------

    def _should_clamp(self) -> bool:
        """Only clamp on the real Windows console driver (not headless/web)."""
        if sys.platform != "win32" or self._driver is None:
            return False
        if self._driver.is_headless or self._driver.is_web:
            return False
        return True

    @property
    def size(self) -> Size:
        size = super().size
        if self._should_clamp():
            if self._visible_height is None:
                self._visible_height = _visible_console_height()
            if self._visible_height is not None:
                size = size.with_height(min(size.height, self._visible_height))
        return size

    async def _on_resize(self, event: events.Resize) -> None:
        if self._should_clamp():
            self._visible_height = _visible_console_height()
            if self._visible_height is not None:
                event.size = event.size.with_height(
                    min(event.size.height, self._visible_height)
                )
        await super()._on_resize(event)

    # --- Layout -----------------------------------------------------------

    def compose(self):
        yield AnzarHeader(self.version, self.provider, self.model, self.workspace_path)
        yield Sidebar(self.workspace_path)
        yield ChatView()
        yield InputBar(self._on_submit)
        yield AnzarFooter(_HINT)

    def on_mount(self) -> None:
        self._render_history()
        self._update_status()
        n = (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conv.id)
            .count()
        )
        if n == 0:
            self.query_one(ChatView).add_splash(
                splash_markup(self.version, self.workspace_path)
            )

    # --- State helpers ----------------------------------------------------

    def _update_status(self) -> None:
        from anzar.cli import _provider_has_key

        n = (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conv.id)
            .count()
        )
        lines = [
            ("Provider", self.provider),
            ("Model", self.model),
            ("API key", "✓" if _provider_has_key(self.provider, self.db, self.user) else "✗"),
            ("Messages", str(n)),
            ("Conversation", (self.conv.title or "(untitled)")[:40]),
            ("Workspace", self.workspace_path),
        ]
        self.query_one(Sidebar).set_status_lines(lines)

    def _render_history(self) -> None:
        msgs = (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conv.id)
            .order_by(Message.created_at.desc())
            .limit(40)
            .all()
        )
        msgs.reverse()
        chat = self.query_one(ChatView)
        for m in msgs:
            if m.role == "user":
                chat.add_user(m.content)
            elif m.role == "assistant":
                chat.begin_assistant()
                chat.push_text(m.content)
                chat.finalize_assistant()

    def _rebuild_agent(self) -> None:
        from anzar.cli import _build_agent

        self.agent = _build_agent(
            self.db,
            self.user,
            self.conv.id,
            self.workspace_path,
            self.config,
            self.config.provider,
            self.config.model,
        )

    def _notify(self, message: str) -> None:
        self.notify(message)

    # --- Input dispatch ---------------------------------------------------

    def _on_submit(self, text: str) -> None:
        if self._busy:
            self.notify("Anzar is working — wait for it to finish.", severity="warning")
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        self._send_message(text)

    def _handle_command(self, raw: str) -> None:
        parts = raw.split(maxsplit=1)
        name = parts[0]
        rest = parts[1].strip() if len(parts) > 1 else ""

        if name in ("/quit", "/exit"):
            self.exit()
        elif name == "/help":
            self.push_screen(HelpScreen())
        elif name == "/clear":
            self.agent.memory.clear()
            self.query_one(ChatView).clear_all()
            self.notify("Conversation cleared.")
            self._update_status()
        elif name == "/list":
            self._cmd_list()
        elif name == "/load":
            self._cmd_load(rest)
        elif name in ("/rename", "/save"):
            if not rest:
                self.notify("Usage: /rename <title>", severity="error")
                return
            self.conv.title = rest[:255]
            self.db.commit()
            self.notify(f'Renamed to "{rest}"')
            self._update_status()
        elif name == "/export":
            self._cmd_export(rest or None)
        elif name == "/search":
            self._cmd_search(rest)
        elif name == "/workspace":
            self._cmd_workspace(rest)
        elif name == "/model":
            self._cmd_model(rest)
        elif name == "/step_limit":
            self._cmd_step_limit(rest)
        elif name == "/apikey":
            self._cmd_apikey(rest)
        elif name == "/list-keys":
            self._cmd_list_keys()
        elif name == "/refresh":
            self.query_one(Sidebar).refresh_files()
        elif name == "/graph":
            self._cmd_graph()
        elif name == "/paste":
            self.notify("Multi-line: use Shift+Enter or Ctrl+Enter in the input.", severity="information")
        else:
            self.notify(f"Unknown command: {name}  (try /help)", severity="error")

    # --- Command implementations -----------------------------------------

    def _cmd_graph(self) -> None:
        """Write the agent graph (Mermaid) to a temp file and report its path."""
        import tempfile

        try:
            diagram = self.agent.mermaid_all()
        except Exception as e:
            self.notify(f"Could not render graph: {e}", severity="error")
            return
        fd, path = tempfile.mkstemp(suffix=".mmd", prefix="anzar-graph-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(diagram)
        self.notify(f"Graph written to {path}")
        self.query_one(InputBar).focus_input()

    def _cmd_list(self) -> None:
        from anzar.cli import _list_conversations

        rows = _list_conversations(self.db, self.user)
        if not rows:
            self.notify("No conversations yet.")
            return
        data = [[str(i), d, t, str(n)] for i, d, t, n in rows]
        self.push_screen(
            TableScreen("Conversations", ["#", "Date", "Title", "Msg"], data),
            self._on_list_row,
        )

    def _on_list_row(self, row_index) -> None:
        if row_index is None:
            return
        self._cmd_load(str(row_index + 1))

    def _cmd_load(self, arg: str) -> None:
        from anzar.cli import _get_conv_by_index

        if not arg:
            self._cmd_list()
            return
        try:
            idx = int(arg)
        except ValueError:
            self.notify("Usage: /load <number>", severity="error")
            return
        target = _get_conv_by_index(self.db, self.user.id, idx)
        if not target:
            self.notify(f"Conversation #{idx} not found.", severity="error")
            return
        self.conv = target
        self._rebuild_agent()
        self.query_one(ChatView).clear_all()
        self._render_history()
        self._update_status()
        self.notify(f'Loaded "{target.title}"')

    def _cmd_search(self, arg: str) -> None:
        from anzar.cli import _search_messages

        query = arg.strip()
        if not query:
            self.notify("Usage: /search <query> [--current]", severity="error")
            return
        current_only = "--current" in query
        if current_only:
            query = query.replace("--current", "").strip()
        results = _search_messages(
            self.db, self.user, query, self.conv.id if current_only else None
        )
        if not results:
            self.notify("No results found.")
            return
        self._last_search_results = results
        data = [
            [
                str(i + 1),
                title[:30],
                msg.created_at.strftime("%m/%d %H:%M"),
                msg.content[:80].replace("\n", " "),
            ]
            for i, (msg, title, _) in enumerate(results)
        ]
        self.push_screen(
            TableScreen("Search results", ["#", "Conversation", "Date", "Snippet"], data),
            self._on_search_row,
        )

    def _on_search_row(self, row_index) -> None:
        if row_index is None:
            return
        results = getattr(self, "_last_search_results", [])
        if row_index >= len(results):
            return
        _msg, _title, cid = results[row_index]
        target = (
            self.db.query(Conversation).filter(Conversation.id == cid).first()
        )
        if target:
            self.conv = target
            self._rebuild_agent()
            self.query_one(ChatView).clear_all()
            self._render_history()
            self._update_status()
            self.notify(f'Loaded "{target.title}"')

    def _cmd_export(self, filepath: str | None) -> None:
        from anzar.cli import _cmd_export

        result = _cmd_export(self.db, self.user, self.conv.id, filepath, reporter=self._notify)
        if result:
            self.notify(f"Exported to {result}")

    def _cmd_workspace(self, arg: str) -> None:
        if not arg:
            self.notify(f"Workspace: {self.workspace_path}")
            return
        abs_path = os.path.abspath(arg)
        if not os.path.isdir(abs_path):
            self.notify(f"Directory not found: {abs_path}", severity="error")
            return
        self.workspace_path = abs_path
        ws = (
            self.db.query(Workspace)
            .filter(Workspace.user_id == self.user.id)
            .first()
        )
        if ws:
            ws.disk_path = abs_path
            self.db.commit()
        self._rebuild_agent()
        self.query_one(Sidebar).set_workspace(abs_path)
        self.query_one(AnzarHeader).set_workspace(abs_path)
        self._update_status()
        self.notify(f"Workspace changed to {abs_path}")

    def _cmd_model(self, arg: str) -> None:
        from anzar.agent.providers import provider_models, provider_name, provider_names
        from anzar.cli import _provider_has_key

        parts = arg.split()
        if parts and parts[0] == "set":
            if len(parts) < 2:
                self.notify("Usage: /model set <provider> [model]", severity="error")
                return
            provider, model = parts[1], parts[2] if len(parts) > 2 else None
            self._select_provider_with_optional_key(provider, model)
            return
        self.push_screen(
            ModelPicker(
                provider_names(),
                provider_models,
                provider_name,
                lambda p: _provider_has_key(p, self.db, self.user),
                self.provider,
                self.config.model or "",
            ),
            self._on_model_picked,
        )

    def _cmd_step_limit(self, arg: str) -> None:
        """``/step_limit [n]`` — show or change the per-turn step budget."""
        current = self.config.max_steps or 50
        arg = arg.strip()
        if not arg:
            self.notify(
                f"Current step limit: {current} (default 50). Usage: /step_limit <1-500>",
                severity="information",
            )
            return
        try:
            value = int(arg)
        except ValueError:
            self.notify("Usage: /step_limit <1-500>", severity="error")
            return
        if not 1 <= value <= 500:
            self.notify("Step limit must be between 1 and 500.", severity="error")
            return
        self.config.max_steps = value
        try:
            self.config.save()
        except Exception as e:
            self.notify(f"Saved in-session but could not persist: {e}", severity="warning")
        self._rebuild_agent()
        self.notify(f"Step limit set to {value}. Agent rebuilt.")
        self._update_status()

    def _select_provider_with_optional_key(self, provider: str, model: str | None) -> None:
        """Apply a provider/model, prompting for an API key first if needed."""
        from anzar.cli import _provider_has_key

        if provider == "ollama":
            self._apply_model_choice(provider, model)
            return
        if not _provider_has_key(provider, self.db, self.user):
            self._prompt_api_key(provider, model)
        else:
            self._apply_model_choice(provider, model)

    def _on_model_picked(self, result) -> None:
        if result is None:
            return
        provider, model = result
        self._select_provider_with_optional_key(provider, model)

    def _apply_model_choice(self, provider: str, model: str | None) -> None:
        from anzar.cli import _apply_model

        toolbar = {"provider": self.provider, "model": self.model, "msgs": ""}
        new_agent, _, _ = _apply_model(
            self.db,
            self.user,
            self.conv,
            self.workspace_path,
            self.config,
            provider,
            model,
            toolbar,
            reporter=self._notify,
        )
        if new_agent is None:
            return
        self.agent = new_agent
        self.provider = self.config.provider
        self.model = self.config.model or self.config.get_default_model()
        self.query_one(AnzarHeader).set_model(self.provider, self.model)
        self._update_status()

    def _cmd_apikey(self, arg: str) -> None:
        """/apikey <provider> <key> | /apikey <provider> clear"""
        from anzar.cli import _set_api_key

        parts = arg.split()
        if not parts:
            self.notify("Usage: /apikey <provider> <key> | /apikey <provider> clear", severity="error")
            return
        provider = parts[0]
        key = parts[1] if len(parts) > 1 else None

        toolbar = {"provider": self.provider, "model": self.model, "msgs": ""}
        new_agent, provider, model = _set_api_key(
            self.db,
            self.user,
            self.conv,
            self.workspace_path,
            self.config,
            provider,
            key,
            toolbar,
            reporter=self._notify,
        )
        if new_agent is not None:
            self.agent = new_agent
            self.provider = provider
            self.model = model or self.config.get_default_model()
            self.query_one(AnzarHeader).set_model(self.provider, self.model)
            self._update_status()
            self.query_one(InputBar).focus_input()

    def _apply_set_key_result(
        self,
        new_agent,
        provider: str,
        model: str | None,
    ) -> None:
        """Bookkeeping after _set_api_key: commit agent/provider/model + UI."""
        if new_agent is not None:
            self.agent = new_agent
            self.provider = provider
            self.model = model or self.config.get_default_model()
            self.query_one(AnzarHeader).set_model(self.provider, self.model)
            self._update_status()

    def _prompt_api_key(self, provider: str, model: str | None) -> None:
        """Push the masked key-prompt modal, then store + apply the model."""
        from anzar.cli import _set_api_key
        from anzar.agent.providers import provider_name
        from anzar.tui.screens.picker import KeyInputScreen

        def _on_key(key):
            if not key:
                self.query_one(InputBar).focus_input()
                return
            toolbar = {"provider": self.provider, "model": self.model, "msgs": ""}
            new_agent, prov, resolved_model = _set_api_key(
                self.db,
                self.user,
                self.conv,
                self.workspace_path,
                self.config,
                provider,
                key,
                toolbar,
                reporter=self._notify,
                model=model,
            )
            self._apply_set_key_result(new_agent, prov, resolved_model)
            self.query_one(InputBar).focus_input()

        self.push_screen(
            KeyInputScreen(provider_name(provider)),
            _on_key,
        )

    def _cmd_list_keys(self) -> None:
        from anzar.cli import _cmd_list_keys

        _cmd_list_keys(self.db, self.user, reporter=self._notify)
        self.query_one(InputBar).focus_input()

    # --- Chat + agent worker ---------------------------------------------

    def _send_message(self, text: str) -> None:
        from anzar.cli import _auto_title

        self._busy = True
        chat = self.query_one(ChatView)
        chat.add_user(text)
        chat.begin_assistant()
        self.query_one(AnzarHeader).set_status("thinking")
        _auto_title(self.db, self.conv.id, text)
        self._update_status()
        self._worker = self.run_worker(lambda: self._run_agent(text), thread=True)

    def _run_agent(self, text: str) -> None:
        """Runs on a worker thread; posts events to the UI thread."""
        try:
            for evt in self._stream_events(text):
                self.post_message(AgentEventMessage(evt))
                if self._worker is not None and getattr(self._worker, "is_cancelled", False):
                    break
        except Exception as e:
            from anzar.agent.llm import friendly_error_message

            self.post_message(AgentEventMessage(AgentError(message=friendly_error_message(e))))

    def _stream_events(self, text: str):
        """Call ``stream_events`` with the choice handler when it supports one.

        Stubs (tests) replace ``stream_events`` with plain functions that take
        only the user message; the real agent accepts ``on_choice`` for HITL.
        """
        import inspect

        try:
            sig = inspect.signature(self.agent.stream_events)
            has_choice = "on_choice" in sig.parameters
        except (TypeError, ValueError):
            has_choice = False
        if has_choice:
            return self.agent.stream_events(text, on_choice=self._choice_handler)
        return self.agent.stream_events(text)

    def _choice_handler(self, payload) -> str | None:
        """Runs on the worker thread when the agent hits an ``ask_user`` interrupt.

        Posts a ``ChoiceRequired`` event so the UI shows the popup, then blocks
        until the popup is dismissed (or the user interrupts). Returns the
        confirmed choice string, or ``None`` to cancel the request.
        """
        q: queue.Queue = queue.Queue()
        self._pending_choice_queue = q
        self.post_message(
            AgentEventMessage(
                ChoiceRequired(
                    question=str(payload.get("question", "")),
                    options=[str(o) for o in (payload.get("options") or [])],
                )
            )
        )
        return q.get()

    def _on_choice_dismissed(self, result) -> None:
        """UI thread: the ChoiceScreen was dismissed — unblock the worker."""
        q = self._pending_choice_queue
        self._pending_choice_queue = None
        if q is not None:
            q.put(result)
        self.query_one(InputBar).focus_input()

    def on_agent_event_message(self, message: AgentEventMessage) -> None:
        evt = message.event
        chat = self.query_one(ChatView)
        sidebar = self.query_one(Sidebar)
        header = self.query_one(AnzarHeader)

        if isinstance(evt, TextDelta):
            chat.push_text(evt.text)
        elif isinstance(evt, ToolStarted):
            chat.add_tool_started(evt.call_id, evt.name, evt.args)
            sidebar.add_task(evt.call_id, evt.name)
            header.set_status("working", tool_verb(evt.name))
        elif isinstance(evt, ToolFinished):
            chat.finish_tool(evt.call_id, evt.name, evt.ok, evt.output)
            sidebar.finish_task(evt.call_id, evt.name, evt.ok)
            header.set_status("working")
        elif isinstance(evt, ChoiceRequired):
            self.push_screen(
                ChoiceScreen(evt.question, evt.options), self._on_choice_dismissed
            )
        elif isinstance(evt, AgentError):
            chat.add_error(evt.message)
            header.set_status("error", "Error")
            self._busy = False
        elif isinstance(evt, Done):
            chat.finalize_assistant()
            header.set_status("ready")
            self._busy = False
            self._worker = None
            sidebar.refresh_files()
            self._update_status()
            self.query_one(InputBar).focus_input()

    # --- Bindings ---------------------------------------------------------

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one(Sidebar)
        sidebar.display = not sidebar.display

    def action_refresh_files(self) -> None:
        self.query_one(Sidebar).refresh_files()

    def action_interrupt_or_quit(self) -> None:
        if self._busy:
            if self._worker is not None:
                self._worker.cancel()
            if self._pending_choice_queue is not None:
                # Unblock the worker stuck on the choice popup.
                q = self._pending_choice_queue
                self._pending_choice_queue = None
                q.put(None)
            self._busy = False
            self.query_one(ChatView).finalize_assistant()
            self.query_one(AnzarHeader).set_status("ready")
            self.notify("Interrupted.")
        else:
            self.exit()

    def on_tree_node_selected(self, event) -> None:
        node = event.node
        label = str(node.label.plain) if node.label else ""
        if label.endswith("/"):
            return
        rel = self._node_path(node)
        if rel:
            self.query_one(Sidebar).add_open_file(rel)

    @staticmethod
    def _node_path(node) -> str | None:
        parts: list[str] = []
        cur = node
        while cur is not None and not cur.is_root:
            label = str(cur.label.plain)
            parts.append(label.rstrip("/"))
            cur = cur.parent
        return "/".join(reversed(parts)) if parts else None

    def on_unmount(self) -> None:
        self.config.last_conversation_id = str(self.conv.id)
        self.config.last_workspace = self.workspace_path
        self.config.save()
        self.db.close()
