"""Chat transcript: user bubbles, assistant markdown, and a tool timeline."""

from __future__ import annotations

import itertools
import time

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Markdown, Static

from anzar.tui.render import tool_args_summary, tool_verb, truncate
from anzar.tui.theme import ACCENT, ERROR, MUTED, SUCCESS, WARNING

_SPINNER = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_STREAM_REFRESH = 0.08


class UserBubble(Widget):
    """A single user message."""

    DEFAULT_CLASSES = "user"

    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self._text = text

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {ACCENT}]❯ You[/bold {ACCENT}]", classes="user-label")
        yield Static(self._text, markup=False)


class AssistantBubble(Widget):
    """Streams raw text while generating, then renders as Markdown."""

    DEFAULT_CLASSES = "assistant"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.buffer = ""
        self._stream: Static | None = None
        self._last_render = 0.0

    def compose(self) -> ComposeResult:
        yield Static(f"[{MUTED}]● Anzar[/{MUTED}]", classes="assistant-label")
        self._stream = Static("", markup=False)
        yield self._stream

    def push(self, text: str) -> None:
        self.buffer += text
        if self._stream is None:
            return
        now = time.monotonic()
        if now - self._last_render >= _STREAM_REFRESH:
            self._last_render = now
            self._stream.update(self.buffer)

    def finalize(self) -> None:
        if self._stream is not None:
            self._stream.remove()
            self._stream = None
        md = Markdown(self.buffer or "_no response_")
        self.mount(md)


class ToolEntry(Widget):
    """A timeline row for one tool call, with a spinner until it finishes."""

    DEFAULT_CLASSES = "tool-line"

    def __init__(self, call_id: str, name: str, args: dict, **kwargs):
        super().__init__(**kwargs)
        self.call_id = call_id
        self._name = name
        self.args = args
        self._summary = tool_args_summary(name, args)
        self._status = "running"
        self._output = ""
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.1, self._tick)

    def _tick(self) -> None:
        if self._status == "running":
            self.refresh()

    def _label(self) -> str:
        name = tool_verb(self._name)
        detail = f": {self._summary}" if self._summary else ""
        if self._status == "running":
            frame = next(_SPINNER)
            return f"[{ACCENT}]{frame} {name}[/{ACCENT}][{MUTED}]{detail}...[/{MUTED}]"
        mark, color = {
            "ok": ("✓", SUCCESS),
            "warn": ("⚠", WARNING),
            "err": ("✗", ERROR),
        }[self._status]
        extra = f" — {truncate(self._output, 90)}" if self._output else ""
        return f"[{color}]{mark} {name}[/{color}][{MUTED}]{detail}{extra}[/{MUTED}]"

    def render(self) -> Text:
        return Text.from_markup(self._label())

    def finish(self, ok: bool, output: str) -> None:
        self._status = "ok" if ok else "err"
        self._output = output.strip()
        if self._timer is not None:
            self._timer.stop()
        self.refresh()


class ChatView(VerticalScroll):
    """Scrollable transcript owned by the app."""

    def __init__(self, **kwargs):
        super().__init__(id="chat", **kwargs)
        self._active_assistant: AssistantBubble | None = None
        self._tool_entries: dict[str, ToolEntry] = {}
        self._splash: Static | None = None

    def add_splash(self, markup: str) -> None:
        """Show a startup banner (logo etc.) at the top of the transcript."""
        self.remove_splash()
        self._splash = Static(markup, classes="splash")
        self.mount(self._splash, before=0)
        self.scroll_home(animate=False)

    def remove_splash(self) -> None:
        if self._splash is not None:
            self._splash.remove()
            self._splash = None

    def add_user(self, text: str) -> None:
        self.mount(UserBubble(text))
        self.scroll_end(animate=False)

    def clear_all(self) -> None:
        self._active_assistant = None
        self._tool_entries.clear()
        self._splash = None
        self.remove_children()

    def begin_assistant(self) -> None:
        bubble = AssistantBubble()
        self.mount(bubble)
        self._active_assistant = bubble
        self.scroll_end(animate=False)

    def push_text(self, text: str) -> None:
        if self._active_assistant is None:
            self.begin_assistant()
        self._active_assistant.push(text)
        self.scroll_end(animate=False)

    def finalize_assistant(self) -> None:
        if self._active_assistant is not None:
            self._active_assistant.finalize()
            self._active_assistant = None
        self.scroll_end(animate=False)

    def cancel_assistant(self) -> None:
        if self._active_assistant is not None:
            bubble = self._active_assistant
            self._active_assistant = None
            if not bubble.buffer.strip():
                bubble.remove()

    def add_error(self, message: str) -> None:
        self.cancel_assistant()
        self.mount(Static(f"[{ERROR}]✗ Error:[/{ERROR}] {message}", markup=False))
        self.scroll_end(animate=False)

    def add_tool_started(self, call_id: str, name: str, args: dict) -> None:
        entry = ToolEntry(call_id, name, args)
        self._tool_entries[call_id] = entry
        self.mount(entry)
        self.scroll_end(animate=False)

    def finish_tool(self, call_id: str, name: str, ok: bool, output: str) -> None:
        entry = self._tool_entries.pop(call_id, None)
        if entry is None:
            return
        entry.finish(ok, output)
        self.scroll_end(animate=False)
