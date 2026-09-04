"""Fixed bottom input bar with history, multiline, and command completion."""

from __future__ import annotations

from collections import deque

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Static, TextArea

from anzar.tui.theme import ACCENT

COMMANDS = [
    "/help",
    "/quit",
    "/clear",
    "/list",
    "/load",
    "/rename",
    "/save",
    "/export",
    "/search",
    "/workspace",
    "/model",
    "/model set",
    "/step_limit",
    "/apikey",
    "/list-keys",
    "/paste",
    "/refresh",
    "/graph",
]


class PromptArea(TextArea):
    """TextArea that owns Enter/Tab/Up/Down handling.

    Keys are consumed by the focused widget, so container-level bindings never
    fire while the area has focus. This widget handles them directly:

    * Enter sends the message (Shift/Ctrl+Enter inserts a newline)
    * Up/Down cycle submitted history on single-line input, and move the
      cursor when editing multiline content
    * Tab completes /commands (handled by the parent InputBar)
    """

    BINDINGS = [
        Binding("up", "history_previous", "History", show=False),
        Binding("down", "history_next", "History", show=False),
    ]

    def __init__(self, submit_callback, **kwargs):
        super().__init__(soft_wrap=True, **kwargs)
        self._submit_callback = submit_callback
        self._history: deque[str] = deque(maxlen=200)
        self._hpos = -1
        self._draft = ""

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            self.action_submit()
            event.stop()
            event.prevent_default()
            return
        if event.key in ("shift+enter", "ctrl+enter", "alt+enter"):
            self.insert("\n")
            event.stop()
            event.prevent_default()
            return
        await super()._on_key(event)

    # --- Actions ---

    def action_submit(self) -> None:
        text = self.text.strip()
        if not text:
            return
        self._history.append(text)
        self._hpos = -1
        self.text = ""
        self._submit_callback(text)

    def action_history_previous(self) -> None:
        if self._multiline():
            self.action_cursor_up()
            return
        if not self._history:
            return
        if self._hpos == -1:
            self._draft = self.text
        self._hpos = min(self._hpos + 1, len(self._history) - 1)
        self.text = self._history[-1 - self._hpos]
        self._cursor_to_end()

    def action_history_next(self) -> None:
        if self._multiline():
            self.action_cursor_down()
            return
        if self._hpos == -1:
            return
        self._hpos -= 1
        if self._hpos == -1:
            self.text = self._draft
        else:
            self.text = self._history[-1 - self._hpos]
        self._cursor_to_end()

    # --- Helpers ---

    def _multiline(self) -> bool:
        return "\n" in self.text

    def _cursor_to_end(self) -> None:
        rows = self.text.split("\n")
        self.move_cursor((len(rows) - 1, len(rows[-1])))


class InputBar(Horizontal):
    """Bottom prompt: prompt marker + PromptArea + command completion.

    Tab (handled here, bubbles up from the focused PromptArea) completes
    /commands; everything else lives on the PromptArea itself.
    """

    BINDINGS = [
        Binding("tab", "autocomplete", "Complete", show=False),
    ]

    def __init__(self, submit_callback, **kwargs):
        super().__init__(id="inputbar", **kwargs)
        self._submit_callback = submit_callback
        self._area: PromptArea | None = None

    def compose(self) -> ComposeResult:
        yield Static(f"[bold {ACCENT}]❯[/bold {ACCENT}]", id="prompt")
        self._area = PromptArea(self._submit_callback, id="prompt-area")
        yield self._area

    def on_mount(self) -> None:
        self._area.focus()

    def action_autocomplete(self) -> None:
        text = self._area.text
        line = text.split("\n")[-1]
        word = line.rsplit(" ", 1)[-1]
        if word.startswith("/"):
            matches = [c for c in COMMANDS if c.startswith(word)]
            if len(matches) == 1:
                self._area.text = text[: len(text) - len(word)] + matches[0]
                self._area._cursor_to_end()
            elif matches:
                self.app.notify("  ".join(matches), title="Commands")
            else:
                self.app.notify("No matching command", severity="warning")
        else:
            self.app.notify("Tab completes /commands", severity="information")

    def set_text(self, text: str) -> None:
        self._area.text = text
        self._area._cursor_to_end()

    def focus_input(self) -> None:
        self._area.focus()
