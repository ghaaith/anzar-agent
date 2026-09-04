"""In-app command reference."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Markdown

_HELP_MD = """
# Anzar — commands

| Command | Description |
| --- | --- |
| `/help` | Show this reference |
| `/quit` | Exit Anzar |
| `/clear` | Clear the current conversation |
| `/list` | List saved conversations |
| `/load <n>` | Load conversation `#n` from `/list` || `/rename <title>` | Rename the current conversation |
| `/export [file]` | Export conversation as markdown |
| `/search <q> [--current]` | Search messages (optionally this conversation) |
| `/workspace [path]` | Show or change the workspace directory |
| `/model` | Open the interactive model picker |
| `/model set <p> [m]` | Set provider and optional model |
| `/apikey <p> <key>` | Store an API key for a provider (auto-selects it); `/apikey <p> clear` to delete |
| `/list-keys` | Show providers with a usable key |
| `/paste` | Enter multi-line input |

## Input

- **Enter** — send
- **Shift+Enter / Ctrl+Enter** — new line
- **↑ / ↓** — cycle history
- **Tab** — complete `/commands`
- **Ctrl+B** — toggle sidebar
- **F5** — refresh file tree
- **Ctrl+C** — interrupt generation / quit
- **Ctrl+Q** — quit

## Tips

- Launching Anzar in the same workspace auto-resumes the last conversation; `/clear` starts fresh.
- The agent reads files before editing and runs checks after — let it work.
- Tool actions appear in the timeline with a spinner, then ✓ or ✗.
- Code blocks and diffs are syntax highlighted.
"""


class HelpScreen(ModalScreen):
    BINDINGS = [("escape", "close_help")]

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="model-body"):
            yield Markdown(_HELP_MD)

    def close_help(self) -> None:
        self.dismiss()
