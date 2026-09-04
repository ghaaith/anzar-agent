"""Bottom status footer with keybinding hints."""

from __future__ import annotations

from textual.widgets import Static


class AnzarFooter(Static):
    """Single-line hint bar."""

    def __init__(self, text: str = "", **kwargs):
        super().__init__(text, id="footer", **kwargs)

    def set_hint(self, text: str) -> None:
        self.update(text)
