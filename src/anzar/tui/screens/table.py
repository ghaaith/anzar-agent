"""Generic data table modal used by /list and /search.

Dismisses with the selected row index (``int``), or ``None`` on cancel.
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Static


class TableScreen(ModalScreen[int]):
    BINDINGS = [("escape", "cancel")]

    def __init__(
        self,
        title: str,
        headers: list[str],
        rows: list[list[str]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._title = title
        self._headers = headers
        self._rows = rows

    def compose(self) -> ComposeResult:
        with Vertical(classes="model-body"):
            yield Static(f"[bold]{self._title}[/bold]")
            yield DataTable(id="table", cursor_type="row")
            yield Button("Close", id="close")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(*self._headers)
        table.add_rows(self._rows)
        table.focus()

    def on_data_table_row_selected(self, event) -> None:
        row = event.cursor_row
        if row is not None and row < len(self._rows):
            self.dismiss(row)

    def on_button_pressed(self, event) -> None:
        if event.button.id == "close":
            self.action_cancel()

    def action_cancel(self) -> None:
        self.dismiss(None)
