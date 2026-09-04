"""Top header: logo, provider/model, workspace, and a live status pill."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Label, Static

from anzar.tui.render import short_path
from anzar.tui.theme import ACCENT, MUTED


class AnzarHeader(Horizontal):
    """Compact header with Anzar branding and live state."""

    STATUSES = ("ready", "thinking", "working", "error")

    def __init__(
        self,
        version: str,
        provider: str,
        model: str,
        workspace_path: str,
        **kwargs,
    ):
        super().__init__(id="header", **kwargs)
        self._version = version
        self._provider = provider
        self._model = model
        self._workspace_path = workspace_path

    def compose(self) -> ComposeResult:
        yield Static(
            f"[{ACCENT}]●[/{ACCENT}] anzar [dim]v{self._version}[/dim]",
            id="logo",
        )
        yield Static("", id="model-chip")
        yield Static("", id="workspace-chip")
        yield Label("● Ready", id="status", classes="status-pill ready")

    def on_mount(self) -> None:
        self._refresh_state()

    def _refresh_state(self) -> None:
        self.query_one("#model-chip", Static).update(
            f"[{MUTED}]{self._provider}[/{MUTED}]"
            f" [dim]/[/dim] "
            f"[{MUTED}]{self._model}[/{MUTED}]"
        )
        self.query_one("#workspace-chip", Static).update(
            f"[dim]{short_path(self._workspace_path)}[/dim]"
        )

    def set_model(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._refresh_state()

    def set_workspace(self, path: str) -> None:
        self._workspace_path = path
        self._refresh_state()

    def set_status(self, status: str, label: str | None = None) -> None:
        pill = self.query_one("#status", Label)
        for cls in self.STATUSES:
            pill.remove_class(cls)
        if status not in self.STATUSES:
            status = "ready"
        pill.add_class(status)
        pill.update(f"● {label or status.title()}")
