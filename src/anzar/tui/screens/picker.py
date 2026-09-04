"""Two-step provider → model picker modal + HITL choice popup."""

from __future__ import annotations

from typing import Callable

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static


class ModelPicker(ModalScreen):
    """Interactive model switcher mirroring the legacy /model menu.

    Dismisses with ``(provider, model_or_None)``, or ``None`` to keep current.
    """

    BINDINGS = [("escape", "cancel")]

    def __init__(
        self,
        providers: list[str],
        models_for: Callable[[str], list[str]],
        name_for: Callable[[str], str],
        has_key: Callable[[str], bool],
        current_provider: str,
        current_model: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._providers = providers
        self._models_for = models_for
        self._name_for = name_for
        self._has_key = has_key
        self._current_provider = current_provider
        self._current_model = current_model
        self._step = 0
        self._selected_provider: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="model-body"):
            yield Static("[bold]Select provider[/bold]", id="title")
            yield OptionList(id="providers")
            yield OptionList(id="models", disabled=True)

    def on_mount(self) -> None:
        self._render_providers()

    def _render_providers(self) -> None:
        ol = self.query_one("#providers", OptionList)
        ol.clear_options()
        for pid in self._providers:
            label = f"{self._name_for(pid)}  [{pid}]"
            if pid == self._current_provider:
                label += "  (current)"
            if not self._has_key(pid):
                label += "  (no API key)"
            ol.add_option(label)
        ol.add_option("── Keep current ──")
        ol.focus()

    def _render_models(self) -> None:
        pid = self._selected_provider or ""
        self.query_one("#title", Static).update(
            f"[bold]Select model for {self._name_for(pid)}[/bold]"
        )
        self.query_one("#providers", OptionList).display = False
        ol = self.query_one("#models", OptionList)
        ol.disabled = False
        ol.display = True
        ol.clear_options()
        for m in self._models_for(pid):
            label = m
            if pid == self._current_provider and m == self._current_model:
                label += "  (current)"
            ol.add_option(label)
        ol.add_option("── Use default ──")
        ol.focus()

    def on_option_list_option_selected(self, event) -> None:
        ol = event.option_list
        if self._step == 0:
            idx = event.option_index
            if idx >= len(self._providers):
                self.dismiss(None)
                return
            self._selected_provider = self._providers[idx]
            self._step = 1
            self._render_models()
        else:
            idx = event.option_index
            models = self._models_for(self._selected_provider or "")
            if idx >= len(models):
                self.dismiss((self._selected_provider, None))
                return
            self.dismiss((self._selected_provider, models[idx]))

    def action_cancel(self) -> None:
        self.dismiss(None)


class ChoiceScreen(ModalScreen[str | None]):
    """Human-in-the-loop popup: pick an option or type a custom answer.

    Dismisses with the confirmed choice string, or ``None`` on Esc (cancel —
    the agent then stops working on the request). Selection is not final
    until the Confirm button is pressed, so the user must explicitly confirm.
    """

    BINDINGS = [("escape", "cancel")]

    DEFAULT_CSS = """
    ChoiceScreen {
        align: center middle;
    }
    ChoiceScreen Vertical {
        width: 64;
        max-height: 20;
        background: $boost;
        border: solid $primary;
        padding: 2 3;
    }
    #choice-options {
        height: auto;
        max-height: 10;
        margin: 1 0 1 0;
    }
    #choice-custom {
        width: 100%;
    }
    #choice-confirm {
        width: 100%;
    }
    """

    def __init__(self, question: str, options: list[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._question = question
        self._options = list(options)
        self._chosen: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"[bold]{self._question}[/bold]", id="choice-question")
            yield OptionList(*self._options, id="choice-options")
            yield Input(
                placeholder="…or type your own answer and press Enter",
                id="choice-custom",
            )
            yield Button("Confirm", id="choice-confirm")
            yield Static("[dim]Esc to cancel (the agent stops)[/dim]", id="choice-hint", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#choice-options", OptionList).focus()

    def _update_chosen(self, value: str | None) -> None:
        self._chosen = value or None
        hint = self.query_one("#choice-hint", Static)
        if self._chosen:
            hint.update(f"[dim]Selection: {self._chosen[:50]} — press Confirm[/dim]")
        else:
            hint.update("[dim]Esc to cancel (the agent stops)[/dim]")

    def on_option_list_option_selected(self, event) -> None:
        idx = event.option_index
        if idx is not None and 0 <= idx < len(self._options):
            self._update_chosen(self._options[idx])
        self.query_one("#choice-confirm", Button).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self._update_chosen(value)
        self.query_one("#choice-confirm", Button).focus()

    def on_button_pressed(self, event) -> None:
        if event.button.id != "choice-confirm":
            return
        if self._chosen:
            self.dismiss(self._chosen)
        else:
            self.notify("Pick an option or type your own answer first.", severity="warning")
            self.query_one("#choice-options", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)


class KeyInputScreen(ModalScreen[str | None]):
    """Modal that prompts for a provider API key (masked input).

    Dismisses with the trimmed key string, or ``None`` on cancel (Esc/empty).
    """

    BINDINGS = [("escape", "cancel")]

    DEFAULT_CSS = """
    KeyInputScreen {
        align: center middle;
    }
    KeyInputScreen Vertical {
        width: 52;
        background: $boost;
        border: solid $primary;
        padding: 2 3;
    }
    #key-input {
        width: 100%;
    }
    """

    def __init__(self, provider_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._provider_name = provider_name

    def compose(self) -> ComposeResult:
        yield Static(f"[bold]Enter API key for {self._provider_name}[/bold]")
        yield Input(placeholder="sk-…", password=True, id="key-input")
        yield Static("[dim]Esc to cancel[/dim]", id="hint", classes="hint")

    def on_mount(self) -> None:
        self.query_one("#key-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if value:
            self.dismiss(value)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        self.dismiss(None)
