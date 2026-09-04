"""Structured events streamed from the agent to the TUI.

The TUI consumes these instead of the raw string protocol used by the web
server (``AnzarAgent.stream`` / ``astream``), which remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentEvent:
    """Base class for events emitted while the agent processes a message."""


@dataclass
class TextDelta(AgentEvent):
    """A chunk of the assistant's response text."""

    text: str


@dataclass
class ToolStarted(AgentEvent):
    """A tool call was announced by the model."""

    call_id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolFinished(AgentEvent):
    """A tool finished executing with a result."""

    call_id: str
    name: str
    ok: bool
    output: str = ""


@dataclass
class Done(AgentEvent):
    """The agent finished responding."""

    text: str = ""


@dataclass
class ChoiceRequired(AgentEvent):
    """The agent is waiting on the user to answer a question (HITL).

    The TUI must show a popup with ``options`` and/or a custom answer and
    only then resume the agent. Blocking happens on the worker thread via
    the ``on_choice`` callback; this event just signals the UI to render.
    """

    question: str
    options: list[str] = field(default_factory=list)


@dataclass
class AgentError(AgentEvent):
    """The agent raised an error while processing."""

    message: str = ""
