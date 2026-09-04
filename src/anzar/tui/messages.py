"""Textual messages used to move agent events from the worker thread to the UI."""

from __future__ import annotations

from textual.message import Message

from anzar.tui.events import AgentEvent


class AgentEventMessage(Message):
    """Posted from the agent worker thread; handled on the UI thread."""

    def __init__(self, event: AgentEvent):
        super().__init__()
        self.event = event
