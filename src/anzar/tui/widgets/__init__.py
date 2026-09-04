"""Widgets used by the Anzar TUI."""

from anzar.tui.widgets.chat import AssistantBubble, ChatView, ToolEntry, UserBubble
from anzar.tui.widgets.footer import AnzarFooter
from anzar.tui.widgets.header import AnzarHeader
from anzar.tui.widgets.inputbar import InputBar
from anzar.tui.widgets.sidebar import FileTree, Sidebar

__all__ = [
    "AnzarFooter",
    "AnzarHeader",
    "AssistantBubble",
    "ChatView",
    "FileTree",
    "InputBar",
    "Sidebar",
    "ToolEntry",
    "UserBubble",
]
