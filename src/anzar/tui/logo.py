"""ASCII Anzar wordmark for the TUI.

Block-letter "ANZAR" logotype rendered with the theme accent color, plus a
startup splash banner shown in the chat before the first message.
"""

from __future__ import annotations

from anzar.tui.theme import ACCENT, MUTED, TEXT

_ASCII_LOGO = (
    " █████╗  ███╗   ██╗ ███████╗  █████╗  ██████╗",
    "██╔══██╗ ████╗  ██║ ╚══███╔╝ ██╔══██╗ ██╔══██╗",
    "███████║ ██╔██╗ ██║   ███╔╝  ███████║ ██████╔╝",
    "██╔══██║ ██║╚██╗██║  ███╔╝   ██╔══██║ ██╔══██╗",
    "██║  ██║ ██║ ╚████║ ███████╗ ██║  ██║ ██║  ██║",
    "╚═╝  ╚═╝ ╚═╝  ╚═══╝ ╚══════╝ ╚═╝  ╚═╝ ╚═╝  ╚═╝",
)


def logo_markup(color: str = ACCENT) -> str:
    """The ANZAR wordmark as Rich markup (one color for all rows)."""
    return "\n".join(f"[{color}]{line}[/{color}]" for line in _ASCII_LOGO)


def splash_markup(version: str, workspace: str) -> str:
    """Startup banner: logo, version, and the active workspace."""
    return (
        logo_markup()
        + "\n\n"
        + f"[bold {TEXT}]Anzar[/bold {TEXT}] [dim]v{version}[/dim]"
        + f"\n[{MUTED}]workspace: {workspace}[/{MUTED}]"
    )
