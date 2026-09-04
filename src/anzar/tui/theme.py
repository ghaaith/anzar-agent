"""Warm "candlelit" theme for the Anzar TUI, modeled on Claude Code.

Near-black warm charcoal background, bright azure accent, and soft cream
text keep the terminal calm and understated — minimal borders, plain
message bodies, and dim tool lines instead of heavy chrome.
"""

from __future__ import annotations

# --- Color tokens ---------------------------------------------------------

BG = "#141413"
SURFACE = "#1E1D1B"
SURFACE_ALT = "#262421"
BORDER = "#3B3833"
TEXT = "#EDE8E0"
MUTED = "#8E8880"
ACCENT = "#3B82F6"
ACCENT_DIM = "#2563EB"
ACCENT_BRIGHT = "#93C5FD"
SUCCESS = "#7CAA6E"
WARNING = "#D8A13C"
ERROR = "#E0695A"
PURPLE = "#C6A8D8"
CYAN = "#8AC0B8"

CSS = f"""
AnzarTui {{
    background: {BG};
    color: {TEXT};
    scrollbar-color: {BORDER} {BG};
    scrollbar-size: 1 1;
}}

/* --- Header ------------------------------------------------------------- */

#header {{
    height: 3;
    dock: top;
    padding: 0 1;
    background: {BG};
    border-bottom: solid {BORDER};
}}

#header #logo {{
    width: auto;
    padding: 0;
    text-style: bold;
    color: {ACCENT};
}}

#header #model-chip,
#header #workspace-chip {{
    width: auto;
    color: {MUTED};
}}

#header Label.status-pill {{
    width: auto;
    padding: 0 1;
    color: {MUTED};
    text-style: bold;
}}

#header Label.status-pill.thinking {{
    color: {ACCENT_BRIGHT};
}}

#header Label.status-pill.working {{
    color: {ACCENT};
}}

#header Label.status-pill.ready {{
    color: {SUCCESS};
}}

#header Label.status-pill.error {{
    color: {ERROR};
}}

/* --- Sidebar ------------------------------------------------------------ */

#sidebar {{
    dock: left;
    width: 34;
    background: {SURFACE};
    padding: 0 1 0 1;
    overflow-y: auto;
}}

#sidebar .collapsible--title {{
    color: {ACCENT};
    text-style: bold;
    padding: 0 0 0 1;
}}

/* --- Chat --------------------------------------------------------------- */

#chat {{
    padding: 1 2 1 2;
    background: {BG};
}}

#chat .splash {{
    padding: 3 1 2 1;
    margin: 1 0 1 0;
    text-style: bold;
    content-align: center top;
}}

.user, .assistant {{
    height: auto;
    margin: 0 2 0 0;
    padding: 1 0 1 1;
    border-top: solid {BORDER};
}}

.user {{
    border-left: solid {ACCENT};
}}

.assistant {{
    border-left: solid {BORDER};
}}

.user .user-label {{
    color: {ACCENT};
    text-style: bold;
}}

.assistant .assistant-label {{
    color: {MUTED};
}}

.tool-line {{
    padding: 0 0 0 3;
    margin: 0 0 0 0;
}}

/* --- Input bar ---------------------------------------------------------- */

#inputbar {{
    dock: bottom;
    height: auto;
    min-height: 3;
    padding: 1 1 1 1;
    margin: 0 2 0 2;
    background: {SURFACE_ALT};
    border: round {BORDER};
    align-vertical: middle;
}}

#inputbar:focus-within {{
    border: round {ACCENT};
}}

#prompt {{
    width: 3;
    content-align: left middle;
    color: {ACCENT};
    text-style: bold;
}}

#prompt-area {{
    height: 3;
    min-height: 3;
    max-height: 9;
    background: transparent;
    color: {TEXT};
    border: none;
}}

/* --- Footer ------------------------------------------------------------- */

#footer {{
    dock: bottom;
    height: 1;
    color: {MUTED};
    background: {BG};
    padding: 0 1 0 1;
}}

/* --- Markdown ----------------------------------------------------------- */

Markdown {{
    background: {BG};
}}

MarkdownH1, MarkdownH2, MarkdownH3, MarkdownH4 {{
    color: {ACCENT};
    margin: 0 0 0 0;
}}

MarkdownFence, MarkdownTable {{
    margin: 0 0 1 0;
    border: round {BORDER};
    background: {SURFACE_ALT};
}}

.assistant .markdown--inline-code {{
    color: {ACCENT_BRIGHT};
    background: {SURFACE_ALT};
}}

/* --- Modal screens ------------------------------------------------------ */

Screen > Container.model-body {{
    width: 70;
    height: auto;
    max-height: 22;
    border: round {BORDER};
    background: {SURFACE};
    padding: 1 2 1 2;
    color: {TEXT};
}}

Screen OptionList {{
    background: {SURFACE};
    border: round {BORDER};
    padding: 0 1 0 1;
}}

Screen OptionList > .option-list--option--active {{
    background: {ACCENT_DIM};
    color: {TEXT};
}}

Screen DataTable {{
    background: {SURFACE};
    border: round {BORDER};
    height: auto;
    max-height: 18;
    margin: 1 0 0 0;
}}

Screen DataTable > .datatable--header {{
    color: {ACCENT};
    text-style: bold;
}}

Screen DataTable > .datatable--cursor {{
    background: {ACCENT_DIM};
    color: {TEXT};
}}

Screen Button {{
    background: {SURFACE_ALT};
    color: {TEXT};
    border: none;
    padding: 0 2;
    margin: 1 0 0 0;
}}

Screen Button:focus {{
    background: {ACCENT_DIM};
    color: {TEXT};
}}

/* --- Tree --------------------------------------------------------------- */

Tree {{
    background: {SURFACE};
    padding: 0 0 0 1;
}}

Tree .tree--label {{
    color: {TEXT};
}}

Tree .tree--guide {{
    color: {BORDER};
}}

Tree .tree--cursor {{
    background: {SURFACE_ALT};
    color: {ACCENT_BRIGHT};
}}
"""
