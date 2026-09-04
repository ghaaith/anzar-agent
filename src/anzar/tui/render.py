"""Rendering helpers shared by TUI widgets."""

from __future__ import annotations

import re
from pathlib import Path

# File extension -> pygments/markdown language name (approximate).
_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".r": "r",
    ".jl": "julia",
    ".lua": "lua",
    ".vue": "vue",
    ".svelte": "svelte",
    ".dockerfile": "docker",
    ".txt": "text",
}

_DIFF_RE = re.compile(r"^\+\+\+ |^--- ", re.MULTILINE)


def language_from_path(path: str) -> str:
    """Best-effort language name for a file path."""
    suffix = Path(path).suffix.lower()
    name = Path(path).name.lower()
    if name in ("dockerfile",):
        return "docker"
    if name.endswith(".d.ts"):
        return "typescript"
    return _LANG_BY_EXT.get(suffix, "text")


def is_diff_output(text: str) -> bool:
    """True if the text looks like a unified diff."""
    return bool(_DIFF_RE.search(text)) and "diff" in text or bool(re.search(r"^[-+][^-+]", text, re.MULTILINE))


def truncate(text: str, limit: int = 160) -> str:
    """Truncate long tool output for the timeline."""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def short_path(path: str, width: int = 30) -> str:
    """Shorten a long absolute path for the header/sidebar."""
    path = path.replace("\\", "/")
    if len(path) <= width:
        return path
    head, _, tail = path.rpartition("/")
    if not head:
        return path[-width:]
    return "…" + "/" + tail if len(tail) + 2 <= width else "…" + path[-(width - 1):]


def tool_verb(name: str) -> str:
    """Human label for a tool name."""
    return {
        "run_command": "Ran command",
        "read_file": "Read file",
        "write_file": "Wrote file",
        "list_files": "Listed files",
        "search_code": "Searched code",
        "get_file_info": "Got file info",
    }.get(name, name.replace("_", " ").title())


def tool_args_summary(name: str, args: dict) -> str:
    """Short, human summary of a tool call's most relevant argument.

    Returns "" when there is nothing worth showing (e.g. a bare ``list_files``
    at the workspace root), so callers can omit the ``: <detail>`` suffix.
    """
    if not args:
        return ""
    if name in ("read_file", "write_file", "get_file_info"):
        return str(args.get("path", ""))
    if name == "list_files":
        path = args.get("path", ".")
        return "" if path in (".", "") else str(path)
    if name == "run_command":
        return str(args.get("command", ""))
    if name == "search_code":
        return str(args.get("query", ""))
    first = next(iter(args.values()), "")
    return str(first)
