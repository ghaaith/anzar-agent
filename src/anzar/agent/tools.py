"""Agent tools for interacting with the workspace sandbox."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from anzar.agent.policy import CommandPolicy

MAX_TOOL_OUTPUT_CHARS = 4000


def _cap(text: str, limit: int) -> tuple[str, bool]:
    """Return ``(text[:limit], truncated)``."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def truncate_output(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """Trim long tool output so it can't bloat the LLM context."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n…[truncated, {len(text) - limit} chars omitted]"


class AskUserInput(BaseModel):
    question: str = Field(description='The question to ask the user, e.g. "Which approach do you prefer?"')
    options: list[str] = Field(
        description="2 to 6 short, mutually exclusive choices for the user to pick from (each ~5 words or fewer)."
    )


def make_ask_user_tool() -> StructuredTool:
    """Create the human-in-the-loop tool that surfaces choices to the user.

    Built separately from :func:`create_tools` so the standard 6-tool contract
    is unchanged; the graph binds this tool additionally.
    """

    def ask_user(question: str, options: list[str]) -> str:
        from langgraph.types import interrupt

        if not options:
            return "(no options provided)"
        choice = interrupt(
            {
                "type": "choice",
                "question": question,
                "options": [str(o) for o in options[:6]],
            }
        )
        if choice is None:
            return "(user cancelled — stop working on this request)"
        return f"User selected: {choice}"

    return StructuredTool(
        name="ask_user",
        description=(
            "Ask the user to choose between options when multiple valid approaches "
            "exist and their preference matters. Provide a concrete question and "
            "2-6 short options. The user's selected option is returned."
        ),
        func=ask_user,
        args_schema=AskUserInput,
    )


def create_tools(
    workspace_path: str,
    policy: CommandPolicy | None = None,
) -> list[StructuredTool]:
    """Create tool instances bound to a specific workspace path.

    Args:
        workspace_path: Absolute path to the user's workspace directory.
        policy: Optional command policy for ``run_command``. A default policy
            bound to the workspace is used when omitted.

    Returns:
        List of LangChain tools ready for use.
    """
    return [
        _make_run_command(workspace_path, policy),
        _make_read_file(workspace_path),
        _make_write_file(workspace_path),
        _make_list_files(workspace_path),
        _make_search_code(workspace_path),
        _make_get_file_info(workspace_path),
    ]


# -- Input schemas (what the LLM sees) ------------------------------------


class RunCommandInput(BaseModel):
    command: str = Field(description='The shell command to execute (e.g., "python script.py", "pip install requests")')
    approved: bool = Field(
        default=False,
        description=(
            "Set to True ONLY after the user explicitly approved the command "
            "via ask_user. Never set it on the first attempt."
        ),
    )


class ReadFileInput(BaseModel):
    path: str = Field(description='Relative path to the file (e.g., "src/main.py", "README.md")')
    start_line: int = Field(
        default=0,
        description="0-based line to start reading from. Use to page through large files: call with start_line equal to the last shown line to continue.",
    )


class WriteFileInput(BaseModel):
    path: str = Field(description='Relative path where the file should be written (e.g., "src/utils.py")')
    content: str = Field(description="The content to write to the file")


class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="Relative path to list (default: current directory)")


class SearchCodeInput(BaseModel):
    query: str = Field(description="Text or regex pattern to search for")


class GetFileInfoInput(BaseModel):
    path: str = Field(description="Relative path to the file or directory")


# -- Tool factories --------------------------------------------------------


def _format_result(
    *,
    status: str,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    ms: int = 0,
    note: str = "",
) -> str:
    """Render an execution result as compact tagged text.

    The ``[STATUS: ...]`` / ``[EXIT: ...]`` markers let the LLM reliably parse
    the outcome regardless of the command's own output. stdout/stderr are capped
    by the caller before this formatting step.
    """
    lines = [f"[STATUS: {status}]", f"[EXIT: {exit_code}]", f"[MS: {ms}]"]
    if note:
        lines.append(f"[NOTE: {note}]")
    if stdout:
        lines.append(stdout)
    if stderr:
        lines.append(f"[STDERR]\n{stderr}")
    return "\n".join(lines)


def _make_run_command(workspace_path: str, policy: CommandPolicy | None = None) -> StructuredTool:
    def run(command: str, approved: bool = False) -> str:
        from anzar.agent.policy import CommandPolicy, Verdict
        from anzar.sandbox import get_manager

        pol = policy if policy is not None else CommandPolicy(workspace_path)
        decision = pol.evaluate(command)

        if decision.verdict is Verdict.BLOCK:
            return (
                f"[STATUS: blocked] [EXIT: -1] [MS: 0]\n"
                f"[REASON: {decision.reason}] Command was refused without executing: {command}"
            )

        if decision.verdict is Verdict.APPROVE and not approved:
            return (
                f"[STATUS: approval_required] [EXIT: -1] [MS: 0]\n"
                f"[REASON: {decision.reason}] Command NOT executed. "
                f"Ask the user via ask_user whether to run it. If approved, "
                f"re-invoke run_command with the SAME command and approved=True. "
                f"Command: {command}"
            )

        manager = get_manager()
        start = time.monotonic()
        result = manager.exec_workspace_command(
            command,
            workspace_path=workspace_path,
            timeout=pol.timeout,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)

        exit_code = result.get("exit_code", -1)
        stdout = result.get("stdout", "")
        stderr = result.get("stderr", "")

        if result.get("timed_out"):
            return _format_result(
                status="timeout", exit_code=-1, ms=elapsed_ms,
                note=f"command timed out after {pol.timeout}s",
            )

        stdout_capped, stdout_trunc = _cap(stdout, pol.max_stdout_chars)
        stderr_capped, stderr_trunc = _cap(stderr, pol.max_stderr_chars)

        if exit_code != 0:
            status = "error"
        elif stderr_trunc or stdout_trunc:
            status = "success"
        else:
            status = "success"

        note_bits = []
        if stdout_trunc:
            note_bits.append("STDOUT TRUNCATED")
        if stderr_trunc:
            note_bits.append("STDERR TRUNCATED")
        if result.get("executor") == "container":
            note_bits.append("ran in container")

        return _format_result(
            status=status,
            exit_code=exit_code,
            stdout=stdout_capped,
            stderr=stderr_capped,
            ms=elapsed_ms,
            note=", ".join(note_bits) if note_bits else "",
        )

    return StructuredTool(
        name="run_command",
        description=(
            "Execute a shell command in the workspace sandbox. Commands "
            "classified as requiring approval are NOT run — they return "
            "[STATUS: approval_required]; ask the user via ask_user, then "
            "re-run with approved=True only after explicit approval."
        ),
        func=run,
        args_schema=RunCommandInput,
    )


def _make_read_file(workspace_path: str) -> StructuredTool:
    MAX_LINES = 150

    def run(path: str, start_line: int = 0) -> str:
        full_path = Path(workspace_path) / path
        if not full_path.exists():
            return f"Error: File not found: {path}"
        if not full_path.is_file():
            if full_path.is_dir():
                return f"Error: {path} is a directory — use list_files to browse it."
            return f"Error: Not a file: {path}"
        if full_path.stat().st_size > 1024 * 1024:
            return f"Error: File too large ({full_path.stat().st_size} bytes). Use run_command with 'head' to view parts."

        try:
            content = full_path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")
            total = len(lines)
            start = max(start_line, 0)
            if start >= total:
                return f"# {path}: end of file ({total} lines total). No more lines to read."
            end = min(start + MAX_LINES, total)
            numbered = [
                f"{i + 1:4d} | {line}" for i, line in enumerate(lines[start:end], start=start)
            ]
            result = "\n".join(numbered)
            header = f"# {path}: lines {start + 1}-{end} of {total}"
            if end < total:
                header += f' (truncated — call read_file(path="{path}", start_line={end}) to continue)'
            return f"{header}\n{result}"
        except Exception as e:
            return f"Error reading file: {e}"

    return StructuredTool(
        name="read_file",
        description=(
            "Read the contents of a file in the workspace. Returns up to 150 lines at a time; "
            "use start_line to page through larger files."
        ),
        func=run,
        args_schema=ReadFileInput,
    )


def _make_write_file(workspace_path: str) -> StructuredTool:
    def run(path: str, content: str) -> str:
        full_path = Path(workspace_path) / path

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    return StructuredTool(
        name="write_file",
        description="Write content to a file in the workspace. Creates parent directories if needed. Overwrites existing files.",
        func=run,
        args_schema=WriteFileInput,
    )


def _make_list_files(workspace_path: str) -> StructuredTool:
    def run(path: str = ".") -> str:
        full_path = Path(workspace_path) / path
        if not full_path.exists():
            return f"Error: Directory not found: {path}"
        if not full_path.is_dir():
            return f"Error: Not a directory: {path}"

        entries = []
        try:
            for item in sorted(full_path.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    entries.append(f"  [DIR]  {item.name}/")
                else:
                    size = item.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size / 1024:.1f}KB"
                    else:
                        size_str = f"{size / (1024 * 1024):.1f}MB"
                    entries.append(f"  [FILE] {item.name} ({size_str})")
        except PermissionError:
            return f"Error: Permission denied: {path}"

        if not entries:
            return f"Directory '{path}' is empty"

        return f"Contents of {path}:\n" + "\n".join(entries)

    return StructuredTool(
        name="list_files",
        description="List files and directories at the given path in the workspace.",
        func=run,
        args_schema=ListFilesInput,
    )


def _make_search_code(workspace_path: str) -> StructuredTool:
    def run(query: str) -> str:
        results = []
        max_results = 50
        max_line_chars = 200

        try:
            workspace = Path(workspace_path)
            for root, dirs, files in os.walk(workspace):
                dirs[:] = [
                    d for d in dirs
                    if not d.startswith(".") and d not in ("node_modules", "__pycache__", "venv", ".git")
                ]

                for filename in files:
                    if filename.startswith("."):
                        continue
                    filepath = Path(root) / filename

                    try:
                        text = filepath.read_text(encoding="utf-8", errors="replace")
                        for i, line in enumerate(text.split("\n"), 1):
                            if query.lower() in line.lower():
                                rel = filepath.relative_to(workspace)
                                shown = line.strip()
                                if len(shown) > max_line_chars:
                                    shown = shown[:max_line_chars] + "…"
                                results.append(f"{rel}:{i}: {shown}")
                                if len(results) >= max_results:
                                    results.append(f"... (truncated, {max_results} matches shown)")
                                    return "\n".join(results)
                    except (PermissionError, OSError):
                        continue
        except Exception as e:
            return f"Search error: {e}"

        if not results:
            return f"No matches found for '{query}'"

        return f"Found {len(results)} match(es):\n" + "\n".join(results)

    return StructuredTool(
        name="search_code",
        description="Search for a pattern in workspace files (like grep).",
        func=run,
        args_schema=SearchCodeInput,
    )


def _make_get_file_info(workspace_path: str) -> StructuredTool:
    def run(path: str) -> str:
        full_path = Path(workspace_path) / path
        if not full_path.exists():
            return f"Error: Path not found: {path}"

        stat = full_path.stat()
        info = {
            "path": path,
            "type": "directory" if full_path.is_dir() else "file",
            "size_bytes": stat.st_size,
            "modified": stat.st_mtime,
        }

        if full_path.is_file():
            info["extension"] = full_path.suffix
            info["name"] = full_path.name

        return json.dumps(info, indent=2)

    return StructuredTool(
        name="get_file_info",
        description="Get metadata about a file (size, type, modified time).",
        func=run,
        args_schema=GetFileInfoInput,
    )
