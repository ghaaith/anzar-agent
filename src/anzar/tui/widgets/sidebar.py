"""Collapsible sidebar: tasks, project files, open files, and status."""

from __future__ import annotations

import itertools
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Collapsible, ListItem, ListView, Static, Tree

from anzar.tui.theme import ERROR, MUTED, SUCCESS

_SPINNER = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
_SKIP = {".git", "node_modules", "__pycache__", "venv", ".venv", "dist", "build", ".next", "target"}


class FileTree(Tree[dict]):
    """Lazy depth-limited file browser for the workspace."""

    def __init__(self, **kwargs):
        super().__init__("workspace", **kwargs)
        self._workspace_path = ""

    def rebuild(self, workspace_path: str, max_depth: int = 3) -> None:
        self._workspace_path = workspace_path
        self.clear()
        root = Path(workspace_path)
        self.root.set_label(root.name or workspace_path)
        self.root.expand()
        self._populate(self.root, root, 0, max_depth)

    def _populate(self, node, path: Path, depth: int, max_depth: int) -> None:
        try:
            entries = sorted(
                (p for p in path.iterdir() if not p.name.startswith(".") and p.name not in _SKIP),
                key=lambda p: (p.is_file(), p.name.lower()),
            )
        except OSError:
            return
        for p in entries:
            if p.is_dir():
                child = node.add(p.name + "/", expand=False)
                if depth + 1 < max_depth:
                    self._populate(child, p, depth + 1, max_depth)
            else:
                node.add_leaf(p.name)


class Sidebar(Container):
    """Left-hand panel with files, open files, tasks, and status."""

    def __init__(self, workspace_path: str, **kwargs):
        super().__init__(id="sidebar", **kwargs)
        self._workspace_path = workspace_path
        self._open_files: list[str] = []
        self._tasks: dict[str, ListItem] = {}
        self._task_spinners: dict[str, Static] = {}
        self._task_timer = None

    def compose(self) -> ComposeResult:
        with Collapsible(title="[bold]Tasks[/bold]", collapsed=False):
            yield ListView(id="tasks-list")
        with Collapsible(title="[bold]Files[/bold]", collapsed=False):
            yield FileTree(id="file-tree")
        with Collapsible(title="[bold]Open files[/bold]", collapsed=False):
            yield ListView(id="open-files")
        with Collapsible(title="[bold]Status[/bold]", collapsed=False):
            yield Static("", id="status-text")

    def on_mount(self) -> None:
        self.query_one(FileTree).rebuild(self._workspace_path)
        self._task_timer = self.set_interval(0.1, self._tick_tasks)

    def _tick_tasks(self) -> None:
        if not self._task_spinners:
            return
        frame = next(_SPINNER)
        for static in self._task_spinners.values():
            static.update(f"[{MUTED}]{frame} {static._task_name}[/{MUTED}]")

    def refresh_files(self) -> None:
        self.query_one(FileTree).rebuild(self._workspace_path)

    def set_workspace(self, path: str) -> None:
        self._workspace_path = path
        self.refresh_files()

    def add_open_file(self, path: str) -> None:
        if path in self._open_files:
            return
        self._open_files.append(path)
        self._open_files = self._open_files[-20:]
        lv = self.query_one("#open-files", ListView)
        lv.clear()
        for p in self._open_files:
            lv.append(ListItem(Static(p)))

    def add_task(self, call_id: str, name: str) -> None:
        static = Static(f"[{MUTED}]⠋ {name}[/{MUTED}]")
        static._task_name = name
        item = ListItem(static)
        self._tasks[call_id] = (item, static)
        self._task_spinners[call_id] = static
        lv = self.query_one("#tasks-list", ListView)
        lv.append(item)
        collapsible = lv.parent
        if getattr(collapsible, "collapsed", False):
            collapsible.collapsed = False
        self.scroll_end(animate=False)

    def finish_task(self, call_id: str, name: str, ok: bool) -> None:
        entry = self._tasks.pop(call_id, None)
        if entry is None:
            return
        _item, static = entry
        self._task_spinners.pop(call_id, None)
        mark = "✓" if ok else "✗"
        color = SUCCESS if ok else ERROR
        static.update(f"[{color}]{mark} {name}[/{color}]")

    def clear_tasks(self) -> None:
        self._tasks.clear()
        self._task_spinners.clear()
        self.query_one("#tasks-list", ListView).clear()

    def set_status_lines(self, lines: list[tuple[str, str]]) -> None:
        text = "\n".join(f"[{MUTED}]{k}:[/{MUTED}] [bold]{v}[/bold]" for k, v in lines)
        self.query_one("#status-text", Static).update(text)
