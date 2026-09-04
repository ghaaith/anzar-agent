"""Checkpoint, diff, rollback, and task history for the Anzar agent.

Provides workspace snapshotting with automatic Git integration when available,
falling back to file-based copies for non-Git projects.

Safety guarantees:
- Checkpoint is created BEFORE any destructive modification.
- Rollback only restores files that existed at checkpoint time.
- Rollback never deletes files created after the checkpoint.
- Failed checkpoint creation aborts the agent turn.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from anzar.db.models import Checkpoint, Task

# Where file-based snapshots live
_SNAPSHOT_ROOT = Path.home() / ".anzar" / "checkpoints"

# Directories excluded from snapshots (same as verifier)
_SKIP_DIRS = {
    ".git", ".hg", ".svn",
    "node_modules", "venv", ".venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "dist", "build", "target", ".cache", "coverage",
    ".anzar",
}


# ---------------------------------------------------------------------------
# Git detection
# ---------------------------------------------------------------------------

def _is_git_repo(path: str) -> bool:
    """Return True if *path* is inside a Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0 and "true" in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_has_commits(path: str) -> bool:
    """Return True if the repo has at least one commit."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Workspace file scanning
# ---------------------------------------------------------------------------

def _scan_workspace(workspace_path: str) -> dict[str, str]:
    """Return ``{relative_path: sha256_hex}`` for all source files."""
    root = Path(workspace_path)
    manifest: dict[str, str] = {}
    if not root.is_dir():
        return manifest
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                continue
            try:
                h = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                continue
            manifest[rel] = h
    return manifest


def _scan_workspace_meta(workspace_path: str) -> dict[str, tuple[int, float]]:
    """Return ``{relative_path: (size, mtime)}`` — same shape as verifier snapshot."""
    root = Path(workspace_path)
    snap: dict[str, tuple[int, float]] = {}
    if not root.is_dir():
        return snap
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            try:
                st = p.stat()
                rel = str(p.relative_to(root)).replace("\\", "/")
                snap[rel] = (st.st_size, st.st_mtime)
            except OSError:
                continue
    return snap


# ---------------------------------------------------------------------------
# Checkpoint dataclass
# ---------------------------------------------------------------------------

@dataclass
class CheckpointResult:
    """Result of creating a checkpoint."""
    id: uuid.UUID
    workspace_path: str
    is_git: bool
    git_commit: str | None = None
    snapshot_path: str | None = None
    description: str | None = None
    file_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class DiffResult:
    """Result of computing a diff between checkpoint and current workspace."""
    added: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    unchanged: int = 0

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    @property
    def summary(self) -> str:
        parts = []
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.modified:
            parts.append(f"{len(self.modified)} modified")
        if self.deleted:
            parts.append(f"{len(self.deleted)} deleted")
        if not parts:
            return "No changes"
        return ", ".join(parts)


@dataclass
class TaskRecord:
    """A recorded agent task."""
    id: uuid.UUID
    checkpoint_id: uuid.UUID | None
    description: str | None
    status: str
    files_created: list[str]
    files_modified: list[str]
    files_deleted: list[str]
    verification_status: str | None
    created_at: datetime


# ---------------------------------------------------------------------------
# Checkpoint creation
# ---------------------------------------------------------------------------

def create_checkpoint(
    workspace_path: str,
    db: Session | None = None,
    description: str | None = None,
) -> CheckpointResult:
    """Create a recoverable snapshot of the workspace.

    If the workspace is a Git repo with commits, uses ``git stash`` to capture
    the state.  Otherwise copies source files to ``~/.anzar/checkpoints/<id>/``.

    Returns a :class:`CheckpointResult`.  If *db* is provided the checkpoint
    metadata is persisted; otherwise it lives only in memory (useful for tests).
    """
    ws = os.path.abspath(workspace_path)
    cp_id = uuid.uuid4()
    manifest = _scan_workspace(ws)
    is_git = _is_git_repo(ws)
    git_commit: str | None = None
    snapshot_path: str | None = None

    if is_git and _git_has_commits(ws):
        # --- Git-based checkpoint ---
        git_commit = _git_checkpoint(ws)
    else:
        # --- File-based checkpoint ---
        snapshot_path = _file_checkpoint(ws, cp_id)

    result = CheckpointResult(
        id=cp_id,
        workspace_path=ws,
        is_git=is_git,
        git_commit=git_commit,
        snapshot_path=snapshot_path,
        description=description,
        file_count=len(manifest),
    )

    if db is not None:
        next_ordinal = (db.query(func.max(Checkpoint.ordinal)).scalar() or 0) + 1
        cp = Checkpoint(
            id=cp_id,
            ordinal=next_ordinal,
            workspace_path=ws,
            git_commit=git_commit,
            snapshot_path=snapshot_path,
            description=description,
            files_manifest=json.dumps(manifest),
        )
        db.add(cp)
        db.commit()

    return result


def _git_checkpoint(workspace_path: str) -> str:
    """Create a Git checkpoint by snapshotting the current state as a commit.

    Stages all changes (including untracked files), snaps the index tree into a
    *dangling* commit via ``write-tree`` + ``commit-tree``, then unstages
    (``git reset --mixed HEAD``) so the working tree and the user's uncommitted
    work are left untouched. The returned commit hash is the diff/rollback
    baseline and includes every file present at checkpoint time.
    """
    # If the tree is clean (no tracked changes, no untracked files), the
    # baseline is simply HEAD — no new commit is needed.
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if changed.returncode == 0 and not changed.stdout.strip():
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return head.stdout.strip() if head.returncode == 0 else "unknown"

    # Stage everything (tracked modifications + untracked files).
    subprocess.run(
        ["git", "add", "-A"],
        cwd=workspace_path,
        capture_output=True,
        timeout=30,
    )
    tree = subprocess.run(
        ["git", "write-tree"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if tree.returncode != 0 or not tree.stdout.strip():
        # Nothing staged (clean tree) or write failed — baseline is HEAD.
        subprocess.run(
            ["git", "reset", "--mixed", "HEAD"],
            cwd=workspace_path,
            capture_output=True,
            timeout=30,
        )
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return head.stdout.strip() if head.returncode == 0 else "unknown"

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=5,
    )
    parent = head.stdout.strip() if head.returncode == 0 else "--orphan"
    commit_args = ["git", "commit-tree", tree.stdout.strip(), "-m", "anzar checkpoint"]
    if parent != "--orphan":
        commit_args += ["-p", parent]
    commit = subprocess.run(
        commit_args,
        cwd=workspace_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    commit_hash = commit.stdout.strip() if commit.returncode == 0 else "unknown"

    # Restore the index so the working tree & staging are exactly as before.
    subprocess.run(
        ["git", "reset", "--mixed", "HEAD"],
        cwd=workspace_path,
        capture_output=True,
        timeout=30,
    )
    return commit_hash


def _file_checkpoint(workspace_path: str, cp_id: uuid.UUID) -> str:
    """Copy workspace source files to a snapshot directory. Returns the path."""
    snap_dir = _SNAPSHOT_ROOT / cp_id.hex
    snap_dir.mkdir(parents=True, exist_ok=True)
    root = Path(workspace_path)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            src = base / name
            try:
                rel = src.relative_to(root)
            except ValueError:
                continue
            dst = snap_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(str(src), str(dst))
            except OSError:
                continue
    return str(snap_dir)


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def compute_diff(checkpoint_id: uuid.UUID, workspace_path: str, db: Session) -> DiffResult:
    """Compute the diff between a checkpoint and the current workspace."""
    cp = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    if cp is None:
        raise ValueError(f"Checkpoint {checkpoint_id} not found")

    if cp.git_commit and cp.git_commit != "unknown":
        return _git_diff(workspace_path, cp.git_commit)
    elif cp.snapshot_path:
        return _file_diff(cp.snapshot_path, workspace_path)
    else:
        return DiffResult()


def compute_diff_from_result(cp: CheckpointResult, workspace_path: str) -> DiffResult:
    """Compute diff from a CheckpointResult (no DB needed — for tests)."""
    if cp.git_commit and cp.git_commit != "unknown":
        return _git_diff(workspace_path, cp.git_commit)
    elif cp.snapshot_path:
        return _file_diff(cp.snapshot_path, workspace_path)
    return DiffResult()


def get_latest_checkpoint(workspace_path: str, db: Session) -> Checkpoint | None:
    """Get the most recent checkpoint for a workspace."""
    ws = os.path.abspath(workspace_path)
    return (
        db.query(Checkpoint)
        .filter(Checkpoint.workspace_path == ws)
        .order_by(Checkpoint.ordinal.desc())
        .first()
    )


def _git_tree_files(workspace_path: str, ref: str) -> list[str]:
    """Return the paths present in a git ref's tree (empty on failure)."""
    try:
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", ref],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
        return [f for f in result.stdout.strip().splitlines() if f]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def _git_diff(workspace_path: str, from_commit: str) -> DiffResult:
    """Use Git to compute added/modified/deleted files."""
    result = DiffResult()
    try:
        # Check if commit still exists
        check = subprocess.run(
            ["git", "cat-file", "-t", from_commit],
            cwd=workspace_path,
            capture_output=True,
            timeout=5,
        )
        if check.returncode != 0:
            return result

        # Get diff between commit and current working tree (including untracked)
        diff = subprocess.run(
            ["git", "diff", "--name-status", from_commit],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in diff.stdout.strip().splitlines():
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue
            status, name = parts[0], parts[1]
            if status == "A":
                result.added.append(name)
            elif status == "M":
                result.modified.append(name)
            elif status == "D":
                result.deleted.append(name)

        # Also detect untracked files — but only ones that were NOT already
        # present at the baseline. The baseline is a ``stash create -u`` commit,
        # so pre-existing untracked files live in its tree and are not new.
        baseline_files = set(_git_tree_files(workspace_path, from_commit))
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in untracked.stdout.strip().splitlines():
            if line and line not in result.added and line not in baseline_files:
                result.added.append(line)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return result


def _file_diff(snapshot_path: str, workspace_path: str) -> DiffResult:
    """Compare a file-based snapshot with the current workspace."""
    result = DiffResult()
    snap_dir = Path(snapshot_path)
    root = Path(workspace_path)

    if not snap_dir.is_dir():
        return result

    # Build snapshot manifest
    snap_files: dict[str, str] = {}
    for p in snap_dir.rglob("*"):
        if p.is_file():
            try:
                rel = str(p.relative_to(snap_dir)).replace("\\", "/")
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                snap_files[rel] = h
            except OSError:
                continue

    # Build current manifest
    cur_files: dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        base = Path(dirpath)
        for name in filenames:
            p = base / name
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                cur_files[rel] = h
            except OSError:
                continue

    # Compute diff
    all_files = set(snap_files.keys()) | set(cur_files.keys())
    for f in sorted(all_files):
        in_snap = f in snap_files
        in_cur = f in cur_files
        if in_snap and not in_cur:
            result.deleted.append(f)
        elif not in_snap and in_cur:
            result.added.append(f)
        elif snap_files[f] != cur_files[f]:
            result.modified.append(f)
        else:
            result.unchanged += 1

    return result


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------

def rollback_checkpoint(
    checkpoint_id: uuid.UUID,
    workspace_path: str,
    db: Session,
) -> list[str]:
    """Restore the workspace to a checkpoint. Returns list of restored file paths.

    Only restores files that existed at checkpoint time. Never deletes files
    created after the checkpoint.
    """
    cp = db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
    if cp is None:
        raise ValueError(f"Checkpoint {checkpoint_id} not found")

    if cp.git_commit and cp.git_commit != "unknown":
        return _git_rollback(workspace_path, cp.git_commit)
    elif cp.snapshot_path:
        return _file_rollback(cp.snapshot_path, workspace_path)
    return []


def rollback_from_result(cp: CheckpointResult, workspace_path: str) -> list[str]:
    """Rollback from a CheckpointResult (no DB needed — for tests)."""
    if cp.git_commit and cp.git_commit != "unknown":
        return _git_rollback(workspace_path, cp.git_commit)
    elif cp.snapshot_path:
        return _file_rollback(cp.snapshot_path, workspace_path)
    return []


def _git_rollback(workspace_path: str, from_commit: str) -> list[str]:
    """Restore files using Git checkout from a specific commit."""
    restored: list[str] = []
    try:
        # Get list of files at the checkpoint commit
        result = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", from_commit],
            cwd=workspace_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return restored

        files = [f for f in result.stdout.strip().splitlines() if f]
        # Checkout each file from the checkpoint commit
        for f in files:
            checkout = subprocess.run(
                ["git", "checkout", from_commit, "--", f],
                cwd=workspace_path,
                capture_output=True,
                timeout=10,
            )
            if checkout.returncode == 0:
                restored.append(f)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return restored


def _file_rollback(snapshot_path: str, workspace_path: str) -> list[str]:
    """Restore files from a file-based snapshot."""
    snap_dir = Path(snapshot_path)
    root = Path(workspace_path)
    restored: list[str] = []

    if not snap_dir.is_dir():
        return restored

    for p in snap_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            rel = p.relative_to(snap_dir)
        except ValueError:
            continue
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(p), str(dst))
            restored.append(str(rel).replace("\\", "/"))
        except OSError:
            continue

    return restored


# ---------------------------------------------------------------------------
# Task history
# ---------------------------------------------------------------------------

def record_task(
    db: Session,
    workspace_path: str,
    description: str | None = None,
    status: str = "completed",
    checkpoint_id: uuid.UUID | None = None,
    files_created: list[str] | None = None,
    files_modified: list[str] | None = None,
    files_deleted: list[str] | None = None,
    verification_status: str | None = None,
) -> Task:
    """Record an agent task in the database."""
    task = Task(
        id=uuid.uuid4(),
        checkpoint_id=checkpoint_id,
        workspace_path=os.path.abspath(workspace_path),
        description=description,
        status=status,
        files_created=json.dumps(files_created or []),
        files_modified=json.dumps(files_modified or []),
        files_deleted=json.dumps(files_deleted or []),
        verification_status=verification_status,
    )
    db.add(task)
    db.commit()
    return task


def get_task_history(
    workspace_path: str,
    db: Session,
    limit: int = 20,
) -> list[Task]:
    """Get recent tasks for a workspace."""
    ws = os.path.abspath(workspace_path)
    return (
        db.query(Task)
        .filter(Task.workspace_path == ws)
        .order_by(Task.created_at.desc())
        .limit(limit)
        .all()
    )


def format_task_history(tasks: list[Task]) -> str:
    """Format task history as a readable table."""
    if not tasks:
        return "No task history found."
    lines = []
    for t in tasks:
        created = t.created_at.strftime("%Y-%m-%d %H:%M") if t.created_at else "?"
        cp_short = str(t.checkpoint_id)[:8] if t.checkpoint_id else "-"
        created_files = json.loads(t.files_created) if t.files_created else []
        modified_files = json.loads(t.files_modified) if t.files_modified else []
        deleted_files = json.loads(t.files_deleted) if t.files_deleted else []
        changes = []
        if created_files:
            changes.append(f"+{len(created_files)}")
        if modified_files:
            changes.append(f"~{len(modified_files)}")
        if deleted_files:
            changes.append(f"-{len(deleted_files)}")
        change_str = " ".join(changes) if changes else "no changes"
        status_icon = {"completed": "ok", "failed": "!!", "interrupted": "~~"}.get(t.status, t.status)
        desc = (t.description or "(no description)")[:60]
        lines.append(f"  [{status_icon}] {created}  cp:{cp_short}  {change_str}  {desc}")
    return "\n".join(lines)
