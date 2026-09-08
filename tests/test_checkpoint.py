"""Tests for the checkpoint, diff, and rollback infrastructure.

Covers file-based (non-Git) snapshotting end-to-end: checkpoint creation,
diff computation, rollback safety (never deletes files created after the
checkpoint, never creates files), and task history recording.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from anzar.checkpoint import (
    CheckpointResult,
    compute_diff,
    compute_diff_between,
    compute_diff_from_result,
    create_checkpoint,
    format_task_history,
    get_latest_checkpoint,
    get_task_history,
    record_task,
    rollback_checkpoint,
    rollback_from_result,
)
from anzar.db.base import Base

GIT_AVAILABLE = shutil.which("git") is not None


def _make_db():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _git(*args, cwd):
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30
    )


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def _init_git_repo(path):
    _git("init", "-q", cwd=str(path))
    _git("config", "user.email", "test@example.com", cwd=str(path))
    _git("config", "user.name", "Test", cwd=str(path))
    (path / "base.txt").write_text("base\n", encoding="utf-8")
    _git("add", ".", cwd=str(path))
    _git("commit", "-q", "-m", "init", cwd=str(path))


# ---------------------------------------------------------------------------
# Checkpoint creation & diff
# ---------------------------------------------------------------------------


def test_create_and_diff_file_based(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "util.py").write_text("x = 2\n", encoding="utf-8")

    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db, description="before work")
    assert isinstance(cp, CheckpointResult)
    assert cp.snapshot_path is not None and cp.file_count == 2

    # Modify + add + delete
    (tmp_path / "main.py").write_text("a = 2\nb = 3\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("y = 9\n", encoding="utf-8")
    (tmp_path / "app" / "util.py").unlink()

    diff = compute_diff(cp.id, str(tmp_path), db)
    assert "main.py" in diff.modified
    assert "new.py" in diff.added
    assert "app/util.py" in diff.deleted

    # No-db path
    diff2 = compute_diff_from_result(cp, str(tmp_path))
    assert diff2.has_changes is True


def test_diff_no_changes(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db)
    diff = compute_diff(cp.id, str(tmp_path), db)
    assert diff.has_changes is False
    assert diff.summary == "No changes"


def test_diff_between_file_based(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("keep\n", encoding="utf-8")

    db = _make_db()
    cp_a = create_checkpoint(str(tmp_path), db=db, description="older")

    # Modify, add, and delete between checkpoints.
    (tmp_path / "main.py").write_text("a = 2\n", encoding="utf-8")
    (tmp_path / "new.py").write_text("n = 1\n", encoding="utf-8")
    (tmp_path / "keep.py").unlink()

    cp_b = create_checkpoint(str(tmp_path), db=db, description="newer")

    diff = compute_diff_between(cp_a.id, cp_b.id, str(tmp_path), db)
    assert "main.py" in diff.modified
    assert "new.py" in diff.added
    assert "keep.py" in diff.deleted
    assert diff.has_changes is True


def test_diff_between_identical_checkpoints(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    db = _make_db()
    cp_a = create_checkpoint(str(tmp_path), db=db)
    cp_b = create_checkpoint(str(tmp_path), db=db)
    diff = compute_diff_between(cp_a.id, cp_b.id, str(tmp_path), db)
    assert diff.has_changes is False
    assert diff.unchanged == 1


def test_diff_between_unknown_checkpoint_raises(tmp_path):
    import uuid as _uuid

    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db)
    try:
        compute_diff_between(_uuid.uuid4(), cp.id, str(tmp_path), db)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_restores_modified_and_deleted(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("keep\n", encoding="utf-8")

    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db)

    # Mutate: edit main.py, delete keep.py
    (tmp_path / "main.py").write_text("CHANGED\n", encoding="utf-8")
    (tmp_path / "keep.py").unlink()

    restored = rollback_checkpoint(cp.id, str(tmp_path), db)

    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "a = 1\n"
    assert (tmp_path / "keep.py").read_text(encoding="utf-8") == "keep\n"
    # keep.py existed at cp time -> restored (or counted as restored)
    assert restored  # at least main.py


def test_rollback_never_deletes_files_created_after(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db)

    # File created AFTER the checkpoint must survive rollback.
    (tmp_path / "after.py").write_text("new\n", encoding="utf-8")

    rollback_checkpoint(cp.id, str(tmp_path), db)

    assert (tmp_path / "after.py").is_file()
    assert (tmp_path / "main.py").is_file()


def test_rollback_from_result_no_db(tmp_path):
    (tmp_path / "main.py").write_text("orig\n", encoding="utf-8")
    cp = create_checkpoint(str(tmp_path), db=None)
    (tmp_path / "main.py").write_text("mutated\n", encoding="utf-8")
    rollback_from_result(cp, str(tmp_path))
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "orig\n"


def test_rollback_nonexistent_checkpoint_raises(tmp_path):
    import uuid as _uuid

    db = _make_db()
    try:
        rollback_checkpoint(_uuid.uuid4(), str(tmp_path), db)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Task history
# ---------------------------------------------------------------------------


def test_record_and_list_tasks(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n", encoding="utf-8")
    db = _make_db()
    cp = create_checkpoint(str(tmp_path), db=db, description="setup")
    (tmp_path / "main.py").write_text("a = 2\n", encoding="utf-8")

    record_task(
        db=db,
        workspace_path=str(tmp_path),
        description="implement feature",
        status="completed",
        checkpoint_id=cp.id,
        files_created=["new.py"],
        files_modified=["main.py"],
        files_deleted=[],
        verification_status="passed",
    )

    tasks = get_task_history(str(tmp_path), db, limit=10)
    assert len(tasks) == 1
    t = tasks[0]
    assert t.description == "implement feature"
    assert t.checkpoint_id == cp.id
    assert "new.py" in t.files_created
    assert "main.py" in t.files_modified
    assert t.verification_status == "passed"


def test_task_history_scope(tmp_path, tmp_path_factory):
    other = tmp_path_factory.mktemp("other")
    (tmp_path / "a.py").write_text("a", encoding="utf-8")
    (other / "b.py").write_text("b", encoding="utf-8")
    db = _make_db()

    record_task(db=db, workspace_path=str(tmp_path), description="task A")
    record_task(db=db, workspace_path=str(other), description="task B")

    tasks_a = get_task_history(str(tmp_path), db)
    assert len(tasks_a) == 1
    assert tasks_a[0].description == "task A"


def test_format_task_history_empty_and_nonempty(tmp_path):
    db = _make_db()
    assert format_task_history([]) == "No task history found."

    record_task(db=db, workspace_path=str(tmp_path), description="x")
    text = format_task_history(get_task_history(str(tmp_path), db))
    assert "x" in text
    assert "ok" in text


def test_get_latest_checkpoint(tmp_path):
    db = _make_db()
    assert get_latest_checkpoint(str(tmp_path), db) is None
    c1 = create_checkpoint(str(tmp_path), db=db, description="one")
    c2 = create_checkpoint(str(tmp_path), db=db, description="two")
    latest = get_latest_checkpoint(str(tmp_path), db)
    assert latest is not None
    assert latest.id == c2.id
    assert latest.id != c1.id


# ---------------------------------------------------------------------------
# Git-based checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_git_checkpoint_uses_commit_not_stash_clean_tree(tmp_path):
    _init_git_repo(tmp_path)
    db = _make_db()

    cp = create_checkpoint(str(tmp_path), db=db, description="clean")

    # Clean tree -> baseline is HEAD, not a stash-created dangling commit.
    assert cp.git_commit, cp.git_commit
    head = _git("rev-parse", "HEAD", cwd=str(tmp_path)).stdout.strip()
    assert cp.git_commit == head


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_git_checkpoint_dirty_tree_not_hidden(tmp_path):
    _init_git_repo(tmp_path)
    db = _make_db()

    # Simulate pre-existing uncommitted work that the user must keep.
    (tmp_path / "wip.py").write_text("wip\n", encoding="utf-8")
    (tmp_path / "base.txt").write_text("modified before agent\n", encoding="utf-8")

    cp = create_checkpoint(str(tmp_path), db=db, description="dirty baseline")

    # The user's work must still be present in the working tree afterwards.
    assert (tmp_path / "wip.py").read_text(encoding="utf-8") == "wip\n"
    assert (tmp_path / "base.txt").read_text(encoding="utf-8") == "modified before agent\n"

    # The agent then creates a new file.
    (tmp_path / "agent_added.py").write_text("agent\n", encoding="utf-8")

    diff = compute_diff(cp.id, str(tmp_path), db)
    # base.txt is the pre-existing dirty change (should NOT appear as new/agent work)
    assert "base.txt" not in diff.modified, diff.modified
    assert "wip.py" not in diff.added, diff.added
    # But the agent's new file should be picked up as added by git diff + untracked.
    assert "agent_added.py" in diff.added, diff.added


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_git_checkpoint_rollback_restores_baseline(tmp_path):
    _init_git_repo(tmp_path)
    db = _make_db()

    # Pre-existing change captured by the checkpoint baseline.
    (tmp_path / "base.txt").write_text("pre-change\n", encoding="utf-8")
    cp = create_checkpoint(str(tmp_path), db=db, description="baseline")

    # Agent overwrites it.
    (tmp_path / "base.txt").write_text("agent overwrote\n", encoding="utf-8")

    restored = rollback_checkpoint(cp.id, str(tmp_path), db)
    assert (tmp_path / "base.txt").read_text(encoding="utf-8") == "pre-change\n"


@pytest.mark.skipif(not GIT_AVAILABLE, reason="git not installed")
def test_diff_between_git_checkpoints(tmp_path):
    _init_git_repo(tmp_path)
    db = _make_db()

    cp_a = create_checkpoint(str(tmp_path), db=db, description="older")
    assert cp_a.git_commit

    (tmp_path / "base.txt").write_text("changed after A\n", encoding="utf-8")
    (tmp_path / "added_after.py").write_text("x = 1\n", encoding="utf-8")

    cp_b = create_checkpoint(str(tmp_path), db=db, description="newer")
    assert cp_b.git_commit

    diff = compute_diff_between(cp_a.id, cp_b.id, str(tmp_path), db)
    assert "base.txt" in diff.modified
    assert "added_after.py" in diff.added
    assert diff.has_changes is True
