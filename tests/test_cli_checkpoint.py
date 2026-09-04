"""Tests for the CLI workspace resolution used by checkpoint/diff/rollback/history.

Verifies the cwd-first behavior added to fix `anzar diff` showing the wrong
workspace (previously it defaulted to the stale ``last_workspace``).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from anzar.checkpoint import create_checkpoint
from anzar.cli import _resolve_subcommand_workspace
from anzar.config import AnzarConfig
from anzar.db.base import Base


def _make_db():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    return Session()


def _known(workspace=None):
    return SimpleNamespace(workspace=workspace)


def _config(last_workspace):
    return AnzarConfig(
        provider="groq",
        model="openai/gpt-oss-120b",
        last_workspace=last_workspace,
    )


def test_explicit_workspace_wins(tmp_path):
    db = _make_db()
    ws = tmp_path / "explicit"
    ws.mkdir()
    result = _resolve_subcommand_workspace(_known(workspace=str(ws)), db, _config(None))
    assert result == str(ws.absolute())


def test_cwd_used_when_it_has_checkpoints(tmp_path, monkeypatch):
    db = _make_db()
    # anzar worked in cwd before -> it has a checkpoint.
    create_checkpoint(str(tmp_path), db=db)
    monkeypatch.chdir(tmp_path)
    result = _resolve_subcommand_workspace(_known(workspace=None), db, _config("someone/else"))
    assert result == str(tmp_path.absolute())


def test_falls_back_to_last_workspace_when_cwd_bare(tmp_path, monkeypatch):
    db = _make_db()
    # cwd has NO checkpoints -> fall back to last_workspace.
    last_ws = tmp_path / "last"
    last_ws.mkdir()
    monkeypatch.chdir(tmp_path)
    result = _resolve_subcommand_workspace(_known(workspace=None), db, _config(str(last_ws)))
    assert result == str(last_ws.absolute())


def test_falls_back_to_cwd_when_nothing_available(tmp_path, monkeypatch):
    db = _make_db()
    monkeypatch.chdir(tmp_path)
    result = _resolve_subcommand_workspace(_known(workspace=None), db, _config(None))
    assert result == str(tmp_path.absolute())
