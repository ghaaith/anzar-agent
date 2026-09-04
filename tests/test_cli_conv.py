"""Tests for CLI conversation selection (auto-resume behavior)."""

from __future__ import annotations

import os
import tempfile
import uuid
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from anzar.cli import _select_conversation
from anzar.config import AnzarConfig
from anzar.db.base import Base
import anzar.db.models as models


def _make_db_user():
    engine = create_engine("sqlite://", poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    user = models.User(email=f"cli-{uuid.uuid4().hex}@test.local", name="CLI Tester")
    db.add(user)
    db.flush()
    return db, user


def _make_conv(db, user, title="Test Chat"):
    conv = models.Conversation(user_id=user.id, title=title)
    db.add(conv)
    db.commit()
    db.refresh(conv)
    return conv


def _known(**overrides):
    base = {"load": None, "resume": False, "new": False}
    base.update(overrides)
    return SimpleNamespace(**base)


def _config(conv_id=None, workspace=None):
    return AnzarConfig(
        provider="groq",
        model="openai/gpt-oss-120b",
        last_conversation_id=str(conv_id) if conv_id else None,
        last_workspace=workspace,
    )


def _fresh_workspace():
    return os.path.join(tempfile.gettempdir(), f"anzar-cli-test-{uuid.uuid4().hex}")


def test_default_resumes_same_workspace():
    db, user = _make_db_user()
    ws = _fresh_workspace()
    conv = _make_conv(db, user)
    config = _config(conv.id, ws)

    selected, is_new = _select_conversation(db, user, _known(), config, ws)

    assert is_new is False
    assert selected is not None
    assert selected.id == conv.id


def test_default_fresh_on_workspace_change():
    db, user = _make_db_user()
    old_ws = _fresh_workspace()
    conv = _make_conv(db, user)
    config = _config(conv.id, old_ws)

    selected, is_new = _select_conversation(db, user, _known(), config, _fresh_workspace())

    assert is_new is True
    assert selected is not None
    assert selected.id != conv.id


def test_new_flag_forces_fresh_conversation():
    db, user = _make_db_user()
    ws = _fresh_workspace()
    conv = _make_conv(db, user)
    config = _config(conv.id, ws)

    selected, is_new = _select_conversation(db, user, _known(new=True), config, ws)

    assert is_new is True
    assert selected.id != conv.id


def test_continue_flag_resumes_across_workspaces():
    db, user = _make_db_user()
    conv = _make_conv(db, user)
    config = _config(conv.id, _fresh_workspace())

    selected, is_new = _select_conversation(
        db, user, _known(resume=True), config, _fresh_workspace()
    )

    assert is_new is False
    assert selected.id == conv.id


def test_load_flag_loads_by_index():
    from datetime import datetime, timedelta, timezone

    db, user = _make_db_user()
    first = _make_conv(db, user, "Old Chat")
    second = _make_conv(db, user, "Recent Chat")
    # Make ordering deterministic (list is sorted by updated_at desc).
    now = datetime.now(timezone.utc)
    first.updated_at = now - timedelta(minutes=5)
    second.updated_at = now
    db.commit()
    config = _config()

    selected, is_new = _select_conversation(db, user, _known(load=1), config, _fresh_workspace())

    assert is_new is False
    assert selected.id == second.id
    assert selected.id != first.id


def test_no_saved_conversation_starts_fresh():
    db, user = _make_db_user()
    ws = _fresh_workspace()
    config = _config(None, ws)

    selected, is_new = _select_conversation(db, user, _known(), config, ws)

    assert is_new is True
    assert selected is not None


def test_stale_conversation_id_starts_fresh():
    db, user = _make_db_user()
    ws = _fresh_workspace()
    config = _config(uuid.uuid4(), ws)

    selected, is_new = _select_conversation(db, user, _known(), config, ws)

    assert is_new is True
    assert selected is not None
