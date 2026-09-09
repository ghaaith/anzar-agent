"""SQLAlchemy base and session management."""

from __future__ import annotations

import logging

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from anzar.config import settings

logger = logging.getLogger("anzar.db")


class Base(DeclarativeBase):
    pass


def _get_engine():
    """Create engine - uses SQLite locally, PostgreSQL in production."""
    url = settings.database_url
    if url.startswith("sqlite"):
        eng = create_engine(url, echo=settings.debug)
        @event.listens_for(eng, "connect")
        def set_sqlite_pragma(dbapi_conn, _):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        return eng
    return create_engine(url, echo=settings.debug)


engine = _get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Dependency that provides a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables and apply any missing migrations."""
    import anzar.db.models  # noqa: F401 — ensure all models are registered
    Base.metadata.create_all(bind=engine)
    _run_migrations()


def _run_migrations():
    """Apply schema migrations for existing databases."""
    try:
        inspector = inspect(engine)

        # Add updated_at to conversations if missing
        columns = [c["name"] for c in inspector.get_columns("conversations")]
        if "updated_at" not in columns:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE conversations ADD COLUMN updated_at TIMESTAMP"))
                conn.commit()
                logger.info("Added missing column: conversations.updated_at")

        # Create index on messages.conversation_id if missing
        indexes = [ix["name"] for ix in inspector.get_indexes("messages")]
        if "ix_messages_conversation_id" not in indexes:
            with engine.connect() as conn:
                conn.execute(text("CREATE INDEX ix_messages_conversation_id ON messages(conversation_id)"))
                conn.commit()
                logger.info("Added missing index: ix_messages_conversation_id")

        # Add max_steps to settings if missing
        settings_cols = [c["name"] for c in inspector.get_columns("settings")]
        if "max_steps" not in settings_cols:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE settings ADD COLUMN max_steps INTEGER"))
                conn.commit()
                logger.info("Added missing column: settings.max_steps")
    except Exception:
        pass  # Table may not exist yet — handled by create_all
