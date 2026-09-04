"""Conversation memory management for the agent."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from sqlalchemy.orm import Session

from anzar.db.models import Conversation, Message


class ConversationMemory:
    """Loads and saves conversation history between the agent and the database."""

    def __init__(self, db: Session, conversation_id: uuid.UUID, workspace_id: uuid.UUID | None = None):
        self.db = db
        self.conversation_id = conversation_id
        self.workspace_id = workspace_id

    def load_messages(self, limit: int = 30) -> list[BaseMessage]:
        """Load conversation history as LangChain messages.

        Args:
            limit: Max recent messages to load.

        Long contents of older messages are truncated in the replay only
        (never written back to the DB) to keep the prompt small and fast.
        """
        db_messages = (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conversation_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
            .all()
        )
        db_messages.reverse()

        # Keep the most recent messages intact for full context; shorten the
        # bodies of older ones to avoid token bloat on every turn.
        keep_full = 6
        max_content = 2000

        def _trim(content: str, index: int) -> str:
            if index < len(db_messages) - keep_full and len(content) > max_content:
                return content[:max_content] + "\n...[truncated in replay]"
            return content

        messages: list[BaseMessage] = []
        for i, msg in enumerate(db_messages):
            if msg.role == "user":
                messages.append(HumanMessage(content=_trim(msg.content, i)))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=_trim(msg.content, i)))
            elif msg.role == "system":
                messages.append(SystemMessage(content=_trim(msg.content, i)))
            elif msg.role == "tool":
                # Parse tool call info from metadata
                metadata = {}
                if msg.metadata_json:
                    try:
                        metadata = json.loads(msg.metadata_json)
                    except json.JSONDecodeError:
                        pass
                messages.append(
                    ToolMessage(
                        content=_trim(msg.content, i),
                        tool_call_id=metadata.get("tool_call_id", ""),
                    )
                )

        return messages

    def add_user_message(self, content: str) -> Message:
        """Save a user message to the database."""
        msg = Message(
            conversation_id=self.conversation_id,
            role="user",
            content=content,
        )
        self.db.add(msg)
        self.db.flush()
        self.db.refresh(msg)
        return msg

    def add_assistant_message(self, content: str, metadata: dict | None = None) -> Message:
        """Save an assistant message to the database."""
        msg = Message(
            conversation_id=self.conversation_id,
            role="assistant",
            content=content,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        self.db.add(msg)
        self.db.flush()
        self.db.refresh(msg)
        return msg

    def add_tool_message(self, content: str, tool_call_id: str = "") -> Message:
        """Save a tool result message to the database."""
        msg = Message(
            conversation_id=self.conversation_id,
            role="tool",
            content=content,
            metadata_json=json.dumps({"tool_call_id": tool_call_id}),
        )
        self.db.add(msg)
        self.db.flush()
        self.db.refresh(msg)
        return msg

    def commit(self):
        """Commit all pending writes in one batch."""
        self.db.commit()

    def get_message_count(self) -> int:
        """Get the total number of messages in this conversation."""
        return (
            self.db.query(Message)
            .filter(Message.conversation_id == self.conversation_id)
            .count()
        )

    def clear(self):
        """Delete all messages in this conversation."""
        self.db.query(Message).filter(
            Message.conversation_id == self.conversation_id
        ).delete()
        self.db.commit()
