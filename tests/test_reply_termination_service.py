"""Tests for reply_termination_service's _persist_cancelled_reply, apart from the full detached-generation
machinery (tests/test_reply_generation_service.py covers it end to end). Uses that module's in-memory
session_factory pattern, not the shared `db` fixture, since AsyncSessionLocal is patched directly.
"""

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Conversation, Message
from app.services import reply_termination_service


async def _build_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _make_streaming_message(session_factory) -> tuple[str, str]:
    async with session_factory() as db:
        conversation = Conversation(owner_id=None, title="general", model="fake-model", params={})
        db.add(conversation)
        await db.commit()
        message = Message(conversation_id=conversation.id, role="assistant", content="partial", status="streaming")
        db.add(message)
        await db.commit()
        return conversation.id, message.id


@pytest.mark.asyncio
async def test_persist_cancelled_reply_stops_the_model_when_the_message_row_is_gone(monkeypatch):
    """Guards the bug where deleting a whole conversation mid-reply cascade-deletes its Message row first, so the
    message lookup returns None. That must NOT skip stop_model: the engine needs the stop signal even with no row
    to write the cancelled text into (conversation_service.delete_conversation relies on it).
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    stopped_models = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    await reply_termination_service._persist_cancelled_reply(
        "missing-message-id", "fake-model", "partial", "missing-conversation-id"
    )

    assert stopped_models == ["fake-model"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_persist_cancelled_reply_writes_the_message_and_stops_the_model(monkeypatch):
    """The normal case: the row exists, so both the error write and the stop_model call happen."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    conversation_id, message_id = await _make_streaming_message(session_factory)
    stopped_models = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    await reply_termination_service._persist_cancelled_reply(message_id, "fake-model", "partial", conversation_id)

    assert stopped_models == ["fake-model"]
    async with session_factory() as db:
        message = await db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
    await engine.dispose()
