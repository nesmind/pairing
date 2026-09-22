"""Unit tests for app/services/reply_termination_service.py's
_persist_cancelled_reply, in isolation from the full detached-generation
machinery tests/test_reply_generation_service.py already covers end to
end. Uses that same module's own lightweight in-memory session_factory
pattern rather than the shared `db` fixture, since AsyncSessionLocal is
monkeypatched directly.
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
    """The bug this guards: deleting a whole conversation mid-reply
    cascade-deletes its Message row before this ever runs, so the
    message-lookup below returns None. That must NOT skip the stop_model
    call — Matricxon/Ollama needs the stop signal regardless of whether
    there's still a row left to write the cancelled text into (see
    conversation_service.delete_conversation, which relies on this)."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    stopped_models = []

    async def fake_stop_model(model):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    await reply_termination_service._persist_cancelled_reply(
        "missing-message-id", "fake-model", "partial", "missing-conversation-id"
    )

    assert stopped_models == ["fake-model"]
    await engine.dispose()


@pytest.mark.asyncio
async def test_persist_cancelled_reply_writes_the_message_and_stops_the_model(monkeypatch):
    """The normal case: the row still exists, so both the error write and
    the stop_model call must happen."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    conversation_id, message_id = await _make_streaming_message(session_factory)
    stopped_models = []

    async def fake_stop_model(model):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    await reply_termination_service._persist_cancelled_reply(message_id, "fake-model", "partial", conversation_id)

    assert stopped_models == ["fake-model"]
    async with session_factory() as db:
        message = await db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
    await engine.dispose()
