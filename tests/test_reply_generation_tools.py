"""A reply that used tools: the tool calls are stored on the message and sent live with the text."""

import pytest
from sqlalchemy import select

from app.models import Conversation, Message
from app.services import (
    reply_cross_instance_service,
    reply_generation_service,
    reply_termination_service,
    tool_event_service,
    tool_loop_service,
)
from tests.test_reply_generation_service import _build_session_factory, _drain_background_tasks, _make_conversation

EVENT = {
    "id": "call_1",
    "name": "fake__add",
    "server": "fake",
    "tool": "add",
    "arguments": {"a": 1, "b": 2},
    "result": "3",
    "is_error": False,
}


@pytest.mark.asyncio
async def test_tool_events_are_stored_on_the_message_and_streamed_with_the_text(monkeypatch):
    engine, session_factory = await _build_session_factory()
    for module in (reply_generation_service, reply_termination_service, reply_cross_instance_service):
        monkeypatch.setattr(module, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)
    seen_params: list[dict] = []

    async def fake_tool_stream(_db, _model, _messages, params):
        seen_params.append(params)

        async def run():
            yield "Hel"
            yield {"tool": EVENT}
            yield "lo"

        return run()

    monkeypatch.setattr(tool_loop_service, "tool_stream_or_none", fake_tool_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        events = [
            event
            async for event in reply_generation_service.stream_reply(
                db, conversation, [{"role": "user", "content": "hi"}], []
            )
        ]
    await _drain_background_tasks()

    assert [e["chunk"] for e in events if "chunk" in e] == ["Hel", "lo"]
    assert [e["tool"] for e in events if "tool" in e] == [EVENT]
    assert events[-1].get("done") is True
    assert "request_id" in seen_params[0]
    async with session_factory() as db:
        message = (await db.execute(select(Message).where(Message.role == "assistant"))).scalar_one()
    assert (message.content, message.status, message.tool_events) == ("Hello", "complete", [EVENT])
    await engine.dispose()


@pytest.mark.asyncio
async def test_a_reply_that_used_tools_but_wrote_nothing_is_marked_as_an_error(monkeypatch):
    engine, session_factory = await _build_session_factory()
    for module in (reply_generation_service, reply_termination_service, reply_cross_instance_service):
        monkeypatch.setattr(module, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_tool_stream(_db, _model, _messages, _params):
        async def run():
            yield {"tool": EVENT}

        return run()

    monkeypatch.setattr(tool_loop_service, "tool_stream_or_none", fake_tool_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        async for _ in reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], []):
            pass
    await _drain_background_tasks()

    async with session_factory() as db:
        message = (await db.execute(select(Message).where(Message.role == "assistant"))).scalars().one()
    assert message.status == "error" and message.error_message == tool_event_service.NO_ANSWER
    assert message.tool_events == [EVENT]
