"""Tests for services/reply_generation_service.py: the detached-generation mechanism every conversation's reply
uses so a mid-reply disconnect can't lose it (see the module), its batched persistence and mid-stream Ollama
failure handling. It's conversation-type-agnostic (personal and channel chats share the code), so tests don't
distinguish them; tests/test_chat_service.py covers the layered personal/channel behavior (titles, RAG scope).

Builds its own engine/session_factory per test (like tests/test_title_service.py's background-task tests), not the
shared `db` fixture: conftest.py's `db` points reply_generation_service.AsyncSessionLocal at its own engine, but
these tests need an *isolated*, separately inspectable database to simulate a session outliving, or disconnected
from, the one that started generation.
"""

import asyncio
import os
import tempfile

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.database import Base
from app.models import Conversation, Message
from app.services import (
    chat_settings_service,
    reply_cross_instance_service,
    reply_generation_service,
    reply_model_unload_service,
)
from app.services.inference_client import InferenceError


async def _build_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _build_file_session_factory():
    """Like _build_session_factory but a real temp file, not ":memory:": needed by tests keeping two sessions
    concurrently "in flight" (a test's commit racing a background task's write to one row). SQLAlchemy shares
    one connection across every ":memory:" session in a process, so genuinely concurrent sessions fight over it, unlike
    a file database where each gets its own connection (as production's AsyncSessionLocal). tmp_path isn't used
    since this module avoids the shared `db` fixture; the caller cleans up after disposing the engine.
    """
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False), path


async def _make_conversation(session_factory) -> str:
    async with session_factory() as db:
        conversation = Conversation(owner_id=None, title="general", model="fake-model", params={})
        db.add(conversation)
        await db.commit()
        return conversation.id


async def _drain_background_tasks() -> None:
    """Waits for every tracked background task, including one a cancelled task spawns to persist its final state
    (reply_model_unload_service.schedule_cancelled_write, tracked in a *separate* set) that doesn't exist when the
    original is cancelled, so awaiting the known tasks isn't enough.
    """
    for _ in range(200):
        tasks = list(reply_generation_service._background_generation_tasks) + list(
            reply_model_unload_service._background_tasks
        )
        if not tasks:
            return
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.sleep(0)
    raise AssertionError("background tasks never drained")


@pytest.mark.asyncio
async def test_stream_reply_creates_a_streaming_placeholder_before_any_chunk(monkeypatch):
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    started = asyncio.Event()
    release = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        started.set()
        await release.wait()
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        anext_task = asyncio.create_task(gen.__anext__())
        await started.wait()

        async with session_factory() as peek_db:
            placeholder = (
                await peek_db.execute(select(Message).where(Message.conversation_id == conversation_id))
            ).scalar_one()
            assert placeholder.status == "streaming"
            assert placeholder.content == ""

        release.set()
        first_event = await anext_task
        assert first_event["chunk"] == "ok"
        await gen.aclose()

    await asyncio.gather(*reply_generation_service._background_generation_tasks)
    await engine.dispose()


@pytest.mark.asyncio
async def test_stream_reply_stores_reply_to_message_id_on_the_placeholder(monkeypatch):
    """conversation_service.delete_message uses this to find and cancel a streaming reply when its triggering user
    message is deleted (Message.reply_to_message_id: stored explicitly, not inferred from timestamps).
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        question = Message(conversation_id=conversation_id, role="user", content="hi")
        db.add(question)
        await db.commit()

        gen = reply_generation_service.stream_reply(
            db, conversation, [{"role": "user", "content": "hi"}], [], reply_to_message_id=question.id
        )
        first_event = await gen.__anext__()
        placeholder = await db.get(Message, first_event["message_id"])
        assert placeholder.reply_to_message_id == question.id
        await gen.aclose()

    await asyncio.gather(*reply_generation_service._background_generation_tasks)
    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_survives_the_consumer_disconnecting_early_by_default(monkeypatch):
    """The core regression for a channel's shared conversation (cancel_on_disconnect defaults to False): the SSE
    generator stopping early (what ASGI does when a client navigates away; routers/chat.py) must not stop
    generation, as other members may rely on it. The personal-chat opposite:
    test_cancel_on_disconnect_stops_generation_and_marks_it_errored.
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_chat_stream(_model, _messages, _params):
        for piece in ["Hello", ", ", "world", "!"]:
            await asyncio.sleep(0)
            yield piece

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulate the client disconnecting right after the first chunk (ASGI closes the generator like this).
        await gen.aclose()

    # The generation task keeps running independently of the closed generator.
    await asyncio.gather(*reply_generation_service._background_generation_tasks)

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "complete"
        assert message.content == "Hello, world!"
        # Shown as the reply's time: when it finished, not when it started.
        assert message.completed_at is not None and message.completed_at >= message.created_at

    await engine.dispose()


@pytest.mark.asyncio
async def test_cancel_on_disconnect_stops_generation_and_marks_it_errored(monkeypatch):
    """The opposite, for a personal chat (chat_service.build_reply_stream passes cancel_on_disconnect=True): a
    disconnect must stop generation, not just watching, since only the owner could be waiting. It still leaves a
    status="error" record with what was generated, not an orphaned "streaming" row or (as before this module)
    a silently lost attempt.

    Uses _build_file_session_factory (not:memory:; see its docstring): this keeps two sessions genuinely
    concurrent (the test's open session and the cancelled task's separate write), which:memory:'s single shared
    connection can't represent.
    """
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    # Blocks forever past the first chunk: proves generation was interrupted, not just slow.
    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    # A cancelled reply also force-stops the model (ollama_client.stop_model), faked to avoid a real Ollama host.
    # reply_model_unload_service calls it (schedule_cancelled_write/_persist_cancelled_reply), not
    # reply_generation_service.
    stopped_models: list[str] = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)

    monkeypatch.setattr(reply_model_unload_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(
            db, conversation, [{"role": "user", "content": "hi"}], [], cancel_on_disconnect=True
        )
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulate a personal chat's owner leaving mid-reply.
        await gen.aclose()

    # Wait for the cancelled task and the fresh task it spawns to persist the cancellation
    # (reply_generation_service._schedule_cancelled_write: a cancelled task can't await its own cleanup).
    await _drain_background_tasks()

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
        assert message.error_message == "Cancelled: left the chat before this reply finished."

    assert stopped_models == ["fake-model"]

    await engine.dispose()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_cancel_generation_stops_an_in_progress_reply_and_marks_it_errored(monkeypatch):
    """conversation_service.delete_message's mechanism for stopping a streaming reply it's removing. Unlike
    cancel_on_disconnect (triggered only by the *sender's* request disconnecting), anyone can trigger this at any
    time while the sender's connection stays open.

    Uses _build_file_session_factory (not:memory:; see its docstring): two sessions are genuinely concurrent (the
    test's and the cancelled task's separate write), which:memory:'s shared connection can't represent.
    """
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    stopped_models: list[str] = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)

    monkeypatch.setattr(reply_model_unload_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # An admin deleting the message elsewhere: the sender's connection (this generator) stays open, unlike
        # cancel_on_disconnect's test.
        assert reply_generation_service.cancel_generation(message_id) is True
        await gen.aclose()

    await _drain_background_tasks()

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
        assert message.error_message == "Cancelled: left the chat before this reply finished."

    assert stopped_models == ["fake-model"]

    await engine.dispose()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_cancel_generation_returns_false_when_theres_nothing_to_cancel():
    assert reply_generation_service.cancel_generation("does-not-exist") is False


@pytest.mark.asyncio
async def test_cancel_generation_never_overwrites_a_message_already_marked_deleted(monkeypatch):
    """conversation_service.delete_message calls mark_message_deleted (synchronously, before cancel_generation) and
    sets status="deleted" without waiting for cancel_generation to land, so the separately scheduled cancelled-write
    (_persist_cancelled_reply) must see that and skip, not clobber it to status="error" later. This closes the race
    a "read DB status, then write" guard couldn't (reply_cancellation_service.mark_message_deleted).

    Uses _build_file_session_factory (not:memory:; see its docstring): two sessions are genuinely concurrent (the
    test's commit and the cancelled task's separate write), which:memory:'s shared connection can't represent.
    """
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulates delete_message's sequence: mark_message_deleted, *then* cancel_generation, then the status="deleted"
        # write, with no `await` between the first two, well before the cancelled task's scheduled write could land.
        message = await db.get(Message, message_id)
        reply_generation_service.mark_message_deleted(message_id)
        reply_generation_service.cancel_generation(message_id)
        message.status = "deleted"
        message.content = ""
        await db.commit()

        await gen.aclose()

    await _drain_background_tasks()

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "deleted"
        assert message.content == ""

    await engine.dispose()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_cross_instance_deletion_detected_mid_stream_also_stops_the_model(monkeypatch):
    """A delete_message served by a *different* sibling instance (reply_cross_instance_service) never reaches this
    instance's in-process reply_cancellation_service registry, so cancel_generation isn't called here; only this
    loop's periodic refresh_or_none check notices. That check once just `return`ed on status="deleted", abandoning
    the open chat_stream generator without telling the engine to stop (a bare `return`/`break` doesn't
    deterministically close an async generator): the "" gap the timeout branch guards with stop_model.

    Uses _build_file_session_factory (not:memory:; see its docstring): two sessions are genuinely concurrent (the
    "other instance"'s delete commit and the generation task's session), which:memory:'s shared connection can't
    represent.
    """
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    proceed = asyncio.Event()
    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        await proceed.wait()
        yield "x" * 300
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    stopped_models: list[str] = []
    stopped_request_ids: list[str | None] = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)
        stopped_request_ids.append(request_id)

    monkeypatch.setattr(reply_generation_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulates delete_message served by a *different* sibling instance: only the shared DB sees it; no
        # mark_message_deleted/cancel_generation reaches this instance's in-process registries.
        other_instance_db = session_factory()
        message = await other_instance_db.get(Message, message_id)
        message.status = "deleted"
        message.content = ""
        await other_instance_db.commit()
        await other_instance_db.close()

        proceed.set()
        await gen.aclose()

    await _drain_background_tasks()

    assert stopped_models == ["fake-model"]
    # Stops only this reply (its message id), never the whole model for other users.
    assert stopped_request_ids == [message_id]

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "deleted"

    await engine.dispose()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_generation_batches_db_writes_rather_than_committing_every_chunk(monkeypatch):
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    release_second_chunk = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "short"
        await release_second_chunk.wait()
        yield " more"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Published to the hub (what the sender's tab sees) but far below the size or time flush thresholds: must not be
        # committed to the DB.
        async with session_factory() as peek_db:
            message = await peek_db.get(Message, message_id)
            assert message.content == ""
            assert message.status == "streaming"

        release_second_chunk.set()
        second_event = await gen.__anext__()
        assert second_event["chunk"] == " more"
        done_event = await gen.__anext__()
        assert done_event["done"] is True

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.content == "short more"
        assert message.status == "complete"

    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_error_preserves_partial_content_and_marks_status_error(monkeypatch):
    """Better than the old inline personal-chat behavior (before this module unified both paths), which discarded the
    whole reply on a mid-stream Ollama failure: what was generated before the error is kept."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial reply"
        raise InferenceError("model crashed")

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        events = [
            event
            async for event in reply_generation_service.stream_reply(
                db, conversation, [{"role": "user", "content": "hi"}], []
            )
        ]

    assert events[-1]["error"] == "model crashed"
    message_id = events[0]["message_id"]

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial reply"
        # The InferenceError's text used to reach only the live SSE event, never persisted, so reopening the
        # conversation
        # showed chat.js's generic "didn't finish" notice with no cause (Message.error_message).
        assert message.error_message == "model crashed"

    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_times_out_and_marks_it_errored(monkeypatch):
    """The admin reply timeout (chat_settings_service.get_reply_timeout_seconds) is a wall-clock deadline over the whole
    generation, not a per-chunk read timeout: this model keeps producing something, just too slowly overall, which
    ollama_client's lower-level timeout wouldn't catch.
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_get_timeout(_db):
        return 0.05

    monkeypatch.setattr(chat_settings_service, "get_reply_timeout_seconds", fake_get_timeout)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    # A timed-out reply also force-stops the model (ollama_client.stop_model): closing our side doesn't reliably
    # interrupt an in-progress llama.cpp compute phase. Faked to avoid a real Ollama host.
    stopped_models: list[str] = []
    stopped_request_ids: list[str | None] = []

    async def fake_stop_model(model, request_id=None):
        stopped_models.append(model)
        stopped_request_ids.append(request_id)

    monkeypatch.setattr(reply_generation_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        events = [
            event
            async for event in reply_generation_service.stream_reply(
                db, conversation, [{"role": "user", "content": "hi"}], []
            )
        ]

    assert "timed out" in events[-1]["error"]
    message_id = events[0]["message_id"]
    # Only this reply is stopped (by its message id), never the whole model for other users.
    assert stopped_request_ids == [message_id]

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
        assert message.error_message == events[-1]["error"]

    assert stopped_models == ["fake-model"]

    await engine.dispose()


@pytest.mark.asyncio
async def test_has_image_uses_the_separate_vision_timeout_setting(monkeypatch):
    """has_image=True must read get_vision_reply_timeout_seconds, not get_reply_timeout_seconds: the *text* timeout is
    generous and the *vision* one tiny, so only a reply consulting the vision setting times out.
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_get_reply_timeout(_db):
        return 3600

    async def fake_get_vision_timeout(_db):
        return 0.05

    monkeypatch.setattr(chat_settings_service, "get_reply_timeout_seconds", fake_get_reply_timeout)
    monkeypatch.setattr(chat_settings_service, "get_vision_reply_timeout_seconds", fake_get_vision_timeout)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(reply_generation_service, "stop_model", lambda _model, request_id=None: asyncio.sleep(0))

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        events = [
            event
            async for event in reply_generation_service.stream_reply(
                db, conversation, [{"role": "user", "content": "hi"}], [], has_image=True
            )
        ]

    assert "timed out" in events[-1]["error"]
    message_id = events[0]["message_id"]

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"

    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_is_not_cut_off_when_timeout_is_disabled(monkeypatch):
    """0 means no timeout (chat_settings_service.DEFAULT_REPLY_TIMEOUT_SECONDS says why it's intentional):
    asyncio.timeout(None) applies no deadline.
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_model_unload_service (a cancelled reply's write, split out of reply_generation_service) imports
    # AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_model_unload_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_get_timeout(_db):
        return 0

    monkeypatch.setattr(chat_settings_service, "get_reply_timeout_seconds", fake_get_timeout)

    async def fake_chat_stream(_model, _messages, _params):
        await asyncio.sleep(0.05)
        yield "slow but fine"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        events = [
            event
            async for event in reply_generation_service.stream_reply(
                db, conversation, [{"role": "user", "content": "hi"}], []
            )
        ]

    assert events[-1]["done"] is True
    message_id = events[0]["message_id"]

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "complete"
        assert message.content == "slow but fine"

    await engine.dispose()
