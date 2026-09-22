"""Unit tests for app/services/reply_generation_service.py: the
detached-generation mechanism every conversation's reply relies on so a
disconnect mid-reply can't lose it (see that module's own docstring),
its batched persistence, and its handling of a mid-stream Ollama
failure. The mechanism itself is conversation-type-agnostic (a personal
chat and a channel's shared conversation go through the exact same
code), so these tests don't distinguish between the two — see
tests/test_chat_service.py for the personal/channel-specific behavior
layered on top (title handling, RAG scope).

Builds its own engine/session_factory per test (like
tests/test_title_service.py's background-task tests) rather than using
the shared `db` fixture: while conftest.py's `db` fixture does point
reply_generation_service.AsyncSessionLocal at its own engine (for tests
that just want a background task's writes visible through that same
session), these tests specifically need an *isolated*, independently
inspectable database to simulate a session outliving — or being
disconnected from — the one that started generation.
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
    reply_termination_service,
)
from app.services.inference_client import InferenceError


async def _build_session_factory():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def _build_file_session_factory():
    """Same as _build_session_factory, but a real temp file rather than
    ":memory:" — needed only by tests that deliberately keep two sessions
    concurrently "in flight" against each other (a test's own commit
    racing a background task's own write to the same row): SQLAlchemy
    transparently shares one single underlying connection across every
    ":memory:" session in a process (there's no other way for them to see
    the same data at all), so two sessions that are genuinely concurrent
    fight over that one connection in a way a real file-backed database
    — where every session gets its own separate connection, exactly like
    production's real AsyncSessionLocal — never does. A temp file across
    conftest.py's tmp_path fixture isn't used here since this whole
    module deliberately avoids the shared `db` fixture (see this file's
    own docstring) — cleaned up by the caller once its engine is disposed."""
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
    """Waits for every currently-tracked background task to finish,
    including one a cancelled task spawns to persist its own final
    state (see reply_termination_service.schedule_cancelled_write, in a
    *separate* module's own tracking set from the main generation
    task's) — which wouldn't exist yet at the moment the *original* task
    is cancelled, so a plain "await the tasks I already know about"
    isn't enough."""
    for _ in range(200):
        tasks = list(reply_generation_service._background_generation_tasks) + list(
            reply_termination_service._background_tasks
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
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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
    """conversation_service.delete_message relies on this to find (and
    cancel) a still-streaming reply when its triggering user message is
    deleted — see Message.reply_to_message_id's own docstring for why
    this is stored explicitly rather than inferred from timestamps."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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
    """The core regression test for a channel's shared conversation
    (cancel_on_disconnect defaults to False): the SSE request generator
    stopping early (exactly what ASGI does when a client navigates away
    mid-reply — see app/routers/chat.py) must not stop generation itself
    — other members may still be relying on it. See
    test_cancel_on_disconnect_stops_generation_and_marks_it_errored below
    for the opposite, personal-chat case.
    """
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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

        # Simulate the client disconnecting right after the first chunk
        # reached it — ASGI closes this generator exactly like this.
        await gen.aclose()

    # The generation task keeps running entirely independently of the
    # now-closed generator above.
    await asyncio.gather(*reply_generation_service._background_generation_tasks)

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "complete"
        assert message.content == "Hello, world!"

    await engine.dispose()


@pytest.mark.asyncio
async def test_cancel_on_disconnect_stops_generation_and_marks_it_errored(monkeypatch):
    """The opposite case, for a personal chat (see
    app.services.chat_service.build_reply_stream, which passes
    cancel_on_disconnect=True there): a disconnect must actually stop
    generation, not just stop watching it, since only the chat's owner
    could ever have been the one waiting on it — but it still leaves a
    clear status="error" record with whatever was generated before that,
    rather than an orphaned "streaming" row or, worse, silently losing
    the attempt the way a personal chat used to before this module
    existed.

    Uses _build_file_session_factory (not :memory:) — see its own
    docstring: this test keeps two sessions genuinely concurrent (this
    test's own open session, and the cancelled task's own separately-
    scheduled write), which :memory:'s single shared connection can't
    represent faithfully."""
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    # Blocks forever once past the first chunk — proves generation was
    # actually interrupted, not just coincidentally slow to finish.
    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    # A cancelled reply also force-stops the model outright (see
    # ollama_client.stop_model's own docstring) — faked here rather than
    # hitting a real Ollama host from a unit test. reply_termination_service
    # is the one that actually calls it (see schedule_cancelled_write/
    # _persist_cancelled_reply), not reply_generation_service directly.
    stopped_models: list[str] = []

    async def fake_stop_model(model):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(
            db, conversation, [{"role": "user", "content": "hi"}], [], cancel_on_disconnect=True
        )
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulate a personal chat's owner leaving mid-reply.
        await gen.aclose()

    # Wait for the cancelled task, and the fresh task it spawns to
    # actually persist that cancellation (see
    # reply_generation_service._schedule_cancelled_write — a cancelled
    # task can't safely await its own cleanup, so a separate task does
    # the write instead), to both finish.
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
    """conversation_service.delete_message's own mechanism for stopping a
    still-streaming reply it's about to remove — unlike
    cancel_on_disconnect above (only ever triggered by the *sender's own*
    request disconnecting), this can be triggered by anyone, at any time,
    while the sender's own connection stays open throughout.

    Uses _build_file_session_factory (not :memory:) — see its own
    docstring: this test keeps two sessions genuinely concurrent (this
    test's own open session, and the cancelled task's own separately-
    scheduled write), which :memory:'s single shared connection can't
    represent faithfully."""
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    stopped_models: list[str] = []

    async def fake_stop_model(model):
        stopped_models.append(model)

    monkeypatch.setattr(reply_termination_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # An admin deleting the message, elsewhere — the sender's own
        # connection (this generator) is left open throughout, unlike
        # cancel_on_disconnect's own test above.
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
    """conversation_service.delete_message calls mark_message_deleted
    (synchronously, before cancel_generation) and sets status="deleted"
    itself, without waiting for cancel_generation's own effect to land —
    so the separately-scheduled cancelled-write (see
    _persist_cancelled_reply) must recognize that and skip, rather than
    clobbering the deletion back to status="error" once it eventually
    runs. This is what actually closes the race a plain "read the DB
    status, then write" guard couldn't: see
    reply_cancellation_service.mark_message_deleted's own docstring.

    Uses _build_file_session_factory (not the usual :memory: one — see
    its own docstring): this test deliberately keeps two sessions
    genuinely concurrent (this test's own commit below, and the
    cancelled task's own separately-scheduled write), which :memory:'s
    single shared connection can't represent faithfully."""
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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

        # Simulates conversation_service.delete_message's own sequence:
        # mark_message_deleted, *then* cancel_generation, then the
        # status="deleted" write — all synchronously, no `await` between
        # the first two, well before the cancelled task's own
        # separately-scheduled write could ever land.
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
    """A delete_message call served by a *different* sibling instance (see
    reply_cross_instance_service's own docstring) never reaches this
    instance's own in-process reply_cancellation_service registry, so
    cancel_generation is never called here — the only thing that notices
    at all is this loop's own periodic refresh_or_none check below. Before
    this fix, that check just `return`ed once it saw status="deleted",
    abandoning the still-open chat_stream generator without ever telling
    the engine to actually stop (an async generator exited via a bare
    `return`/`break` isn't deterministically closed) — same "confirmed
    live" gap the timeout branch already guards against with its own
    stop_model call, just missing here.

    Uses _build_file_session_factory (not the usual :memory: one — see its
    own docstring): this test deliberately keeps two sessions genuinely
    concurrent (the "other instance"'s own delete commit below, and the
    generation task's own separate session), which :memory:'s single
    shared connection can't represent faithfully."""
    engine, session_factory, db_path = await _build_file_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    proceed = asyncio.Event()
    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        await proceed.wait()
        yield "x" * 300  # over _FLUSH_CHARS, forces the mid-loop cross-instance check below
        await never.wait()  # would hang forever (still "generating") if stop_model didn't fire
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    stopped_models: list[str] = []

    async def fake_stop_model(model):
        stopped_models.append(model)

    monkeypatch.setattr(reply_generation_service, "stop_model", fake_stop_model)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Simulates delete_message served by a *different* sibling instance: only the shared
        # database sees this — no mark_message_deleted/cancel_generation call ever reaches this
        # instance's own in-process registries.
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

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "deleted"

    await engine.dispose()
    os.unlink(db_path)


@pytest.mark.asyncio
async def test_generation_batches_db_writes_rather_than_committing_every_chunk(monkeypatch):
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    release_second_chunk = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "short"  # well under the character flush threshold
        await release_second_chunk.wait()
        yield " more"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    async with session_factory() as db:
        conversation = await db.get(Conversation, conversation_id)
        gen = reply_generation_service.stream_reply(db, conversation, [{"role": "user", "content": "hi"}], [])
        first_event = await gen.__anext__()
        message_id = first_event["message_id"]

        # Published to the broadcast hub (what the sender's own tab
        # sees) but nowhere near the size or time flush thresholds yet —
        # must not be committed to the database.
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
    """Strictly better than the old inline behavior a personal chat used
    to have (before this module unified both paths), which discarded the
    entire reply on any mid-stream Ollama failure — whatever was
    generated before the error must be kept."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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
        # The real bug this covers: the InferenceError's own text used to only ever reach the live SSE event
        # above — never persisted — so reopening this conversation later showed chat.js's generic "didn't
        # finish" notice with no way to learn what actually happened (see Message.error_message's own docstring).
        assert message.error_message == "model crashed"

    await engine.dispose()


@pytest.mark.asyncio
async def test_generation_times_out_and_marks_it_errored(monkeypatch):
    """The admin-configured reply timeout (see
    app.services.chat_settings_service.get_reply_timeout_seconds) is a
    wall-clock deadline over the whole generation, not just a per-chunk
    read timeout — this model keeps producing *something*, just too
    slowly overall, which app.services.ollama_client's own lower-level
    timeout wouldn't catch on its own."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_get_timeout(_db):
        return 0.05

    monkeypatch.setattr(chat_settings_service, "get_reply_timeout_seconds", fake_get_timeout)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()  # would hang forever if the timeout didn't fire
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    # A timed-out reply also force-stops the model outright (see
    # ollama_client.stop_model's own docstring) — closing our side of the
    # connection doesn't reliably interrupt an in-progress llama.cpp
    # compute phase on its own. Faked here rather than hitting a real
    # Ollama host from a unit test.
    stopped_models: list[str] = []

    async def fake_stop_model(model):
        stopped_models.append(model)

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

    async with session_factory() as verify_db:
        message = await verify_db.get(Message, message_id)
        assert message.status == "error"
        assert message.content == "partial"
        assert message.error_message == events[-1]["error"]

    assert stopped_models == ["fake-model"]

    await engine.dispose()


@pytest.mark.asyncio
async def test_has_image_uses_the_separate_vision_timeout_setting(monkeypatch):
    """has_image=True must read get_vision_reply_timeout_seconds instead of get_reply_timeout_seconds — proven
    here by setting the *text* timeout to something generous and the *vision* one to something tiny, so only a
    reply that actually consulted the vision setting would time out."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
    monkeypatch.setattr(reply_cross_instance_service, "AsyncSessionLocal", session_factory)
    conversation_id = await _make_conversation(session_factory)

    async def fake_get_reply_timeout(_db):
        return 3600  # would never fire in this test if this were the one actually used

    async def fake_get_vision_timeout(_db):
        return 0.05

    monkeypatch.setattr(chat_settings_service, "get_reply_timeout_seconds", fake_get_reply_timeout)
    monkeypatch.setattr(chat_settings_service, "get_vision_reply_timeout_seconds", fake_get_vision_timeout)

    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()  # would hang forever if the vision timeout didn't fire
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(reply_generation_service, "stop_model", lambda _model: asyncio.sleep(0))

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
    """0 means no timeout (see
    chat_settings_service.DEFAULT_REPLY_TIMEOUT_SECONDS's own docstring
    for why that's a real, intentional option) — asyncio.timeout(None)
    applies no deadline at all."""
    engine, session_factory = await _build_session_factory()
    monkeypatch.setattr(reply_generation_service, "AsyncSessionLocal", session_factory)
    # reply_termination_service (a cancelled reply's own write — split
    # out of reply_generation_service, see that module's own docstring)
    # imports AsyncSessionLocal independently, so it needs this too.
    monkeypatch.setattr(reply_termination_service, "AsyncSessionLocal", session_factory)
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
