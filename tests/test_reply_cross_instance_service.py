"""Unit tests for app/services/reply_cross_instance_service.py — the
database-fallback helpers reply_generation_service relies on so a
delete_message call served by a *different* sibling instance (see
app.services.instance_pool) isn't completely invisible to a reply's own
generation task. See that module's own docstring for the full picture;
these tests cover each helper in isolation."""

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import Conversation, Message
from app.schemas import ConversationCreate
from app.services import conversation_service, reply_cross_instance_service


@pytest.mark.asyncio
async def test_refresh_or_none_returns_the_message_with_current_status(db, user):
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add(message)
    await db.commit()

    # Simulates a *different* session/process changing the row.
    message.status = "deleted"
    await db.commit()

    fresh = await reply_cross_instance_service.refresh_or_none(db, message)
    assert fresh is not None
    assert fresh.status == "deleted"


@pytest.mark.asyncio
async def test_refresh_or_none_returns_none_for_a_hard_deleted_row(db, user):
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add(message)
    await db.commit()

    # Performed through a *separate* session, matching how this actually
    # happens in production (refresh_or_none is always called from a
    # session that never touched the delete itself — see
    # reply_cross_instance_service's own docstring). Reusing `db` for
    # both the delete and the refresh_or_none call below would detach
    # `message` from `db`'s own identity map once the cascade-delete
    # commits, which raises a different (and here, irrelevant)
    # InvalidRequestError instead of exercising the real "row is gone"
    # path this test is actually after.
    other_session_factory = async_sessionmaker(db.bind, expire_on_commit=False)
    async with other_session_factory() as other_db:
        other_conversation = await other_db.get(Conversation, conversation.id)
        # conversation.messages (lazy="selectin") was loaded empty when
        # `conversation` was first committed above, before `message` was
        # added directly via db.add() — the ORM's own cascade delete
        # below only touches what's in that *in-memory* collection, so
        # without this refresh it wouldn't know to delete `message` at
        # all.
        await other_db.refresh(other_conversation, attribute_names=["messages"])
        await other_db.delete(other_conversation)  # cascades to the message row too
        await other_db.commit()

    assert await reply_cross_instance_service.refresh_or_none(db, message) is None


def test_terminal_event_for_deleted_and_none_both_synthesize_a_deleted_event():
    deleted = Message(role="assistant", status="deleted")
    assert reply_cross_instance_service.terminal_event_for(deleted, "msg-1") == {
        "deleted": True,
        "message_id": "msg-1",
    }
    assert reply_cross_instance_service.terminal_event_for(None, "msg-2") == {
        "deleted": True,
        "message_id": "msg-2",
    }


def test_terminal_event_for_error_and_complete():
    errored = Message(role="assistant", status="error")
    assert reply_cross_instance_service.terminal_event_for(errored, "msg-3") == {
        "error": "Reply generation failed.",
        "message_id": "msg-3",
    }
    completed = Message(role="assistant", status="complete", sources=["a.txt"])
    assert reply_cross_instance_service.terminal_event_for(completed, "msg-4") == {
        "done": True,
        "sources": ["a.txt"],
        "message_id": "msg-4",
    }


@pytest.mark.asyncio
async def test_watch_for_deletion_cancels_the_task_once_the_row_is_marked_deleted(db, user):
    """The core regression case: a reply's own generation task is stuck
    (e.g. a slow cold model load, no chunks yet — reply_generation_service's
    own flush-point check never even runs) when an admin deletes it —
    the watchdog must notice via the database directly and cancel the
    task, regardless of which instance actually served the delete."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add(message)
    await db.commit()

    async def stuck_forever():
        await asyncio.Event().wait()

    task = asyncio.create_task(stuck_forever())
    watchdog = asyncio.create_task(
        reply_cross_instance_service.watch_for_deletion(message.id, task, poll_interval=0.05)
    )

    await asyncio.sleep(0.1)  # let the watchdog get at least one poll in
    message.status = "deleted"
    await db.commit()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    await asyncio.wait_for(watchdog, timeout=2)
    assert task.cancelled()


@pytest.mark.asyncio
async def test_watch_for_deletion_leaves_a_normally_finishing_task_alone(db, user):
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add(message)
    await db.commit()

    async def finishes_quickly():
        await asyncio.sleep(0.05)
        return "done"

    task = asyncio.create_task(finishes_quickly())
    watchdog = asyncio.create_task(reply_cross_instance_service.watch_for_deletion(message.id, task, poll_interval=0.5))

    result = await asyncio.wait_for(task, timeout=2)
    assert result == "done"
    await asyncio.wait_for(watchdog, timeout=2)
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_watch_for_deletion_cancels_when_the_row_is_gone_entirely(db, user):
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming")
    db.add(message)
    await db.commit()
    message_id = message.id
    # See test_refresh_or_none_returns_none_for_a_hard_deleted_row's own
    # comment on why this is needed before the cascade delete below.
    await db.refresh(conversation, attribute_names=["messages"])

    async def stuck_forever():
        await asyncio.Event().wait()

    task = asyncio.create_task(stuck_forever())
    watchdog = asyncio.create_task(
        reply_cross_instance_service.watch_for_deletion(message_id, task, poll_interval=0.05)
    )

    await asyncio.sleep(0.1)
    await db.delete(conversation)  # cascades: the message row disappears outright
    await db.commit()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=2)
    await asyncio.wait_for(watchdog, timeout=2)
    assert task.cancelled()
