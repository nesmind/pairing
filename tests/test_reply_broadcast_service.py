"""Unit tests for app/services/reply_broadcast_service.py — the
in-process pub/sub hub app.services.reply_generation_service's relay and
app/routers/chat.py's `/subscribe` endpoint both read from."""

import asyncio

import pytest

from app.services import reply_broadcast_service


def test_subscribe_before_any_topic_returns_no_snapshot():
    state, queue = reply_broadcast_service.subscribe("conv-1")
    assert state is None
    assert isinstance(queue, asyncio.Queue)


@pytest.mark.asyncio
async def test_publish_chunk_is_delivered_to_a_subscriber():
    reply_broadcast_service.start_topic("conv-1", "msg-1")
    _state, queue = reply_broadcast_service.subscribe("conv-1")

    reply_broadcast_service.publish_chunk("conv-1", "hello")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"chunk": "hello", "message_id": "msg-1"}


@pytest.mark.asyncio
async def test_publish_done_is_delivered_and_marks_the_topic_complete():
    reply_broadcast_service.start_topic("conv-2", "msg-2")
    _state, queue = reply_broadcast_service.subscribe("conv-2")

    reply_broadcast_service.publish_done("conv-2", ["a.txt"])

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"done": True, "sources": ["a.txt"], "message_id": "msg-2"}


@pytest.mark.asyncio
async def test_publish_error_is_delivered_and_marks_the_topic_errored():
    reply_broadcast_service.start_topic("conv-3", "msg-3")
    _state, queue = reply_broadcast_service.subscribe("conv-3")

    reply_broadcast_service.publish_error("conv-3", "boom")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"error": "boom", "message_id": "msg-3"}


@pytest.mark.asyncio
async def test_publish_chunk_done_and_error_all_carry_message_id_for_subscribe_watchers():
    """Regression test for a real, shipped bug: app/routers/chat.py's
    /subscribe endpoint relays this hub's raw events completely
    unmodified (unlike reply_generation_service.stream_reply's own relay
    for the sender's tab, which additionally stamps "message_id" onto
    whatever it yields) — so any channel member other than the sender,
    watching live in "real" delivery mode, saw every chunk/done/error
    event with no message_id at all, and chat.js's handleReplyLiveEvent
    requires one to do anything with an event. In practice, that member
    never saw a live reply appear (or finish, or fail) at all until they
    left the channel and came back — only the fix (this dict carrying its
    own "message_id", not relying on a relay to add it) closes that."""
    reply_broadcast_service.start_topic("conv-10", "msg-10")
    _state, queue = reply_broadcast_service.subscribe("conv-10")

    reply_broadcast_service.publish_chunk("conv-10", "hi")
    reply_broadcast_service.publish_done("conv-10", None)

    chunk_event = await asyncio.wait_for(queue.get(), timeout=1)
    done_event = await asyncio.wait_for(queue.get(), timeout=1)
    assert chunk_event["message_id"] == "msg-10"
    assert done_event["message_id"] == "msg-10"


@pytest.mark.asyncio
async def test_late_subscriber_syncs_against_the_current_snapshot_not_a_replay():
    reply_broadcast_service.start_topic("conv-4", "msg-4")
    reply_broadcast_service.publish_chunk("conv-4", "Hel")
    reply_broadcast_service.publish_chunk("conv-4", "lo")

    # Subscribing *after* both chunks were published: a late joiner must
    # see the accumulated content via the snapshot, not the individual
    # chunk events it missed (those were only delivered to subscribers
    # that existed at publish time).
    state, queue = reply_broadcast_service.subscribe("conv-4")
    assert state.message_id == "msg-4"
    assert state.content == "Hello"
    assert state.status == "streaming"
    assert queue.empty()


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery_without_affecting_other_subscribers():
    reply_broadcast_service.start_topic("conv-5", "msg-5")
    _state, queue_a = reply_broadcast_service.subscribe("conv-5")
    _state, queue_b = reply_broadcast_service.subscribe("conv-5")

    reply_broadcast_service.unsubscribe("conv-5", queue_a)
    reply_broadcast_service.publish_chunk("conv-5", "still here")

    assert queue_a.empty()
    event = await asyncio.wait_for(queue_b.get(), timeout=1)
    assert event == {"chunk": "still here", "message_id": "msg-5"}


@pytest.mark.asyncio
async def test_multiple_subscribers_all_receive_the_same_events():
    reply_broadcast_service.start_topic("conv-6", "msg-6")
    _state, queue_a = reply_broadcast_service.subscribe("conv-6")
    _state, queue_b = reply_broadcast_service.subscribe("conv-6")

    reply_broadcast_service.publish_chunk("conv-6", "fan-out")

    assert await asyncio.wait_for(queue_a.get(), timeout=1) == {"chunk": "fan-out", "message_id": "msg-6"}
    assert await asyncio.wait_for(queue_b.get(), timeout=1) == {"chunk": "fan-out", "message_id": "msg-6"}


@pytest.mark.asyncio
async def test_subscriber_connecting_before_any_generation_still_receives_the_next_one():
    """Regression test for a real, shipped bug: a channel member who
    opens the chat before anyone has ever sent a message there (no topic
    exists yet — see test_subscribe_before_any_topic_returns_no_snapshot)
    must still receive the *next* message's events once generation
    starts, not be silently dropped because their queue predates the
    topic that carries it."""
    state, queue = reply_broadcast_service.subscribe("conv-7")
    assert state is None  # nothing to sync against yet — correct

    reply_broadcast_service.start_topic("conv-7", "msg-7")
    reply_broadcast_service.publish_chunk("conv-7", "hello")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"chunk": "hello", "message_id": "msg-7"}


@pytest.mark.asyncio
async def test_subscriber_survives_across_multiple_messages_in_the_same_conversation():
    """Regression test for the same bug, the other way it showed up: a
    member's /subscribe connection is opened once per chat visit (see
    chat.js's startReplyWatch), not once per message — it must keep
    receiving events for every message sent while they're still there,
    not just the one that happened to be in flight when they connected.
    An earlier version of start_topic() replaced the whole _Topic object
    (subscribers included) on every new message, silently orphaning
    every already-connected subscriber after the first message."""
    reply_broadcast_service.start_topic("conv-8", "msg-8a")
    _state, queue = reply_broadcast_service.subscribe("conv-8")

    reply_broadcast_service.publish_done("conv-8", None)
    first = await asyncio.wait_for(queue.get(), timeout=1)
    assert first == {"done": True, "sources": None, "message_id": "msg-8a"}

    # A second message in the same conversation, same subscriber still
    # connected throughout (chat.js never re-opens the connection
    # between messages) — it must see this one too.
    reply_broadcast_service.start_topic("conv-8", "msg-8b")
    reply_broadcast_service.publish_chunk("conv-8", "second message")

    second = await asyncio.wait_for(queue.get(), timeout=1)
    assert second == {"chunk": "second message", "message_id": "msg-8b"}


@pytest.mark.asyncio
async def test_start_topic_resets_state_without_dropping_subscribers():
    reply_broadcast_service.start_topic("conv-9", "msg-9a")
    _state, queue = reply_broadcast_service.subscribe("conv-9")
    reply_broadcast_service.publish_chunk("conv-9", "first")
    await asyncio.wait_for(queue.get(), timeout=1)

    reply_broadcast_service.start_topic("conv-9", "msg-9b")
    state, _queue = reply_broadcast_service.subscribe("conv-9")
    # The snapshot reset to the new message, empty content again — not
    # still carrying "first" over from the previous one.
    assert state.message_id == "msg-9b"
    assert state.content == ""
    assert state.status == "streaming"
