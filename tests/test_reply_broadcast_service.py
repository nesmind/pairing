"""Tests for services/reply_broadcast_service.py, the in-process pub/sub hub read by
reply_generation_service's relay and routers/chat.py's `/subscribe`.
"""

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
    """Regression for a shipped bug: routers/chat.py's /subscribe relays this hub's raw events unmodified (unlike
    stream_reply's relay for the sender's tab, which stamps "message_id"), so a channel member other than the sender,
    watching live in "real" delivery mode, got every chunk/done/error without a message_id, and chat.js's
    handleReplyLiveEvent needs one: they never saw a live reply appear, finish or fail until re-entering the
    channel. Only this dict carrying its own "message_id" fixes it.
    """
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

    # Subscribing *after* both chunks: a late joiner sees the accumulated content via the snapshot, not the chunk
    # events it missed (delivered only to subscribers present at publish time).
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
    """Regression for a shipped bug: a member opening the chat before anyone has sent a message (no topic yet; see
    test_subscribe_before_any_topic_returns_no_snapshot) must still receive the *next* message's events, not be dropped
    because their queue predates the topic.
    """
    state, queue = reply_broadcast_service.subscribe("conv-7")
    assert state is None

    reply_broadcast_service.start_topic("conv-7", "msg-7")
    reply_broadcast_service.publish_chunk("conv-7", "hello")

    event = await asyncio.wait_for(queue.get(), timeout=1)
    assert event == {"chunk": "hello", "message_id": "msg-7"}


@pytest.mark.asyncio
async def test_subscriber_survives_across_multiple_messages_in_the_same_conversation():
    """Same bug, the other way it appeared: a /subscribe connection is opened once per chat visit (chat.js
    startReplyWatch), not per message, so it must keep receiving events for every message sent while they stay. An
    earlier start_topic replaced the whole _Topic (subscribers included) per message, orphaning connected subscribers
    after the first.
    """
    reply_broadcast_service.start_topic("conv-8", "msg-8a")
    _state, queue = reply_broadcast_service.subscribe("conv-8")

    reply_broadcast_service.publish_done("conv-8", None)
    first = await asyncio.wait_for(queue.get(), timeout=1)
    assert first == {"done": True, "sources": None, "message_id": "msg-8a"}

    # A second message in the same conversation, the same subscriber still connected (chat.js never reopens the
    # connection): it must see this one too.
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
    # The snapshot reset to the new message with empty content, not still carrying "first".
    assert state.message_id == "msg-9b"
    assert state.content == ""
    assert state.status == "streaming"
