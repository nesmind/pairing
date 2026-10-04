"""Assistant reply generation, detached from the initiating request via a
background asyncio.Task, for every conversation, personal or channel
alike. Split out of app.services.chat_service.build_reply_stream (which
still handles conversation-specific setup) to keep that file under
CLAUDE.md's file-size rule, and because this path's whole point is
running independently of whatever request triggered it.

Mechanism: stream_reply creates the placeholder assistant Message, hands
the ML engine call off to a detached asyncio.Task (_run_generation), then
becomes a pure relay on reply_broadcast_service's hub — it never itself
awaits the ML engine, so a disconnecting SSE request only cancels this
generator's `await queue.get()`, never the task directly. Whether the
task then keeps running or is cancelled in turn differs by conversation
type: `cancel_on_disconnect` (below) is False for a channel, True for a
personal chat (only its owner could ever be watching). Either way, a
failed/cancelled/timed-out attempt always leaves a clean status="error"
record, never silently vanishing. The same hub is also what a channel's
other `/subscribe` connections tap into for "real" delivery mode, and
what lets an owner reconnect fast enough to catch an in-flight reply's
tail — see reply_cross_instance_service for what happens when this hub's
own in-process nature isn't enough."""

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Conversation, Message
from app.services import (
    chat_settings_service,
    reply_broadcast_service,
    reply_cancellation_service,
    reply_cross_instance_service,
    reply_termination_service,
    tool_event_service,
    tool_loop_service,
)
from app.services.inference_client import InferenceError, chat_stream, stop_model

logger = logging.getLogger("llama_chat")

# Batches how often a streaming reply's content is committed to the database, rather than on every single token —
# neither a poller nor a late /subscribe join needs finer granularity. Every raw chunk still reaches the broadcast
# hub immediately, uncoupled from this write cadence, so a live viewer never sees the batching.
_FLUSH_CHARS = 200
_FLUSH_INTERVAL_SECONDS = 0.4

# How often stream_reply's relay also reads the database, not just the queue (see reply_cross_instance_service).
_CROSS_INSTANCE_POLL_SECONDS = 3.0

# Same reasoning as title_service._background_title_tasks: asyncio only holds a *weak* reference to a bare
# asyncio.create_task() result, so without this the task could be garbage-collected mid-run.
_background_generation_tasks: set[asyncio.Task] = set()

# delete_message's way of stopping a streaming reply — split out to reply_cancellation_service, re-exported here.
cancel_generation = reply_cancellation_service.cancel_generation
mark_message_deleted = reply_cancellation_service.mark_message_deleted


async def stream_reply(
    db: AsyncSession,
    conversation: Conversation,
    ollama_messages: list[dict],
    source_filenames: list[str],
    *,
    reply_to_message_id: str | None = None,
    cancel_on_disconnect: bool = False,
    model: str | None = None,
    has_image: bool = False,
) -> AsyncIterator[dict]:
    """Yields plain structured events for app/routers/chat.py to SSE-encode
    ({"chunk"} / {"error"} / {"done", "sources"} / {"deleted"} — see
    reply_broadcast_service.publish_deleted), each carrying "message_id" so
    the frontend knows which bubble it belongs to (see this module's own
    docstring for `cancel_on_disconnect`). `reply_to_message_id` is stored
    on the placeholder purely for delete_message's benefit
    (Message.reply_to_message_id) — never read back here. `model`, when
    given, overrides `conversation.model` for this call only (see
    chat_service.build_reply_stream's image handling). `has_image` picks
    which of chat_settings_service's two reply-timeout settings applies —
    see get_vision_reply_timeout_seconds' own docstring for why."""
    message = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="",
        status="streaming",
        reply_to_message_id=reply_to_message_id,
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)

    reply_broadcast_service.start_topic(conversation.id, message.id)
    effective_model = model or conversation.model
    task = _schedule_generation(
        message.id, effective_model, ollama_messages, conversation.params, source_filenames, has_image
    )

    _state, queue = reply_broadcast_service.subscribe(conversation.id)
    finished = False
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_CROSS_INSTANCE_POLL_SECONDS)
            except TimeoutError:
                # Cross-instance fallback (see reply_cross_instance_service) —
                # without this, this connection (the sender's own tab)
                # hangs forever if whatever it's waiting for happened on a
                # different sibling instance.
                fresh = await reply_cross_instance_service.refresh_or_none(db, message)
                if fresh is None or fresh.status != "streaming":
                    yield reply_cross_instance_service.terminal_event_for(fresh, message.id)
                    finished = True
                    return
                continue
            yield {**event, "message_id": message.id}
            if event.get("done") or event.get("error") or event.get("deleted"):
                finished = True
                return
    finally:
        reply_broadcast_service.unsubscribe(conversation.id, queue)
        if cancel_on_disconnect and not finished:
            task.cancel()


def _schedule_generation(
    message_id: str,
    model: str,
    ollama_messages: list[dict],
    params: dict,
    source_filenames: list[str],
    has_image: bool,
) -> asyncio.Task:
    """Kicks off generation fully detached from the current request — same strong-ref-set + done-callback pattern as
    title_service.schedule_smart_title_generation. Also registers with reply_cancellation_service so
    stream_reply/cancel_generation can cancel it in turn, and spawns reply_cross_instance_service's own watchdog
    alongside it (see that function's own docstring for why this is needed even with cancel_generation already in
    place)."""
    task = asyncio.create_task(_run_generation(message_id, model, ollama_messages, params, source_filenames, has_image))
    _background_generation_tasks.add(task)
    task.add_done_callback(_background_generation_tasks.discard)
    reply_cancellation_service.register_task(message_id, task)

    watchdog = asyncio.create_task(reply_cross_instance_service.watch_for_deletion(message_id, task))
    _background_generation_tasks.add(watchdog)
    watchdog.add_done_callback(_background_generation_tasks.discard)
    return task


async def _run_generation(
    message_id: str,
    model: str,
    ollama_messages: list[dict],
    params: dict,
    source_filenames: list[str],
    has_image: bool,
) -> None:
    """Runs against its own fresh AsyncSessionLocal() — the request's db
    session may already be gone by the time this finishes."""
    conversation_id = None
    full_reply = ""
    try:
        async with AsyncSessionLocal() as db:
            message = await db.get(Message, message_id)
            if message is None:
                return  # conversation (and this placeholder) deleted out from under this generation
            conversation_id = message.conversation_id
            # Read fresh (not cached) so a change applies next message. 0
            # means no timeout — asyncio.timeout(None) is "no deadline".
            get_timeout = (
                chat_settings_service.get_vision_reply_timeout_seconds
                if has_image
                else chat_settings_service.get_reply_timeout_seconds
            )
            timeout_seconds = await get_timeout(db)
            effective_timeout = timeout_seconds if timeout_seconds > 0 else None

            unflushed = ""
            last_flush = time.monotonic()
            try:
                async with asyncio.timeout(effective_timeout):
                    gen_params = {**(params or {}), "request_id": message_id, "cache_tag": conversation_id}
                    tools = await tool_loop_service.tool_stream_or_none(db, model, ollama_messages, gen_params)
                    async for chunk in tools if tools is not None else chat_stream(model, ollama_messages, gen_params):
                        if isinstance(chunk, dict):  # a finished tool call (MCP), not text
                            await tool_event_service.record(db, message, chunk["tool"], conversation_id)
                            continue
                        full_reply += chunk
                        unflushed += chunk
                        reply_broadcast_service.publish_chunk(conversation_id, chunk)

                        now = time.monotonic()
                        if len(unflushed) >= _FLUSH_CHARS or (now - last_flush) >= _FLUSH_INTERVAL_SECONDS:
                            message.content = full_reply
                            await db.commit()
                            unflushed = ""
                            last_flush = now
                            # Cross-instance guard (see reply_cross_instance_service); stop_model too: a bare
                            # `return` alone never stops the engine.
                            fresh = await reply_cross_instance_service.refresh_or_none(db, message)
                            if fresh is None or fresh.status == "deleted":
                                await stop_model(model, request_id=message_id)
                                return
            except InferenceError as exc:
                # Keeps whatever was generated before the failure (a timeout too).
                await reply_termination_service.mark_error(db, message, full_reply, conversation_id, str(exc))
                return
            except TimeoutError:
                # asyncio.timeout()'s own deadline, not a raw CancelledError: cancellation state is
                # cleared, so `await` works here (unlike the CancelledError branch below).
                await reply_termination_service.mark_error(
                    db, message, full_reply, conversation_id, f"Reply timed out after {timeout_seconds} seconds."
                )
                # Closing our side doesn't reliably stop an in-progress compute phase (see stop_model).
                await stop_model(model, request_id=message_id)
                return
            except asyncio.CancelledError:
                # cancel_on_disconnect or cancel_generation: cancellation is pending, so an `await` here would
                # just re-raise — write from a fresh, non-cancelled task, then re-raise so this one stays cancelled.
                reply_termination_service.schedule_cancelled_write(message_id, model, full_reply, conversation_id)
                raise

            if reply_cancellation_service.is_message_deleted(message_id):
                return  # rare: finished at the same instant as cancel_generation, no await left to catch it above
            # Same check via the database: the one above only sees deletes served by this instance.
            fresh = await reply_cross_instance_service.refresh_or_none(db, message)
            if fresh is None or fresh.status == "deleted":
                return
            if message.tool_events and not full_reply.strip():  # went quiet after using tools
                await reply_termination_service.mark_error(
                    db, message, full_reply, conversation_id, tool_event_service.NO_ANSWER
                )
                return
            message.content = full_reply
            message.status = "complete"
            message.sources = source_filenames or None
            await db.commit()
            reply_broadcast_service.publish_done(conversation_id, source_filenames)
    except Exception:
        # Broad on purpose, mirroring title_service._generate_title_in_background: a detached background task with
        # no caller left to propagate a failure to — otherwise this would vanish with no context at all.
        logger.exception("Reply generation failed for message %s", message_id)
        if conversation_id is not None:
            reply_broadcast_service.publish_error(conversation_id, "Reply generation failed unexpectedly.")
