"""Assistant reply generation, detached from the initiating request by a background asyncio.Task, for every
conversation (personal or channel). Split from chat_service.build_reply_stream (conversation-specific setup)
for the file-size rule and because it must run independently of its triggering request.

Mechanism: stream_reply creates the placeholder assistant Message, hands the engine call to a detached task
(_run_generation), then only relays reply_broadcast_service's hub. It never awaits the engine, so a
disconnecting SSE request cancels just this generator's `await queue.get`, not the task. Whether the task
continues depends on `cancel_on_disconnect`: False for channels, True for personal chats (only the owner could watch).
Either way a failed, cancelled or timed-out attempt leaves a clean status="error" record. A channel's
`/subscribe` connections tap the same hub for "real" delivery, and an owner can reconnect to catch an
in-flight reply's tail; see reply_cross_instance_service for when the in-process hub isn't enough.
"""

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

# Batches how often streaming content is committed to the DB, not per token: pollers and late /subscribe
# joiners don't need finer. Every chunk still reaches the hub at once, so live viewers never see the batching.
_FLUSH_CHARS = 200
_FLUSH_INTERVAL_SECONDS = 0.4

# How often stream_reply's relay also reads the DB, not just the queue (reply_cross_instance_service).
_CROSS_INSTANCE_POLL_SECONDS = 3.0

# As title_service._background_title_tasks: asyncio keeps only a weak reference to a bare create_task result, so it
# could be garbage-collected mid-run.
_background_generation_tasks: set[asyncio.Task] = set()

# delete_message's way of stopping a streaming reply: lives in reply_cancellation_service, re-exported here.
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
    params: dict | None = None,
) -> AsyncIterator[dict]:
    """Yields events for routers/chat.py to SSE-encode ({"chunk"} / {"error"} / {"done", "sources"} / {"deleted"},
    see reply_broadcast_service.publish_deleted), each with "message_id" so the frontend knows the bubble (see
    `cancel_on_disconnect` in the module docstring). `reply_to_message_id` is stored on the placeholder for
    delete_message only (Message.reply_to_message_id). `model`, if given, overrides `conversation.model` for
    this call (chat_service.build_reply_stream's image handling). `has_image` picks which
    chat_settings_service reply timeout applies (get_vision_reply_timeout_seconds). `params`, if given, replaces
    `conversation.params` for this call (chat_service passes the system-wide num_ctx).
    """
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
        message.id,
        effective_model,
        ollama_messages,
        conversation.params if params is None else params,
        source_filenames,
        has_image,
    )

    _state, queue = reply_broadcast_service.subscribe(conversation.id)
    finished = False
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_CROSS_INSTANCE_POLL_SECONDS)
            except TimeoutError:
                # Cross-instance fallback (reply_cross_instance_service): otherwise the sender's tab hangs forever if
                # what it waits for happened on a sibling.
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
    """Starts generation detached from the request, with the strong-ref set + done-callback pattern of
    title_service.schedule_smart_title_generation. Registers with reply_cancellation_service so
    stream_reply/cancel_generation can cancel it, and spawns reply_cross_instance_service's watchdog beside it
    (needed even with cancel_generation; see that function).
    """
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
    """Runs on its own AsyncSessionLocal: the request's session may be gone by then."""
    conversation_id = None
    full_reply = ""
    try:
        async with AsyncSessionLocal() as db:
            message = await db.get(Message, message_id)
            if message is None:
                return  # conversation (and placeholder) deleted from under this generation
            conversation_id = message.conversation_id
            # Read fresh so a change applies to the next message. 0 means no timeout (asyncio.timeout(None)).
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
                    gen_params = {**(params or {}), "request_id": message_id, "conversation_id": conversation_id}
                    tools = await tool_loop_service.tool_stream_or_none(db, model, ollama_messages, gen_params)
                    async for chunk in tools if tools is not None else chat_stream(model, ollama_messages, gen_params):
                        if isinstance(chunk, dict):  # a finished MCP tool call, not text
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
                            # Cross-instance guard (reply_cross_instance_service); also stop_model: a bare `return`
                            # never stops the engine.
                            fresh = await reply_cross_instance_service.refresh_or_none(db, message)
                            if fresh is None or fresh.status == "deleted":
                                await stop_model(model, request_id=message_id)
                                return
            except InferenceError as exc:
                # Keeps what was generated before the failure (a timeout too).
                await reply_termination_service.mark_error(db, message, full_reply, conversation_id, str(exc))
                return
            except TimeoutError:
                # asyncio.timeout's deadline, not a raw CancelledError: cancellation state is cleared, so `await`
                # works
                # here (unlike the CancelledError branch below).
                await reply_termination_service.mark_error(
                    db, message, full_reply, conversation_id, f"Reply timed out after {timeout_seconds} seconds."
                )
                # Closing our side doesn't reliably stop an in-progress compute phase (see stop_model).
                await stop_model(model, request_id=message_id)
                return
            except asyncio.CancelledError:
                # cancel_on_disconnect or cancel_generation: cancellation is pending, so an `await` here would re-raise.
                # Write
                # from a fresh, non-cancelled task, then re-raise so this one stays cancelled.
                reply_termination_service.schedule_cancelled_write(message_id, model, full_reply, conversation_id)
                raise

            if reply_cancellation_service.is_message_deleted(message_id):
                return  # rare: finished at the same instant as cancel_generation; no await left to catch it above
            # Same check via the DB: the one above sees only deletes served by this instance.
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
        # Broad on purpose, like title_service._generate_title_in_background: a detached task with no caller to
        # propagate to; otherwise the failure would vanish without context.
        logger.exception("Reply generation failed for message %s", message_id)
        if conversation_id is not None:
            reply_broadcast_service.publish_error(conversation_id, "Reply generation failed unexpectedly.")
