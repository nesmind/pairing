"""Cross-instance correctness helpers for reply_generation_service, split out for the file-size rule.

reply_broadcast_service and reply_cancellation_service are in-process memory, so with several instances
(instance_pool, instance_count above 1) a conversation_service.delete_message served by a *different*
instance than the reply's generation task is invisible to its in-process checks: the task could hang waiting
for an event nobody publishes, overwrite the deletion when done, or (watch_for_deletion) keep burning CPU/GPU
on an engine request nobody awaits. All three fall back to what every instance shares: the database.
"""

import asyncio

from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import ObjectDeletedError

from app.database import AsyncSessionLocal
from app.models import Message


async def refresh_or_none(db: AsyncSession, message: Message) -> Message | None:
    """Re-reads just `message`'s status fresh from the DB (not the whole row, so the caller's in-memory content like
    `full_reply` stays authoritative). Returns `message` (mutated) if the row exists, None if hard-deleted (a
    whole conversation, conversation_service.delete_conversation).

    Catches two exceptions for one situation: an explicit refresh raises InvalidRequestError ("Could not
    refresh instance") for a missing row; ObjectDeletedError comes only from implicitly unexpiring an expired
    attribute, which this function's explicit attribute_names refresh doesn't hit. Either can fire depending on prior
    state.
    """
    try:
        await db.refresh(message, attribute_names=["status"])
        return message
    except (ObjectDeletedError, InvalidRequestError):
        return None


def terminal_event_for(message: Message | None, message_id: str) -> dict:
    """Synthesizes the event stream_reply's cross-instance fallback would have relayed from the hub, for a message
    whose status left "streaming" on a sibling instance this process never heard from.
    """
    if message is None or message.status == "deleted":
        return {"deleted": True, "message_id": message_id}
    if message.status == "error":
        # The real error text was published or logged on the instance that ran the generation; message.content holds
        # only
        # the partial reply, so this is a generic stand-in.
        return {"error": "Reply generation failed.", "message_id": message_id}
    return {"done": True, "sources": message.sources, "message_id": message_id}


async def watch_for_deletion(message_id: str, task: asyncio.Task, poll_interval: float = 2.0) -> None:
    """Runs beside a reply's generation task (reply_generation_service._schedule_generation, only caller), re-reading
    the message row from the DB every `poll_interval` seconds. If it becomes "deleted" before `task` ends, cancels
    `task`: the effect of reply_cancellation_service.cancel_generation, triggered locally (this watchdog runs on
    the generation's instance, so it works even if the DELETE landed on a sibling).

    Needed beyond _run_generation's flush-point check, which only runs once a chunk arrives: during a slow cold model
    load (no chunks for tens of seconds) nothing else notices delete_message, cross-instance or not, and the
    engine request keeps burning CPU/GPU unwatched. Exits harmlessly when `task` finishes first.
    """
    while not task.done():
        await asyncio.sleep(poll_interval)
        if task.done():
            return
        async with AsyncSessionLocal() as db:
            message = await db.get(Message, message_id)
        if message is None or message.status == "deleted":
            task.cancel()
            return
