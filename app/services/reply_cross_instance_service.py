"""Cross-instance correctness helpers for reply_generation_service —
split out purely to keep that file under CLAUDE.md's file-size rule.

Needed because reply_broadcast_service and reply_cancellation_service
are both purely in-process memory (see their own docstrings): once this
app runs more than one sibling instance (see app.services.instance_pool
— relevant once an admin raises instance_count above its default of 1),
a conversation_service.delete_message call served by a *different*
instance than the one running a reply's own generation task is
completely invisible to that task's in-process checks — it would either
hang forever waiting for an event nothing will ever publish to it, or
blindly overwrite the deletion once it finishes, or (watch_for_deletion
below) just keep burning CPU/GPU on an Ollama request nobody's waiting
on anymore. All three fall back to the one thing every instance actually
shares: the database.
"""

import asyncio

from sqlalchemy.exc import InvalidRequestError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.exc import ObjectDeletedError

from app.database import AsyncSessionLocal
from app.models import Message


async def refresh_or_none(db: AsyncSession, message: Message) -> Message | None:
    """Re-reads just `message`'s own status column fresh from the
    database — not the whole row, so a caller's own in-memory content
    (like reply_generation_service's `full_reply` accumulator) stays its
    own source of truth, unaffected. Returns `message` (mutated in
    place) if the row still exists, or None if it was hard-deleted
    outright (a whole conversation deleted, not just one message — see
    conversation_service.delete_conversation).

    Catches two different exceptions for what looks like the same
    situation: an explicit refresh() call like this one raises
    InvalidRequestError ("Could not refresh instance") when the row is
    simply gone — ObjectDeletedError is a *different* code path, only
    raised by an implicit lazy-unexpire triggered by touching an
    already-expired attribute, which this function's own explicit
    attribute_names refresh never goes through. Both are kept here since
    either genuinely can fire depending on the object's prior state."""
    try:
        await db.refresh(message, attribute_names=["status"])
        return message
    except (ObjectDeletedError, InvalidRequestError):
        return None


def terminal_event_for(message: Message | None, message_id: str) -> dict:
    """Synthesizes the event reply_generation_service.stream_reply's
    cross-instance fallback would otherwise have relayed from the
    broadcast hub, for a message whose status moved on to something no
    longer "streaming" on a sibling instance this process never heard
    about directly."""
    if message is None or message.status == "deleted":
        return {"deleted": True, "message_id": message_id}
    if message.status == "error":
        # The real error text was already published (or logged) on
        # whichever instance actually ran the generation — message.content
        # here only ever holds the partial *reply* text, not an error
        # description, so this is a generic stand-in.
        return {"error": "Reply generation failed.", "message_id": message_id}
    return {"done": True, "sources": message.sources, "message_id": message_id}


async def watch_for_deletion(message_id: str, task: asyncio.Task, poll_interval: float = 2.0) -> None:
    """Runs alongside a reply's generation task (see
    reply_generation_service._schedule_generation, the only caller),
    re-reading this message's own row directly from the database every
    `poll_interval` seconds. If it's "deleted" before `task` finishes on
    its own, cancels `task` outright — the *same* effect
    reply_cancellation_service.cancel_generation already has, just
    triggered from inside this process instead of requiring an external
    DELETE request to reach it directly (this watchdog always runs on
    the same instance as the generation it's watching, so this call
    always succeeds even when the DELETE request itself landed on a
    different sibling one).

    Needed on top of _run_generation's own flush-point check: that one
    only ever runs once a chunk has actually arrived, so during a slow
    cold model load (no chunks yet — can genuinely take tens of seconds)
    nothing else notices a delete_message call at all, cross-instance or
    not — the underlying Ollama request just keeps burning CPU/GPU with
    nobody watching it. Exits harmlessly once `task` finishes on its
    own, whichever comes first."""
    while not task.done():
        await asyncio.sleep(poll_interval)
        if task.done():
            return
        async with AsyncSessionLocal() as db:
            message = await db.get(Message, message_id)
        if message is None or message.status == "deleted":
            task.cancel()
            return
