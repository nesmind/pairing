"""Persists a reply's terminal, non-success outcome — split out of
reply_generation_service purely to keep that file under CLAUDE.md's
file-size rule. Covers two related but distinct paths: an outright
OllamaError/timeout mid-stream (mark_error, called directly), and a
cancellation — either cancel_on_disconnect or
conversation_service.delete_message's cancel_generation — which can't
safely persist its own write from the cancelled task itself (asyncio
re-raises CancelledError at the very next `await`, regardless of what
it is), so schedule_cancelled_write hands that off to a brand-new,
uncancelled task instead.
"""

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AsyncSessionLocal
from app.models import Message
from app.services import reply_broadcast_service, reply_cancellation_service, reply_cross_instance_service
from app.services.inference_client import stop_model

# Same reasoning as reply_generation_service._background_generation_tasks:
# asyncio only holds a *weak* reference to a bare asyncio.create_task()
# result. Each task removes itself via its own done-callback.
_background_tasks: set[asyncio.Task] = set()


async def mark_error(
    db: AsyncSession, message: Message, full_reply: str, conversation_id: str, error_text: str
) -> None:
    """Shared by reply_generation_service's OllamaError/TimeoutError
    branches and _persist_cancelled_reply below. Both guards matter: the
    in-process one is a fast, common-case short-circuit; reply_cross_instance_service's
    is what actually catches a delete_message call served by a
    *different* sibling instance."""
    if reply_cancellation_service.is_message_deleted(message.id):
        return
    fresh = await reply_cross_instance_service.refresh_or_none(db, message)
    if fresh is None or fresh.status == "deleted":
        return
    message.content = full_reply
    message.status = "error"
    message.error_message = error_text
    await db.commit()
    reply_broadcast_service.publish_error(conversation_id, error_text)


def schedule_cancelled_write(message_id: str, model: str, full_reply: str, conversation_id: str) -> None:
    """Persists a cancelled reply's final state from a brand-new task —
    see this module's own docstring for why it can't just be written
    directly from the cancelled task. `model` is only needed to also
    force-stop it outright afterward (see ollama_client.stop_model's own
    docstring) — cancelling our own task doesn't reliably interrupt an
    in-progress llama.cpp compute phase."""
    task = asyncio.create_task(_persist_cancelled_reply(message_id, model, full_reply, conversation_id))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _persist_cancelled_reply(message_id: str, model: str, full_reply: str, conversation_id: str) -> None:
    async with AsyncSessionLocal() as db:
        message = await db.get(Message, message_id)
        if message is not None:
            # If message is None, the conversation (and this row with it) was
            # deleted outright — there's nothing left to write, but Matricxon/
            # Ollama still needs the stop signal below regardless.
            await mark_error(
                db, message, full_reply, conversation_id, "Cancelled: left the chat before this reply finished."
            )
    await stop_model(model)
