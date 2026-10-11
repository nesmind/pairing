"""Logic for routers/conversations.py: creating chats with the right defaults, the shared "does this
conversation belong to this user" lookup used by every conversation/chat/notes endpoint, and validating
model switches against what's installed.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message, User
from app.models._base import utcnow
from app.schemas import ConversationCreate, ConversationUpdate, MessagesLatestResponse
from app.services import (
    channel_service,
    chat_attachment_service,
    engine_service,
    note_service,
    reply_broadcast_service,
    reply_generation_service,
)
from app.services.matricxon_cache_persistence import MatricxonCachePersistenceService
from app.services.model_catalog_service import HiddenModelTags, installed_chat_models, resolve_installed_model
from app.services.settings_service import get_default_model, get_default_params

logger = logging.getLogger("llama_chat")


async def create_conversation(db: AsyncSession, body: ConversationCreate, user: User) -> Conversation:
    """Starts an empty chat with the user's default model and params
    (settings_service.get_default_model/get_default_params).

    An explicit `body.model` is validated strictly like update_conversation's switch (400 if not installed).
    The *default* path differs: get_default_model has no installed-check and goes stale (engine switch or
    deleted model,), so resolve_installed_model falls back instead of giving a new chat a model that
    404s on first send.
    """
    if body.model is not None:
        installed = await installed_chat_models()
        if body.model not in installed:
            raise HTTPException(status_code=400, detail="Model is not installed")
        if await HiddenModelTags(db).blocks(user, body.model):
            raise HTTPException(status_code=400, detail="Model is disabled by an admin")
        model = body.model
    else:
        model = await resolve_installed_model(db, user, await get_default_model(db, user))

    params = await note_service.seed_initial_disabled_notes(db, user, await get_default_params(db, user))
    conversation = Conversation(
        owner_id=user.id,
        model=model,
        params=params,
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_accessible_conversation_or_404(db: AsyncSession, conversation_id: str, user: User) -> Conversation:
    """Looks up a conversation; 404 (not 403) if missing or not accessible, so ids can't be enumerated. Access
    is by ownership (personal chat) or channel membership (channel_id set); admins need membership too
    (Conversation.channel / Channel.members are selectin-loaded, so no extra query).
    """
    conversation = await db.get(Conversation, conversation_id)
    if conversation is not None:
        if conversation.owner_id == user.id:
            return conversation
        if conversation.channel_id is not None and any(m.user_id == user.id for m in conversation.channel.members):
            return conversation
    raise HTTPException(status_code=404, detail="Conversation not found")


async def update_conversation(
    db: AsyncSession,
    conversation: Conversation,
    body: ConversationUpdate,
    user: User,
) -> Conversation:
    """Saves per-conversation params (Settings page), renames (sidebar) and switches the model mid-chat (model
    badge). Only fields the caller sent change.

    For a channel's shared conversation `title` is not editable (the channel `name` is shown instead), and
    `model`/`params` need an admin or the channel's manager (channel_service.can_manage_channel_conversation);
    members can read and send only. Personal chats: owners keep full rights.
    """
    if conversation.channel_id is not None:
        if body.title is not None:
            raise HTTPException(
                status_code=400,
                detail="Channel conversations are renamed by editing the channel, not the chat.",
            )
        if (body.model is not None or body.params is not None) and not channel_service.can_manage_channel_conversation(
            conversation, user
        ):
            raise HTTPException(
                status_code=403,
                detail="Only admins and this channel's managers can change its model or generation settings.",
            )

    if body.title is not None:
        conversation.title = body.title
    if body.model is not None:
        installed = await installed_chat_models()
        if body.model not in installed:
            raise HTTPException(status_code=400, detail="Model is not installed")
        if await HiddenModelTags(db).blocks(user, body.model):
            raise HTTPException(status_code=400, detail="Model is disabled by an admin")
        conversation.model = body.model
    if body.params is not None:
        saved_tools = (conversation.params or {}).get("tool_snapshot", [])
        conversation.params = {**body.params.model_dump(), "tool_snapshot": saved_tools}
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_messages_since(
    db: AsyncSession, conversation: Conversation, since: datetime | None
) -> MessagesLatestResponse:
    """Poll for "cheap" channel delivery (GET /api/conversations/{id}/messages/latest;
    chat_settings_service.get_channel_delivery_mode). `since=None` returns the loaded history (a poller's first
    tick, no extra query). Otherwise queries messages with `updated_at` newer than `since`; since
    Message.updated_at has onupdate=utcnow, this catches new messages and ones whose content grew while streaming.
    """
    server_time = utcnow()
    if since is None:
        messages = conversation.messages
    else:
        messages = (
            (
                await db.execute(
                    select(Message)
                    .where(Message.conversation_id == conversation.id, Message.updated_at > since)
                    .order_by(Message.created_at)
                )
            )
            .scalars()
            .all()
        )
    return MessagesLatestResponse(messages=messages, server_time=server_time)


async def has_active_reply(db: AsyncSession, conversation: Conversation) -> bool:
    """True if the conversation has a generating assistant reply (status "streaming"): the guard for "one AI
    reply at a time per channel" (routers/chat.py stream_reply, the only caller; personal chats skip it).

    Best-effort, not race-free: there is a gap between this SELECT and stream_reply's placeholder INSERT, so
    racing callers can both pass. A soft UX guard, not exactly-once (no unique constraint or advisory lock),
    like reply_broadcast_service's single-instance note.
    """
    result = await db.execute(
        select(Message.id).where(Message.conversation_id == conversation.id, Message.status == "streaming").limit(1)
    )
    return result.scalar_one_or_none() is not None


def visible_messages(conversation: Conversation) -> list[Message]:
    """Messages minus ones a channel admin/manager deleted (soft-deleted, so an open "cheap" poll sees the
    status change; get_messages_since deliberately does NOT use this filter). For every other full read
    (fresh GET /messages, chat.js reconcile), which must not show deleted messages.
    """
    return [m for m in conversation.messages if m.status != "deleted"]


async def _cancel_and_clear(db: AsyncSession, conversation_id: str, message: Message) -> None:
    """Cancels `message`'s generation if "streaming" (mark_message_deleted *before* cancel_generation, no `await`
    between, else reply_generation_service's own write can race this commit and win), tells live watchers
    directly (publish_deleted) instead of leaving them waiting for a "done"/"error", then soft-deletes it
    (see visible_messages). Used by delete_message for the message and, if it's a user message with a
    still-streaming reply, that reply.
    """
    if message.status == "streaming":
        reply_generation_service.mark_message_deleted(message.id)
        reply_generation_service.cancel_generation(message.id)
        reply_broadcast_service.publish_deleted(conversation_id)
    message.status = "deleted"
    message.content = ""
    message.sources = None
    await chat_attachment_service.delete_attachment_files(db, message)


async def delete_message(db: AsyncSession, conversation: Conversation, message: Message) -> list[str]:
    """Removes one message from a channel's shared conversation (channel admins/managers; router-checked): an
    explicit "stop and remove", unlike a channel's normal disconnect behavior (_cancel_and_clear).

    Deleting a "user" message with a generating reply takes the reply down too, found via
    Message.reply_to_message_id, not timestamps (see that column); showing the model answering a deleted
    question would be odd. A settled reply stays (deletable separately). Returns every soft-deleted id (one or
    two) so chat.js removes all affected bubbles at once.
    """
    deleted_ids = [message.id]
    await _cancel_and_clear(db, conversation.id, message)
    if message.role == "user":
        reply = next(
            (m for m in conversation.messages if m.reply_to_message_id == message.id and m.status == "streaming"),
            None,
        )
        if reply is not None:
            await _cancel_and_clear(db, conversation.id, reply)
            deleted_ids.append(reply.id)
    await db.commit()
    return deleted_ids


async def mark_interrupted_messages_as_errored(db: AsyncSession) -> int:
    """Marks messages still "streaming" from before this process started as errors (an asyncio.Task can't survive
    a restart; otherwise they'd show a permanent "typing" state). Already-flushed content is kept.

    Single-instance restarts only: a sibling recycling while others (instance_count > 1) generate would need a
    per-message owning-instance lease, not worth it while instance_count defaults to 1. Called once from
    startup_service.run_startup_tasks, IS_PRIMARY-gated like its other startup repairs.
    """
    result = await db.execute(
        update(Message)
        .where(Message.status == "streaming")
        .values(status="error", error_message="Interrupted by a server restart.")
    )
    await db.commit()
    return result.rowcount


async def delete_conversation(db: AsyncSession, conversation: Conversation) -> None:
    """Deletes a personal conversation. A channel's conversation can't be deleted this way (a channel always has
    one); only deleting the channel (admin, channel_service.delete_channel) cascades to it.

    A streaming reply is cancelled first, with the same mark_message_deleted-before-cancel_generation order and
    no `await` between as _cancel_and_clear (else the generation's write can race the cascade delete and win).
    Otherwise it notices the missing row only via reply_cross_instance_service's watchdog seconds later.
    """
    if conversation.channel_id is not None:
        raise HTTPException(
            status_code=400,
            detail="Channel conversations can only be deleted by deleting the channel.",
        )
    for message in conversation.messages:
        if message.status == "streaming":
            reply_generation_service.mark_message_deleted(message.id)
            reply_generation_service.cancel_generation(message.id)
    chat_attachment_service.delete_conversation_attachments(conversation.id)
    conversation_id = conversation.id
    await db.delete(conversation)
    await db.commit()
    await _drop_engine_cache(conversation_id)


async def _drop_engine_cache(conversation_id: str) -> None:
    """Matricxon keeps each chat's prompt cache on disk; tell it the chat is gone. A slow or unreachable host is
    skipped and never delays or fails the delete for long.
    """
    if engine_service.current_engine() != "matricxon":
        return
    try:
        await asyncio.wait_for(MatricxonCachePersistenceService().forget_chat(conversation_id), timeout=3)
    except TimeoutError:
        logger.info("dropping a deleted chat's prompt cache timed out")
