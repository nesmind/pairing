"""
Business logic behind app/routers/conversations.py: creating a chat with
the right defaults, the shared "does this conversation belong to this
user" lookup every conversation/chat/notes endpoint relies on, and
validating a model switch against what's actually installed.
"""

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Message, User
from app.models._base import utcnow
from app.schemas import ConversationCreate, ConversationUpdate, MessagesLatestResponse
from app.services import channel_service, chat_attachment_service, reply_broadcast_service, reply_generation_service
from app.services.model_catalog_service import installed_chat_models
from app.services.settings_service import get_default_model, get_default_params


async def create_conversation(db: AsyncSession, body: ConversationCreate, user: User) -> Conversation:
    """Starts a brand-new, empty chat with the current user's own default
    model and generation params (see
    app.services.settings_service.get_default_model/get_default_params)."""
    conversation = Conversation(
        owner_id=user.id,
        model=body.model or await get_default_model(db, user),
        params=await get_default_params(db, user),
    )
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_accessible_conversation_or_404(db: AsyncSession, conversation_id: str, user: User) -> Conversation:
    """Looks up a conversation and 404s (not 403) if it doesn't exist, OR
    belongs to someone else with no way in for `user` — the same
    response either way, so a caller can't distinguish "not found" from
    "not yours" and enumerate other conversation ids. `user` can reach a
    conversation either by owning it (a personal chat) or, if it's a
    channel's shared conversation (channel_id set — see
    app/models/conversation.py), by being a member of that channel;
    global admins get no special-case here — they need membership too
    (relies on Conversation.channel / Channel.members both being
    lazy="selectin", so this needs no extra query)."""
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
    """Used by the Settings page to save per-conversation params, by the
    sidebar to rename a chat, and by the chat page's model badge to
    switch an existing conversation to a different (installed) model
    mid-conversation. Only fields the caller actually sent are changed —
    everything else is left as-is.

    For a channel's shared conversation, `title` isn't editable here at
    all (the channel's own `name`, edited via the admin Channels tab, is
    what the sidebar/header show instead), and `model`/`params` require
    `user` to be an admin or that channel's manager (see
    channel_service.can_manage_channel_conversation) — a plain member can
    read and send messages but not reconfigure the channel's chat.
    Personal conversations are unaffected: their owner always has full
    rights, exactly as before this check existed.
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
        conversation.model = body.model
    if body.params is not None:
        conversation.params = body.params.model_dump()
    await db.commit()
    await db.refresh(conversation)
    return conversation


async def get_messages_since(
    db: AsyncSession, conversation: Conversation, since: datetime | None
) -> MessagesLatestResponse:
    """Powers "cheap" channel delivery mode's poll endpoint (GET
    /api/conversations/{id}/messages/latest — see
    app.services.chat_settings_service.get_channel_delivery_mode). `since=None`
    returns the full, already-loaded history — covers a poller's very
    first tick, with no extra query. Otherwise queries fresh for every
    message whose `updated_at` is newer than `since`: because
    Message.updated_at has onupdate=utcnow (see
    app/models/conversation.py), this catches both a brand-new message
    and an existing one whose content just grew as a channel reply
    streams in, not only newly-created rows."""
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
    """True if this conversation already has an assistant reply currently
    generating (Message.status == "streaming") — the backend's
    authoritative guard behind "only one AI reply at a time per channel"
    (see app/routers/chat.py's stream_reply, its only caller). A
    channel's shared conversation can have more than one member trying
    to trigger a reply at once; a personal chat has exactly one owner,
    so the router never even checks this for one.

    Deliberately best-effort, not race-free: there's a real gap between
    this SELECT and reply_generation_service.stream_reply's own
    placeholder Message INSERT moments later, so two callers racing
    within that window could both pass this check. Not solved with a
    DB-level unique constraint or advisory lock — this is a soft UX
    guard against the ordinary case (one member notices the AI is
    already replying and tries anyway), not a hard exactly-once
    guarantee, consistent with this codebase's other accepted,
    documented limitations (e.g. reply_broadcast_service's own
    single-instance-only note)."""
    result = await db.execute(
        select(Message.id).where(Message.conversation_id == conversation.id, Message.status == "streaming").limit(1)
    )
    return result.scalar_one_or_none() is not None


def visible_messages(conversation: Conversation) -> list[Message]:
    """This conversation's messages minus any a channel admin/manager has
    deleted (see delete_message) — soft-deleted rather than removed from
    the table, so an already-open "cheap" mode poll can still see the
    status transition itself (a deleted message's updated_at bump — see
    get_messages_since, which deliberately does NOT use this filter, for
    exactly that reason) and remove a bubble it already rendered. This
    filter is for every *other* read of the full history instead: a
    fresh GET /messages load, or chat.js's own full reconcile, neither of
    which should ever show a deleted message in the first place."""
    return [m for m in conversation.messages if m.status != "deleted"]


async def _cancel_and_clear(db: AsyncSession, conversation_id: str, message: Message) -> None:
    """Cancels `message`'s generation if it's "streaming" (mark_message_deleted
    *before* cancel_generation, no `await` between — else
    reply_generation_service's own write for this reply can race this
    commit and win), tells every live watcher directly (publish_deleted)
    rather than leaving them waiting on a "done"/"error" that never
    comes, then soft-deletes it (see visible_messages for why). Shared by
    delete_message for the message being deleted and, if it's a user
    message with a still-streaming reply, that reply too."""
    if message.status == "streaming":
        reply_generation_service.mark_message_deleted(message.id)
        reply_generation_service.cancel_generation(message.id)
        reply_broadcast_service.publish_deleted(conversation_id)
    message.status = "deleted"
    message.content = ""
    message.sources = None
    await chat_attachment_service.delete_attachment_files(db, message)


async def delete_message(db: AsyncSession, conversation: Conversation, message: Message) -> list[str]:
    """Removes a single message from a channel's shared conversation —
    channel admins/managers only (checked by the router) — an explicit
    "stop and remove" action, unlike a channel's normal disconnect
    behavior (see _cancel_and_clear above).

    Deleting a "user" message whose reply is still generating takes that
    reply down too, found via Message.reply_to_message_id rather than
    inferred from timestamps (see that column's docstring for why):
    leaving the model visibly still answering a just-deleted question
    would be a strange state to show. An already-settled reply is left
    alone — deletable separately, the normal way, if wanted. Returns
    every id actually soft-deleted (one or two) — chat.js uses it to
    remove every affected bubble immediately, not just the clicked one."""
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
    """Sweeps any message still marked "streaming" from before this
    process started — an asyncio.Task can't survive a restart, so
    without this a row would stay stuck showing a permanent "typing"
    state. Whatever content had already been flushed is kept in place.

    Scoped to a single-instance restart only: one sibling recycling while others (instance_count > 1) stay up mid-
    generation needs a per-message owning-instance lease to handle correctly, not justified given instance_count
    defaults to 1. Called once from app.services.startup_service.run_startup_tasks, IS_PRIMARY-gated like that
    function's other one-time startup repairs.
    """
    result = await db.execute(update(Message).where(Message.status == "streaming").values(status="error"))
    await db.commit()
    return result.rowcount


async def delete_conversation(db: AsyncSession, conversation: Conversation) -> None:
    """Deletes a personal conversation. A channel's shared conversation
    can't be deleted this way — a channel is never left without one, so
    that only happens by deleting the whole channel (admin-only, see
    channel_service.delete_channel), which cascades to it instead."""
    if conversation.channel_id is not None:
        raise HTTPException(
            status_code=400,
            detail="Channel conversations can only be deleted by deleting the channel.",
        )
    chat_attachment_service.delete_conversation_attachments(conversation.id)
    await db.delete(conversation)
    await db.commit()
