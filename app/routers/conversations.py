"""
CRUD endpoints for chat sessions ("conversations") and their message
history. The chat UI calls these to list past chats in the sidebar, load
a chat's messages when it's clicked, rename/re-configure it, or delete it.
Sending a *new* message lives in app/routers/chat.py instead, since that
path also talks to Ollama and streams a response.

Every endpoint here requires a logged-in user (Depends(get_current_user))
and every query is scoped to that user's own conversations — one user
never sees, and can't be told the existence of, another user's chats.
See app/services/conversation_service.py for the actual business logic.
"""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Conversation, User
from app.schemas import (
    ConversationCreate,
    ConversationOut,
    ConversationUpdate,
    DeleteMessageResponse,
    MessageOut,
    MessagesLatestResponse,
    OkResponse,
)
from app.services import channel_service, conversation_service
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
async def list_conversations(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Newest-first list of the current user's own chats, for populating
    the sidebar."""
    return (
        (
            await db.execute(
                select(Conversation).where(Conversation.owner_id == user.id).order_by(Conversation.updated_at.desc()),
            )
        )
        .scalars()
        .all()
    )


@router.post("", response_model=ConversationOut)
async def create_conversation(
    body: ConversationCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await conversation_service.create_conversation(db, body, user)


@router.get("/{conversation_id}", response_model=ConversationOut)
async def get_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)


@router.get("/{conversation_id}/messages", response_model=list[MessageOut])
async def get_messages(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return conversation_service.visible_messages(conversation)


@router.get("/{conversation_id}/messages/latest", response_model=MessagesLatestResponse)
async def get_latest_messages(
    conversation_id: str,
    since: datetime | None = None,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """ "Cheap" channel delivery mode's poll endpoint (see
    app.services.chat_settings_service.get_channel_delivery_mode) — chat.js
    calls this every few seconds for a channel chat it has open instead
    of refetching the full history each tick. `since` should be the
    `server_time` echoed back by the previous call to this same
    endpoint; omitted, it returns the full current history."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return await conversation_service.get_messages_since(db, conversation, since)


@router.delete("/{conversation_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(
    conversation_id: str,
    message_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Channel admins/managers only — removes a single message from a
    channel's shared conversation, live if it's still generating (see
    conversation_service.delete_message, whose return value — every id
    actually soft-deleted, one or two — chat.js uses to remove every
    affected bubble from the deleter's own tab immediately). Not offered
    for a personal chat: only its owner could ever see it there, and this
    endpoint exists for moderating a *shared* conversation, not editing
    your own history."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    if conversation.channel_id is None or not channel_service.can_manage_channel_conversation(conversation, user):
        raise HTTPException(status_code=403, detail="Only this channel's admins/managers can delete a message.")
    message = next((m for m in conversation.messages if m.id == message_id), None)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    deleted_ids = await conversation_service.delete_message(db, conversation, message)
    return DeleteMessageResponse(deleted_message_ids=deleted_ids)


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Used by the Settings page to save per-conversation params, by the
    sidebar to rename a chat, and by the chat page's model badge to
    switch an existing conversation to a different (installed) model
    mid-conversation."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return await conversation_service.update_conversation(db, conversation, body, user)


@router.delete("/{conversation_id}", response_model=OkResponse)
async def delete_conversation(
    conversation_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    await conversation_service.delete_conversation(db, conversation)
    return OkResponse()
