"""CRUD for chats and their message history (sidebar list, load, rename/configure, delete). New messages
go through app/routers/chat.py, which streams from the engine.

Every endpoint needs a logged-in user and is scoped to their own conversations, so others' chats are
neither visible nor confirmable. Logic: app/services/conversation_service.py.
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
    ShareMessageRequest,
)
from app.services import channel_service, conversation_service
from app.services.auth_service import get_current_user
from app.services.message_share_service import MessageShareService

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("", response_model=list[ConversationOut])
async def list_conversations(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The user's chats, newest first (sidebar)."""
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
    """Poll endpoint for "cheap" channel delivery (chat_settings_service.get_channel_delivery_mode): chat.js
    calls it every few seconds. `since` is the previous call's `server_time`; omitted returns the full history.
    """
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return await conversation_service.get_messages_since(db, conversation, since)


@router.delete("/{conversation_id}/messages/{message_id}", response_model=DeleteMessageResponse)
async def delete_message(
    conversation_id: str,
    message_id: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Channel admins/managers only: removes one message from a channel's shared conversation, live if still
    generating. conversation_service.delete_message returns every soft-deleted id (one or two) so chat.js
    can remove all bubbles. Not for personal chats; this moderates shared ones.
    """
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    if conversation.channel_id is None or not channel_service.can_manage_channel_conversation(conversation, user):
        raise HTTPException(status_code=403, detail="Only this channel's admins/managers can delete a message.")
    message = next((m for m in conversation.messages if m.id == message_id), None)
    if message is None:
        raise HTTPException(status_code=404, detail="Message not found")
    deleted_ids = await conversation_service.delete_message(db, conversation, message)
    return DeleteMessageResponse(deleted_message_ids=deleted_ids)


@router.post("/{conversation_id}/messages/{message_id}/share", response_model=MessageOut)
async def share_message(
    conversation_id: str,
    message_id: str,
    body: ShareMessageRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Copies one message from the caller's private chat into a channel they belong to, as their own plain
    message (no AI reply; MessageShareService). Returns the new message.
    """
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return await MessageShareService(db).share_to_channel(user, conversation, message_id, body.channel_id)


@router.patch("/{conversation_id}", response_model=ConversationOut)
async def update_conversation(
    conversation_id: str,
    body: ConversationUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves per-conversation params (Settings page), renames (sidebar) and switches the model mid-chat (model
    badge)."""
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
