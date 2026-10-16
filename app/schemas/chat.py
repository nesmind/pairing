"""Request/response shapes for app/routers/chat.py and the message list in app/routers/conversations.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # The only one the frontend fetches (GET /api/chat/attachments/{id}).
    url: str
    filename: str
    # "image"/"text": how chat.js renders it.
    type: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    sources: list[str] | None = None
    # MCP tool calls the reply made, in order (Message.tool_events); null if none.
    tool_events: list[dict] | None = None
    # Only for "user" messages sent with files (chat_attachment_service, Message.attachments).
    attachments: list[AttachmentOut] = []
    # "streaming"/"complete"/"error"/"deleted" (Message.status). "deleted" appears only via GET /messages/latest
    # (conversation_service.get_messages_since); GET /messages filters it (visible_messages). Open polls need
    # it to remove a rendered bubble.
    status: str = "complete"
    # The real failure reason of a status="error" message (Message.error_message, set by
    # reply_model_unload_service.mark_error); null otherwise and for older errors (no backfill). chat.js shows
    # it instead of its generic notice.
    error_message: str | None = None
    # Sender label (username plus first/last name; Message.sender_display_name) for every "user" message, but
    # the frontend shows it only in channels (chat.js selectConversation). Null for assistant/system.
    sender_display_name: str | None = None
    # The sender's avatar (Message.sender_avatar_url); if null, chat.js renders sender_initials as a letter
    # circle (renderAvatar). Both null for assistant/system.
    sender_avatar_url: str | None = None
    sender_initials: str | None = None
    # "user" or "assistant" if shared into a channel from a private chat (Message.shared_from); null normally.
    # chat.js shows a "Shared from a private chat" remark.
    shared_from: str | None = None
    # The id behind sender_display_name/sender_avatar_url; chat.js's avatar-click modal fetches GET
    # /api/account/{id} (routers/account.py get_public_profile).
    sender_id: str | None = None
    created_at: datetime
    # When a reply ended (Message.completed_at); chat.js shows it instead of created_at.
    completed_at: datetime | None = None


class MessagesLatestResponse(BaseModel):
    """GET /api/conversations/{id}/messages/latest, the "cheap" channel delivery poll
    (chat_settings_service.get_channel_delivery_mode, conversation_service.get_messages_since): only
    messages new or changed since `since`.
    """

    messages: list[MessageOut]
    # Sent back as the next `since`; the server's clock avoids client skew gaps or duplicates.
    server_time: datetime


class DeleteMessageResponse(BaseModel):
    """DELETE /api/conversations/{id}/messages/{message_id}: every soft-deleted id
    (conversation_service.delete_message).
    Deleting a "user" message takes its streaming reply too; chat.js needs both ids to remove both bubbles at once.
    """

    deleted_message_ids: list[str]


class ShareMessageRequest(BaseModel):
    """Body of POST /api/conversations/{id}/messages/{message_id}/share."""

    channel_id: str
