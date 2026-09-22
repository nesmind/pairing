"""Request/response shapes for app/routers/chat.py and the message list
under app/routers/conversations.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class AttachmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    # The only one the frontend actually fetches from
    # (GET /api/chat/attachments/{id}).
    url: str
    filename: str
    # "image"/"text" — picks how chat.js renders it.
    type: str


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    role: str
    content: str
    sources: list[str] | None = None
    # Populated only for a "user"-role message sent with one or more
    # files attached (see app.services.chat_attachment_service and
    # app.models.conversation.Message.attachments) — empty for every
    # other message.
    attachments: list[AttachmentOut] = []
    # "streaming"/"complete"/"error"/"deleted" — see
    # app/models/conversation.py: Message.status. "deleted" only ever
    # appears via GET /messages/latest (see
    # conversation_service.get_messages_since): GET /messages filters it
    # out entirely (see conversation_service.visible_messages), so a
    # fresh load never shows one — only an already-open poll needs to see
    # the transition, to remove a bubble it already rendered.
    status: str = "complete"
    # The real reason a status="error" message failed (see
    # app.models.conversation.Message.error_message — set once, by
    # reply_termination_service.mark_error) — null for every non-error message, and for an error row written
    # before this field existed (no backfill was possible — see that migration's own message). chat.js shows
    # this in place of its own generic "didn't finish" notice whenever it's present.
    error_message: str | None = None
    # Who sent this message, formatted for display (username, plus
    # first/last name if either is set — see app/models/conversation.py:
    # Message.sender_display_name) — a real value for every "user"-role
    # message (personal chat included), but the frontend only ever shows
    # it as a text label for a channel's shared conversation (see
    # chat.js: selectConversation); null for every "assistant"/"system"
    # message.
    sender_display_name: str | None = None
    # The sender's small round avatar (see Message.sender_avatar_url) —
    # null when they haven't uploaded a picture, in which case the
    # frontend renders sender_initials as a plain-letter circle instead
    # (see chat.js: renderAvatar). Both are null for an "assistant"/
    # "system" message, which the frontend never shows an avatar for.
    sender_avatar_url: str | None = None
    sender_initials: str | None = None
    # The raw id behind sender_display_name/sender_avatar_url above —
    # lets chat.js's "click an avatar to see more" modal fetch
    # GET /api/account/{id} for that specific sender (see
    # app/routers/account.py's get_public_profile).
    sender_id: str | None = None
    created_at: datetime


class MessagesLatestResponse(BaseModel):
    """GET /api/conversations/{id}/messages/latest — "cheap" channel
    delivery mode's poll endpoint (see
    app.services.chat_settings_service.get_channel_delivery_mode and
    conversation_service.get_messages_since). Returns only messages new
    or changed since `since`, not the full history on every tick."""

    messages: list[MessageOut]
    # Echoed back by the poller as its next request's `since` — using
    # the server's own clock rather than the client's own avoids any
    # poll gaps or duplicate fetches from client/server clock skew.
    server_time: datetime


class DeleteMessageResponse(BaseModel):
    """DELETE /api/conversations/{id}/messages/{message_id} — echoes back
    every message id actually soft-deleted (see
    conversation_service.delete_message), not just the one the caller
    asked for: deleting a "user" message with a still-streaming reply
    takes that reply down too, and chat.js needs both ids to remove both
    bubbles from the deleter's own tab immediately, rather than waiting
    for the next poll/live-watch tick to notice the second one."""

    deleted_message_ids: list[str]
