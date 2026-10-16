"""Conversation and Message models, together because a Message never exists without its Conversation."""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.config import DEFAULT_GENERATION_PARAMS
from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Conversation(Base):
    """One chat session (one sidebar entry)."""

    __tablename__ = "conversations"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Owner; every query in conversation_service.py / chat_service.py filters by it. Nullable for
    # pre-multi-user data and channel conversations. Exactly one of owner_id/channel_id is set on new rows.
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # Set instead of owner_id for a channel's shared thread; access is by membership
    # (conversation_service.get_accessible_conversation_or_404). Unique: one per channel.
    channel_id = Column(String(ID_LEN), ForeignKey("channels.id"), nullable=True, unique=True)
    # Auto-generated sidebar label (chat_service.maybe_generate_title); "New chat" until the first exchange.
    title = Column(String(255), default="New chat")
    # Model for this conversation; stored per chat so old chats keep their model.
    model = Column(String(255), nullable=True)
    # Adjustable generation parameters (temperature, top_p, rag_top_k,...) as JSON: a copy of the
    # defaults, overwritten when the Settings page is saved for this conversation.
    params = Column(JSON, default=lambda: dict(DEFAULT_GENERATION_PARAMS))
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # cascade="all, delete-orphan" removes messages with the conversation. lazy="selectin" makes
    # `conversation.messages` safe under the async session (the app's default for accessed relationships).
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )
    # Pins are deleted with the conversation (a NotePin without one is meaningless).
    note_pins = relationship(
        "NotePin",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    # The owning channel, if any. No cascade: Channel.conversation owns delete-orphan.
    channel = relationship("Channel", back_populates="conversation", lazy="selectin")


class Message(Base):
    """One turn in a conversation, from the user or the model."""

    __tablename__ = "messages"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    conversation_id = Column(String(ID_LEN), ForeignKey("conversations.id"), nullable=False)
    # "user", "assistant" or "system", as the engine's roles, so history is forwarded unmodified.
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    # Filenames of the documents whose chunks were injected for this reply (JSON list), so the "Sources"
    # note survives a reload (app/services/document_service.py). Empty for non-RAG replies.
    sources = Column(JSON, nullable=True)
    # Who typed a "user" message (chat_service.build_reply_stream); null for assistant/system. Matters in
    # channels (MessageOut.sender_display_name); a personal chat has one author.
    sender_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # The "user" message this assistant reply answers, set when reply_generation_service.stream_reply
    # creates the placeholder. conversation_service.delete_message uses it to cancel a still-streaming reply
    # (timestamps are fragile when channel members interleave messages, see build_reply_stream's ask_ai=False).
    reply_to_message_id = Column(String(ID_LEN), ForeignKey("messages.id"), nullable=True)
    # "streaming" while generating in the background (reply_generation_service), "complete" when done (also
    # for all older rows), "error" if it failed partway (`content` keeps what was generated), "deleted" if a
    # channel admin/manager removed it (conversation_service.delete_message). Soft-deleted so open polls can
    # remove the bubble; conversation_service.visible_messages filters them out.
    status = Column(String(20), nullable=False, default="complete")
    # MCP tool calls made while writing this reply, in order: {id, server, tool, arguments, result, is_error}
    # (app.services.tool_loop_service). Null when no tools were used.
    tool_events = Column(JSON, nullable=True)
    # Set on a message shared into a channel from a private chat (app.services.message_share_service): the
    # original role, "user" or "assistant". The copy is a plain "user" message from the sharer (never triggers
    # an AI reply); this only drives the "Shared from a private chat" remark. No link back, to keep the chat private.
    shared_from = Column(String(20), nullable=True)
    # The real failure reason of a status="error" message (e.g. Matricxon's "not enough memory to load...",
    # matricxon_client._error_detail_from_body), set by reply_model_unload_service.mark_error; null otherwise.
    # Before this column the text reached only live SSE viewers (reply_broadcast_service.publish_error);
    # MessageOut.error_message now lets a reload show it too.
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    # When an assistant reply ended (complete, error or cancelled); shown as its time. created_at is when it
    # started and still orders the messages. Null for user messages and replies still streaming.
    completed_at = Column(DateTime, nullable=True)
    # Bumped on every content update, including streaming flushes (reply_generation_service); lets
    # conversation_service.get_messages_since report changes without a change log. Nullable for migrated
    # rows, always set by the ORM default/onupdate.
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    # Files on a "user" message: up to MAX_ATTACHMENT_DOCUMENTS documents and MAX_ATTACHMENT_IMAGES images
    # (chat_attachment_service.save_attachments); never on assistant/system. selectin as in Conversation.messages.
    # Cascade deletes the rows; chat_attachment_service removes the files separately.
    attachments = relationship(
        "MessageAttachment",
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessageAttachment.filename",
        lazy="selectin",
    )
    # lazy="selectin" (see Conversation.messages); sender_display_name reads sender names.
    sender = relationship("User", lazy="selectin")

    @property
    def sender_display_name(self) -> str | None:
        """For MessageOut.sender_display_name (chat.js bubbleFor): "First Last (username)", or just the username.
        Null without sender_id. A property (not a column) because it combines several fields.
        """
        if not self.sender_id:
            return None
        full_name = " ".join(part for part in (self.sender.first_name, self.sender.last_name) if part)
        return f"{full_name} ({self.sender.username})" if full_name else self.sender.username

    @property
    def sender_avatar_url(self) -> str | None:
        """For MessageOut.sender_avatar_url: the avatar chat.js shows beside every "user" bubble. Null without a
        sender or uploaded picture; sender_initials is the fallback.
        """
        return self.sender.avatar_url if self.sender_id else None

    @property
    def sender_initials(self) -> str | None:
        """For MessageOut.sender_initials, the avatar fallback (see User.initials). Null only for assistant/system."""
        return self.sender.initials if self.sender_id else None


class MessageAttachment(Base):
    """One file attached to a "user" message (caps: app.config.MAX_ATTACHMENT_*, chat_attachment_service)."""

    __tablename__ = "message_attachments"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    message_id = Column(String(ID_LEN), ForeignKey("messages.id"), nullable=False)
    # Relative to ATTACHMENTS_DIR; served only by app/routers/chat.py GET /attachments/{attachment_id}.
    path = Column(String(500), nullable=False)
    # Sanitized original name, for display.
    filename = Column(String(255), nullable=False)
    # "image" (sent to a vision model) or "text" (extracted into the system prompt, document_extract.extract_text).
    type = Column(String(20), nullable=False)

    message = relationship("Message", back_populates="attachments")

    @property
    def url(self) -> str:
        """For AttachmentOut.url: where the frontend fetches this file (app/routers/chat.py GET
        /attachments/{attachment_id})."""
        return f"/api/chat/attachments/{self.id}"
