"""Conversation + Message models. Kept in one file since a Message never
exists without its parent Conversation and the two are always reasoned
about together (see app/services/chat_service.py,
app/services/conversation_service.py)."""

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Text
from sqlalchemy.orm import relationship

from app.config import DEFAULT_GENERATION_PARAMS
from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Conversation(Base):
    """One chat session (what shows up as one entry in the sidebar)."""

    __tablename__ = "conversations"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Whose chat this is — every conversation query in
    # app/services/conversation_service.py and app/services/chat_service.py
    # filters by this, so one user never sees another's chats. Nullable
    # at the database level for two reasons: the one-time migration of
    # conversations that existed before multi-user support (see
    # app/database.py:init_db), and — the only ongoing case — a channel
    # conversation (channel_id set below), which has no single owner.
    # Exactly one of owner_id/channel_id is set for any row the
    # application creates.
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # Set instead of owner_id when this is a channel's shared thread (see
    # app/models/channel.py) — access is then gated by channel
    # membership rather than ownership (see
    # app.services.conversation_service.get_accessible_conversation_or_404).
    # Nullable + unique: at most one conversation per channel.
    channel_id = Column(String(ID_LEN), ForeignKey("channels.id"), nullable=True, unique=True)
    # Short auto-generated label shown in the sidebar (see
    # app/services/chat_service.py: maybe_generate_title). Falls back to
    # "New chat" until the first exchange completes.
    title = Column(String(255), default="New chat")
    # Which Ollama model this conversation talks to. Stored per-
    # conversation (not just globally) so old chats keep using the model
    # they were started with even if the user's default changes later.
    model = Column(String(255), nullable=True)
    # The full set of adjustable generation parameters (temperature,
    # top_p, rag_top_k, ...) for this conversation,
    # stored as a JSON blob. Starts as a copy of the global defaults and
    # is overwritten wholesale whenever the user saves the Settings page
    # for this conversation.
    params = Column(JSON, default=lambda: dict(DEFAULT_GENERATION_PARAMS))
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # One conversation has many messages. `cascade="all, delete-orphan"`
    # means deleting a conversation also deletes its messages — no
    # orphaned rows left behind. `lazy="selectin"` (rather than the
    # default lazy-load-on-attribute-access) so `conversation.messages`
    # is safe to read under the app's async session without a separate
    # eager-load at every call site — see app/database.py's docstring on
    # why this app defaults every commonly-accessed relationship to
    # selectin loading.
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin",
    )
    # A conversation being deleted should take its pins with it — a
    # NotePin without a live conversation is meaningless (see NotePin),
    # so this cascades the same way messages does.
    note_pins = relationship(
        "NotePin",
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    # The channel this conversation belongs to, if any (see channel_id
    # above). No cascade here — Channel.conversation (the other side of
    # this relationship) already owns the delete-orphan cascade.
    channel = relationship("Channel", back_populates="conversation", lazy="selectin")


class Message(Base):
    """One turn in a conversation — either from the user or the model."""

    __tablename__ = "messages"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    conversation_id = Column(String(ID_LEN), ForeignKey("conversations.id"), nullable=False)
    # "user", "assistant", or "system" — mirrors Ollama's chat message
    # roles directly so history can be forwarded to Ollama unmodified.
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    # Filenames of the documents whose chunks were injected into context
    # for this reply, if RAG contributed to it (see
    # app/services/document_service.py). Stored as a JSON list so the
    # "Sources" note under a reply survives a page reload instead of
    # only existing for the live stream. Empty/null for user messages
    # and for assistant replies that didn't use RAG.
    sources = Column(JSON, nullable=True)
    # Who actually typed a "user"-role message (see
    # app.services.chat_service.build_reply_stream) — null for
    # "assistant"/"system" messages, which have no human sender. Only
    # meaningful in the UI for a channel's shared conversation, where
    # more than one member can post — see app/routers/conversations.py's
    # MessageOut.sender_display_name. A personal chat only ever has one
    # human participant (its owner), so nothing there needs a per-
    # message sender label.
    sender_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=True)
    # The "user"-role Message this assistant reply was generated for —
    # set once, when reply_generation_service.stream_reply creates the
    # placeholder, and null for every "user"/"system" message (nothing
    # points *from* those). The only consumer today is
    # conversation_service.delete_message: deleting a user message whose
    # reply is still "streaming" cancels and removes that reply too,
    # found via this column rather than inferred from timestamps (fragile
    # once a channel's other members can interleave their own plain
    # messages — see build_reply_stream's ask_ai=False path — between a
    # question and its still-generating answer).
    reply_to_message_id = Column(String(ID_LEN), ForeignKey("messages.id"), nullable=True)
    # "streaming" while an assistant reply is still being generated in
    # the background (see app.services.reply_generation_service, which
    # sets this for a personal chat and a channel's shared conversation
    # alike), "complete" once it's finished (the only value any message
    # created before this column existed ever has), "error" if generation
    # failed partway through (`content` still holds whatever was
    # generated before the failure, not necessarily empty), or "deleted"
    # if a channel admin/manager removed it (see
    # app.services.conversation_service.delete_message) — soft-deleted
    # rather than an actual row removal so an already-open poll can still
    # see the transition and remove a bubble it already rendered; every
    # other read of the history (app.services.conversation_service.visible_messages)
    # filters these out entirely.
    status = Column(String(20), nullable=False, default="complete")
    # The real reason a status="error" message failed (e.g. Matricxon's own "not enough memory to load ...: need
    # ~12.9GB, only 10.9GB available" — see app.services.matricxon_client._error_detail_from_body) — set once, by
    # app.services.reply_termination_service.mark_error, and null for every non-error message. Before this
    # column existed, that text only ever reached whoever was watching the live SSE stream at the exact moment
    # it happened (see reply_broadcast_service.publish_error) — reopening the conversation later, or any other
    # viewer who wasn't actively watching, saw only chat.js's generic REPLY_FAILED_NOTICE with no way to learn
    # what actually went wrong. MessageOut.error_message (from_attributes picks this up automatically) is what
    # lets a reload show the same real detail the live view does.
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    # Bumped on every content update — unlike created_at, this changes
    # again each time a streaming reply's content is flushed (see
    # reply_generation_service), which is what lets
    # conversation_service.get_messages_since tell a poller "this
    # message changed" without needing a separate change-log table.
    # Nullable at the database level (added via migration to existing
    # rows with no backfilled value) but always populated by the ORM
    # default/onupdate below for every row the app itself ever writes.
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    conversation = relationship("Conversation", back_populates="messages")
    # A "user"-role message can carry several files now — up to
    # app.config.MAX_ATTACHMENT_DOCUMENTS documents and
    # MAX_ATTACHMENT_IMAGES image, combined (see
    # app.services.chat_attachment_service.save_attachments) — never
    # populated for an "assistant"/"system" message. `lazy="selectin"`
    # for the same reason as Conversation.messages above: safe to read
    # under the async session with no extra query at every call site.
    # `cascade="all, delete-orphan"` means deleting a Message (or its
    # parent Conversation, which cascades to Message already) also
    # deletes its attachment rows — the files themselves are removed
    # separately by chat_attachment_service, which has no idea a DB
    # cascade exists.
    attachments = relationship(
        "MessageAttachment",
        back_populates="message",
        cascade="all, delete-orphan",
        order_by="MessageAttachment.filename",
        lazy="selectin",
    )
    # lazy="selectin" — see Conversation.messages above for why the app
    # defaults accessed relationships to this under the async session;
    # sender_display_name below reads message.sender.username/first_name/
    # last_name directly.
    sender = relationship("User", lazy="selectin")

    @property
    def sender_display_name(self) -> str | None:
        """Powers MessageOut.sender_display_name — the channel chat
        sender label (see chat.js: bubbleFor): the sender's first/last
        name (whichever are set, space-separated) with their username
        after it in parentheses, e.g. "Alice Smith (asmith)" — or just
        the bare username if neither name is set. Null for a message
        with no sender_id. A plain Python property (not a mapped column)
        since Pydantic's from_attributes mode reads it via getattr()
        just like a real attribute, and this needs to combine several
        fields rather than expose any one of them directly."""
        if not self.sender_id:
            return None
        full_name = " ".join(part for part in (self.sender.first_name, self.sender.last_name) if part)
        return f"{full_name} ({self.sender.username})" if full_name else self.sender.username

    @property
    def sender_avatar_url(self) -> str | None:
        """Powers MessageOut.sender_avatar_url — the small round avatar
        chat.js renders beside every "user"-role bubble (personal chat
        and channel alike, unlike sender_display_name's label which the
        frontend only shows in a channel — see chat.js: bubbleFor). Null
        with no sender, or when the sender has never uploaded a picture;
        sender_initials below is always usable as the fallback in either
        case."""
        return self.sender.avatar_url if self.sender_id else None

    @property
    def sender_initials(self) -> str | None:
        """Powers MessageOut.sender_initials — see sender_avatar_url's
        own docstring for when the frontend actually renders this (any
        "user"-role message, not just a channel one). Null only when
        there's no sender at all (an "assistant"/"system" message); see
        User.initials for the actual fallback text."""
        return self.sender.initials if self.sender_id else None


class MessageAttachment(Base):
    """One file attached to a "user"-role message — a message can carry
    several now (see app.config.MAX_ATTACHMENT_DOCUMENTS/
    MAX_ATTACHMENT_IMAGES and app.services.chat_attachment_service's own
    per-type caps)."""

    __tablename__ = "message_attachments"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    message_id = Column(String(ID_LEN), ForeignKey("messages.id"), nullable=False)
    # Relative to app.config.ATTACHMENTS_DIR, never served directly (see
    # app/routers/chat.py's GET /attachments/{attachment_id}, the only
    # reader of it).
    path = Column(String(500), nullable=False)
    # The original, sanitized name for display.
    filename = Column(String(255), nullable=False)
    # "image" (sent to a vision model as part of the request) or "text"
    # (its extracted content folded into the system prompt instead — see
    # document_extract.extract_text).
    type = Column(String(20), nullable=False)

    message = relationship("Message", back_populates="attachments")

    @property
    def url(self) -> str:
        """Powers AttachmentOut.url — the only URL the frontend ever
        needs to fetch this file (see app/routers/chat.py's GET
        /attachments/{attachment_id})."""
        return f"/api/chat/attachments/{self.id}"
