"""Note + NotePin models — see app/services/note_service.py for how a
pin actually reaches the model's system prompt, and
app/routers/notes.py/app/routers/chat.py for where that's used."""

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Note(Base):
    """A user-authored note. On its own it's just text (see the Notes
    page), but tagging it with a NotePin lets its content actually steer
    a specific conversation's replies — see app/services/note_service.py
    and app/services/chat_service.py, which pull in whatever's pinned to
    the conversation being talked to."""

    __tablename__ = "notes"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False, default="Untitled note")
    content = Column(Text, nullable=False, default="")
    # "persona" | "rules" | "skill" for exactly the 3 notes
    # app.services.note_service.seed_default_notes creates for every
    # user; NULL for every other (regular, user-created) note.
    # Delete-protected in app/routers/notes.py — a user can still create
    # as many additional regular notes as they like, just not remove
    # these 3.
    default_type = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # lazy="selectin" — see app/models/conversation.py's
    # Conversation.messages for why the app defaults accessed
    # relationships to this under the async session.
    pins = relationship("NotePin", back_populates="note", cascade="all, delete-orphan", lazy="selectin")


class NotePin(Base):
    """Attaches one Note to one Conversation under a specific role. A
    conversation can have several pins (even several of the same
    pin_type — see app.services.note_service.build_pinned_system_prompt,
    which combines rather than picks one), but the same note can only be
    pinned to a given conversation once — pinning it again just changes
    its pin_type instead of creating a second row (see
    app/routers/notes.py's upsert logic)."""

    __tablename__ = "note_pins"
    __table_args__ = (UniqueConstraint("note_id", "conversation_id", name="uq_note_pin_target"),)

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    note_id = Column(String(ID_LEN), ForeignKey("notes.id"), nullable=False)
    conversation_id = Column(String(ID_LEN), ForeignKey("conversations.id"), nullable=False)
    # "persona" | "rules" | "skill" — see app/services/note_service.py
    # for how each is phrased differently in the assembled system prompt.
    pin_type = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    # lazy="selectin" on both sides — app/routers/notes.py reads
    # pin.note.title/content and pin.conversation.title directly; see
    # app/models/conversation.py's Conversation.messages for why this is
    # the app's default loading strategy for accessed relationships.
    note = relationship("Note", back_populates="pins", lazy="selectin")
    conversation = relationship("Conversation", back_populates="note_pins", lazy="selectin")
