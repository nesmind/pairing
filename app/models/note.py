"""Note and NotePin models (injection into the system prompt: app/services/note_service.py)."""

from sqlalchemy import Column, DateTime, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class Note(Base):
    """A user-authored note; a NotePin makes it steer a conversation's replies (note_service.py, chat_service.py)."""

    __tablename__ = "notes"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    owner_id = Column(String(ID_LEN), ForeignKey("users.id"), nullable=False)
    title = Column(String(255), nullable=False, default="Untitled note")
    content = Column(Text, nullable=False, default="")
    # "persona" | "rules" | "skill" for the 3 notes note_service.seed_default_notes creates per user; NULL for
    # regular notes. Those 3 are delete-protected (app/routers/notes.py).
    default_type = Column(String(20), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # lazy="selectin" (see Conversation.messages).
    pins = relationship("NotePin", back_populates="note", cascade="all, delete-orphan", lazy="selectin")


class NotePin(Base):
    """Attaches a Note to a Conversation under a role. A conversation may have several pins (even of one
    pin_type; note_service.build_pinned_system_prompt combines them), but a note pins once: re-pinning
    changes its pin_type (app/routers/notes.py upsert).
    """

    __tablename__ = "note_pins"
    __table_args__ = (UniqueConstraint("note_id", "conversation_id", name="uq_note_pin_target"),)

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    note_id = Column(String(ID_LEN), ForeignKey("notes.id"), nullable=False)
    conversation_id = Column(String(ID_LEN), ForeignKey("conversations.id"), nullable=False)
    # "persona" | "rules" | "skill" (phrased differently in the system prompt, see note_service.py).
    pin_type = Column(String(20), nullable=False)
    created_at = Column(DateTime, default=utcnow)

    # lazy="selectin" on both sides: app/routers/notes.py reads pin.note and pin.conversation.
    note = relationship("Note", back_populates="pins", lazy="selectin")
    conversation = relationship("Conversation", back_populates="note_pins", lazy="selectin")
