"""Request/response shapes for app/routers/notes.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import PinType


class NotePinCreate(BaseModel):
    """One entry of NoteCreate/NoteUpdate `pins`: a conversation and a role. The list replaces the note's
    entire pin set (routers/notes.py _sync_pins), so the Notes page saves text and pins in one request.
    """

    conversation_id: str
    pin_type: PinType


class NoteCreate(BaseModel):
    title: str = Field("Untitled note", max_length=200)
    content: str = ""
    pins: list[NotePinCreate] = []


class NoteUpdate(BaseModel):
    """title/content optional so a save can touch one. `pins`, when present, replaces the whole pin set
    (omit to leave untouched; the Notes page always sends it).
    """

    title: str | None = Field(None, max_length=200)
    content: str | None = None
    pins: list[NotePinCreate] | None = None


class NotePinOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    # Denormalized at read time (routers/notes.py) so the Notes page shows "pinned to <chat title>" without a request
    # per pin.
    conversation_title: str
    pin_type: PinType


class NoteOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    content: str
    created_at: datetime
    updated_at: datetime
    pins: list[NotePinOut] = []
    # True for the 3 notes note_service.seed_default_notes creates (persona/rules/skill); the Notes page
    # disables Delete for them (guard in DELETE /api/notes/{note_id}).
    is_default: bool = False


class NoteSlotUpdate(BaseModel):
    """Body for PUT /api/notes/slots/{conversation_id}/{pin_type}.

    only_this_chat=False (default) updates the user's shared default note (all chats); True forks a private
    override pinned to this conversation.
    """

    content: str = ""
    only_this_chat: bool = False


class DefaultNotesEnabledConfig(BaseModel):
    """GET/PUT /api/notes/default-notes-enabled body
    (default_notes_setting.get_default_notes_enabled/set_default_notes_enabled).
    Off by default: a new chat/channel starts with all 3 default notes off (icons faded, see NoteSlotOut.active)
    until the user enables this (Settings > Account) or one slot for one chat.
    """

    enabled: bool = False


class NoteSlotOut(BaseModel):
    """One of a conversation's 3 persona/rules/skill slots for the chat page icons (GET/PUT/DELETE
    /api/notes/slots/{conversation_id}/...). Simpler than NoteOut.pins: one primary note per slot, vs. the
    many-per-type pinning used for prompt building (note_service.resolve_conversation_notes).
    """

    pin_type: PinType
    active: bool  # False = off for this chat: icon faded, not hidden
    is_override: bool  # True = private per-chat fork, not the shared default
    # Set whenever the user has a note for this slot, even when active=False, so a faded icon's editor has content to
    # show.
    note_id: str | None = None
    title: str | None = None
    content: str | None = None
