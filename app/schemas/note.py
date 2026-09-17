"""Request/response shapes for app/routers/notes.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import PinType


class NotePinCreate(BaseModel):
    """One entry in NoteCreate/NoteUpdate's `pins` list — a conversation
    to pin the note to, and under which role. A note can be pinned to
    any number of conversations at once (see
    app/routers/notes.py's _sync_pins, which replaces a note's *entire*
    pin set with whatever list is sent, rather than adding one pin per
    request) — this is what lets the Notes page save a note's text and
    its pins together, in one request, instead of a separate round trip
    per pin."""

    conversation_id: str
    pin_type: PinType


class NoteCreate(BaseModel):
    title: str = Field("Untitled note", max_length=200)
    content: str = ""
    pins: list[NotePinCreate] = []


class NoteUpdate(BaseModel):
    """title/content are optional so a save can touch just one of them
    without needing to resend the other. `pins`, when present, *replaces*
    the note's whole pin set (omit it entirely to leave pins untouched —
    only the Notes page's combined save always sends it)."""

    title: str | None = Field(None, max_length=200)
    content: str | None = None
    pins: list[NotePinCreate] | None = None


class NotePinOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    conversation_id: str
    # Denormalized in at read time (see app/routers/notes.py) purely so
    # the Notes page can show "pinned to <chat title>" without a second
    # round-trip per pin.
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
    # True for exactly the 3 notes app.services.note_service.seed_default_notes
    # creates (persona/rules/skill) — the Notes page uses this to
    # hide/disable Delete for these rows (see the matching guard in
    # DELETE /api/notes/{note_id}).
    is_default: bool = False


class NoteSlotUpdate(BaseModel):
    """Body for PUT /api/notes/slots/{conversation_id}/{pin_type}.

    only_this_chat defaults to False — the default way to edit a slot
    from the chat page updates the user's shared default note for that
    slot (visible in every chat), not just this one. Set it True to
    instead fork/edit a private override pinned to only this
    conversation, same as this endpoint's only behavior before this
    field existed."""

    content: str = ""
    only_this_chat: bool = False


class NoteSlotOut(BaseModel):
    """One of a conversation's 3 persona/rules/skill slots, as shown by
    the chat page's icons (see GET/PUT/DELETE
    /api/notes/slots/{conversation_id}/...). A separate, simpler view
    than NoteOut.pins: exactly one primary note per slot (or none), vs.
    the general many-notes-per-type pinning the Notes page still
    supports for prompt-building (see
    app.services.note_service.resolve_conversation_notes)."""

    pin_type: PinType
    active: bool  # False = turned off for this chat — icon shows faded, not hidden
    is_override: bool  # True = a private per-chat fork, not the shared default
    # Populated whenever the user has a note for this slot (their default,
    # or a per-chat override) — even while active=False, so a faded icon's
    # editor still has real content to show/re-enable instead of a blank box.
    note_id: str | None = None
    title: str | None = None
    content: str | None = None
