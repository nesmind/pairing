"""
Everything about a user's Notes (app/models/note.py: Note, NotePin):
seeding/looking up each user's 3 protected default notes
(persona/rules/skill), resolving which notes are active for a given
conversation, turning that into part of the system prompt actually sent
to the ML engine, and the Notes page's pin-syncing/note CRUD support behind
app/routers/notes.py's endpoints. The chat page's narrower one-slot-at-
a-time view (resolving/editing a single persona/rules/skill icon for one
conversation) lives in app/services/note_slot_service.py instead —
split out purely to stay under CLAUDE.md's file-size rule. Kept separate
from app/services/document_ingest.py since none of this has anything to
do with embeddings/retrieval.
"""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models import Conversation, Note, NotePin, User
from app.schemas import NoteOut, NotePinCreate, NotePinOut
from app.services import channel_service

# Slot types, in the fixed order they're always shown/combined in — a
# plain tuple (not a set) so persona always appears first in the
# assembled prompt (see build_pinned_system_prompt) and the chat page's
# 3 icons always render in the same order.
NOTE_SLOTS: tuple[str, ...] = ("persona", "rules", "skill")

# Simple, real-world-usable starting points for each slot — editable
# (and, unlike a regular note, un-deletable) from the moment a user's
# account exists, so Notes/the chat page's 3 icons are never empty.
DEFAULT_NOTE_SEEDS: dict[str, tuple[str, str]] = {
    "persona": (
        "Persona",
        "You are a friendly, knowledgeable assistant who explains things "
        "clearly and adapts your tone to the person you're talking to.",
    ),
    "rules": (
        "Rules",
        "Never make up facts. If you're not sure about something, say so clearly instead of guessing.",
    ),
    "skill": (
        "Skill",
        "You can help with writing, summarizing, brainstorming, and answering general questions.",
    ),
}


async def seed_default_notes(db: AsyncSession, user: User) -> None:
    """Creates whichever of `user`'s 3 protected default notes don't
    already exist. Idempotent by design — safe to call for a brand-new
    user (app/services/settings_service.create_user,
    app/services/auth_service.seed_default_users) as well as, on every
    startup, every *pre-existing* user as a one-time backfill (see
    app/main.py's on_startup) for accounts that predate this feature.
    Doesn't commit — the caller decides when to."""
    existing_types = set(
        (
            await db.execute(
                select(Note.default_type).where(Note.owner_id == user.id, Note.default_type.isnot(None)),
            )
        )
        .scalars()
        .all()
    )
    for pin_type, (title, content) in DEFAULT_NOTE_SEEDS.items():
        if pin_type not in existing_types:
            db.add(Note(owner_id=user.id, title=title, content=content, default_type=pin_type))


async def get_default_note(db: AsyncSession, owner_id: str, pin_type: str) -> Note | None:
    """`owner_id`'s protected default note for one slot — the note
    app.services.chat_service falls back to when a conversation has no
    explicit override pinned for that slot (see
    resolve_conversation_notes) and the note the chat page's icon
    shows/edits before any override exists (see resolve_slot below)."""
    return (
        await db.execute(select(Note).where(Note.owner_id == owner_id, Note.default_type == pin_type))
    ).scalar_one_or_none()


async def resolve_conversation_notes(
    db: AsyncSession,
    conversation: Conversation,
    owner_id: str,
) -> dict[str, list[Note]]:
    """The *full* set of active notes per slot for one conversation —
    used for prompt-building (see build_pinned_system_prompt below).
    Any note explicitly pinned to this conversation (via the Notes page
    or the chat page's icons) is used for its slot; a slot with no
    explicit pin falls back to the user's default note for that type,
    unless it's been explicitly turned off (conversation.params.
    disabled_default_notes — see app/schemas/common.py:GenerationParams).

    Multiple explicit pins under the same slot still combine (not
    last-pinned-wins) — same behavior as before this change, just now
    also layered with the default-note fallback.
    """
    by_type: dict[str, list[Note]] = {slot: [] for slot in NOTE_SLOTS}
    pins = (
        (
            await db.execute(
                select(NotePin).options(joinedload(NotePin.note)).where(NotePin.conversation_id == conversation.id),
            )
        )
        .scalars()
        .all()
    )
    for pin in pins:
        by_type.setdefault(pin.pin_type, []).append(pin.note)

    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    for slot in NOTE_SLOTS:
        if not by_type[slot] and slot not in disabled:
            default_note = await get_default_note(db, owner_id, slot)
            if default_note:
                by_type[slot] = [default_note]
    return by_type


def build_pinned_system_prompt(notes_by_type: dict[str, list[Note]]) -> str:
    """Assembles the system prompt from active persona/rules/skill notes
    — this *is* the system prompt now (there used to also be a free-text
    per-conversation system prompt field; it was dropped once Notes
    covered the same need, so this is the only source left). Persona
    goes first since identity/tone should frame everything that follows;
    rules and skills come after as more specific, situational guidance.
    """
    parts = []
    if notes_by_type.get("persona"):
        parts.append("Persona:\n" + "\n\n".join(n.content for n in notes_by_type["persona"]))
    if notes_by_type.get("rules"):
        parts.append("Rules you must always follow:\n" + "\n\n".join(f"- {n.content}" for n in notes_by_type["rules"]))
    if notes_by_type.get("skill"):
        parts.append(
            "Skills available for this conversation:\n" + "\n\n".join(n.content for n in notes_by_type["skill"])
        )
    return "\n\n".join(parts)


async def sync_pins(db: AsyncSession, note: Note, user: User, desired: list[NotePinCreate]) -> None:
    """Makes `note`'s pins exactly match `desired`: adds missing ones,
    updates pin_type for any that changed, and removes ones no longer
    present — a full replace rather than an incremental add/remove, so
    the caller (the Notes page) can just send "here's everywhere this
    note should be pinned right now" without tracking a diff itself.

    Every target conversation is validated up front, before any change is
    made, so a bad id can't leave a partial sync — either the whole set
    applies or none of it does. A target is valid if `user` owns it
    (personal chat) or, for a channel's shared conversation, if `user`
    can manage that channel (see
    channel_service.can_manage_channel_conversation) — a plain channel
    member can't pin notes to the shared chat, only admins/managers can.
    """
    desired_by_conv = {pin.conversation_id: pin.pin_type for pin in desired}

    if desired_by_conv:
        candidates = (
            (await db.execute(select(Conversation).where(Conversation.id.in_(desired_by_conv)))).scalars().all()
        )
        allowed_ids = {
            c.id
            for c in candidates
            if c.owner_id == user.id
            or (c.channel_id is not None and channel_service.can_manage_channel_conversation(c, user))
        }
        if set(desired_by_conv) - allowed_ids:
            raise HTTPException(status_code=404, detail="Conversation not found")

    # Queried explicitly (rather than read off note.pins) because a
    # brand-new note reaches here having only been flush()ed, never
    # select()ed — its `pins` collection was never populated by a query,
    # so touching it would try an implicit lazy load, which async
    # SQLAlchemy can't do outside an awaited call (MissingGreenlet).
    existing_pins = (await db.execute(select(NotePin).where(NotePin.note_id == note.id))).scalars().all()
    existing_by_conv = {pin.conversation_id: pin for pin in existing_pins}
    for conversation_id, pin_type in desired_by_conv.items():
        existing = existing_by_conv.get(conversation_id)
        if existing is None:
            db.add(NotePin(note_id=note.id, conversation_id=conversation_id, pin_type=pin_type))
        elif existing.pin_type != pin_type:
            existing.pin_type = pin_type
    for conversation_id, pin in existing_by_conv.items():
        if conversation_id not in desired_by_conv:
            await db.delete(pin)


def to_note_out(note: Note) -> NoteOut:
    """Builds the response by hand rather than relying on NoteOut's
    from_attributes conversion: NotePinOut.conversation_title has no
    matching attribute on the NotePin ORM row itself (it's
    pin.conversation.title), so each pin is constructed explicitly here."""
    return NoteOut(
        id=note.id,
        title=note.title,
        content=note.content,
        created_at=note.created_at,
        updated_at=note.updated_at,
        is_default=note.default_type is not None,
        pins=[
            NotePinOut(
                id=pin.id,
                conversation_id=pin.conversation_id,
                conversation_title=pin.conversation.title or "New chat",
                pin_type=pin.pin_type,
            )
            for pin in note.pins
        ],
    )
