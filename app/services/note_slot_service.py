"""
The chat page's 3 persona/rules/skill icons: resolving and editing one
slot at a time for a single conversation — a narrower, simpler view than
app/services/note_service.py's general Notes-page CRUD/pin-syncing/
prompt-building, split out purely to stay under CLAUDE.md's file-size
rule. Depends on note_service for the shared constants/lookups
(DEFAULT_NOTE_SEEDS, get_default_note) both files need.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Note, NotePin, User
from app.schemas import NoteSlotOut
from app.services.note_service import DEFAULT_NOTE_SEEDS, get_default_note


def save_disabled_slots(conversation: Conversation, disabled: set[str]) -> None:
    """Reassigns conversation.params as a whole new dict (rather than
    mutating the existing one in place) — SQLAlchemy's plain JSON column
    type doesn't track in-place mutation of the dict it holds, only
    reassignment, the same reason PATCH /api/conversations/{id} already
    replaces `params` wholesale instead of editing individual keys."""
    conversation.params = {**(conversation.params or {}), "disabled_default_notes": sorted(disabled)}


async def resolve_slot(db: AsyncSession, conversation: Conversation, user: User, pin_type: str) -> NoteSlotOut:
    """The single primary note for one slot on one conversation — see
    NoteSlotOut's docstring for how this differs from the general,
    combine-everything view note_service.resolve_conversation_notes
    gives app/services/chat_service.py.

    Always returns a title/content when the user has a default note for
    this slot, even if the slot is currently turned off (`active=False`)
    — so the chat page's icon can stay in place (just faded) instead of
    disappearing, and its editor panel always has something sensible to
    show/re-enable rather than a blank textarea."""
    pin = (
        await db.execute(
            select(NotePin).where(NotePin.conversation_id == conversation.id, NotePin.pin_type == pin_type),
        )
    ).scalar_one_or_none()
    if pin is not None:
        return NoteSlotOut(
            pin_type=pin_type,
            active=True,
            is_override=True,
            note_id=pin.note.id,
            title=pin.note.title,
            content=pin.note.content,
        )

    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    is_active = pin_type not in disabled

    default_note = await get_default_note(db, user.id, pin_type)
    if default_note is None:
        return NoteSlotOut(pin_type=pin_type, active=is_active, is_override=False)
    return NoteSlotOut(
        pin_type=pin_type,
        active=is_active,
        is_override=False,
        note_id=default_note.id,
        title=default_note.title,
        content=default_note.content,
    )


async def update_slot_content(
    db: AsyncSession,
    conversation: Conversation,
    user: User,
    pin_type: str,
    content: str,
    only_this_chat: bool,
) -> None:
    """The save behind PUT /api/notes/slots/{conversation_id}/{pin_type}
    — chat page's slot editor, gated by its "Only for this chat"
    checkbox (only_this_chat).

    Unchecked (the default): updates `user`'s shared *default* note for
    this slot directly, so the new content is visible in every one of
    their chats, not just this one — and removes any per-chat override
    already pinned here, so this conversation immediately reflects the
    new global content instead of silently keeping stale pinned text
    underneath it (resolve_slot/note_service.resolve_conversation_notes
    already fall back to the default note once there's no pin, so
    nothing else needs to change for that).

    Checked: forks/edits a private note pinned to *only* this
    conversation, never touching the shared default or any other chat
    — this endpoint's only behavior before the checkbox existed.

    Doesn't commit — the caller does, alongside whatever else it also
    saves in the same request (see app/routers/notes.py:update_slot,
    which also clears disabled_default_notes)."""
    pin = (
        await db.execute(
            select(NotePin).where(NotePin.conversation_id == conversation.id, NotePin.pin_type == pin_type),
        )
    ).scalar_one_or_none()

    if only_this_chat:
        if pin is not None:
            pin.note.content = content
        else:
            default_title, _ = DEFAULT_NOTE_SEEDS[pin_type]
            chat_label = conversation.channel.name if conversation.channel_id else (conversation.title or "this chat")
            note = Note(
                owner_id=user.id,
                title=f"{default_title} — {chat_label}",
                content=content,
                default_type=None,
            )
            db.add(note)
            await db.flush()
            db.add(NotePin(note_id=note.id, conversation_id=conversation.id, pin_type=pin_type))
        return

    default_note = await get_default_note(db, user.id, pin_type)
    if default_note is None:
        # Shouldn't normally happen — seed_default_notes guarantees one
        # per slot for every user — but fail safe (create it) rather
        # than silently doing nothing.
        default_title, _ = DEFAULT_NOTE_SEEDS[pin_type]
        default_note = Note(owner_id=user.id, title=default_title, content=content, default_type=pin_type)
        db.add(default_note)
    else:
        default_note.content = content
    if pin is not None:
        await db.delete(pin)
