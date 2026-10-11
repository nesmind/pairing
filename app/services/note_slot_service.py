"""The chat page's 3 persona/rules/skill icons: resolving and editing one slot of one conversation. A narrower
view than note_service.py's CRUD/pin-syncing/prompt-building, split for the file-size rule. Shares
DEFAULT_NOTE_SEEDS and get_default_note with note_service.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Conversation, Note, NotePin, User
from app.schemas import NoteSlotOut
from app.services.note_service import DEFAULT_NOTE_SEEDS, default_notes_owner_id, get_default_note


def save_disabled_slots(conversation: Conversation, disabled: set[str]) -> None:
    """Reassigns conversation.params as a new dict: SQLAlchemy's plain JSON column doesn't track in-place edits
    (PATCH /api/conversations/{id} also replaces `params` wholesale).
    """
    conversation.params = {**(conversation.params or {}), "disabled_default_notes": sorted(disabled)}


async def resolve_slot(db: AsyncSession, conversation: Conversation, user: User, pin_type: str) -> NoteSlotOut:
    """The single primary note of a slot on a conversation (see NoteSlotOut vs. note_service.resolve_conversation_notes'
    combine-everything view for chat_service).

    Always returns a title/content when the user has a default note, even when the slot is off (`active=False`),
    so the icon stays (faded) and the editor has content to re-enable, not a blank textarea.
    """
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

    default_note = await get_default_note(db, default_notes_owner_id(conversation, user.id), pin_type)
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
    """The save behind PUT /api/notes/slots/{conversation_id}/{pin_type} (the slot editor's "Only for this chat"
    checkbox).

    Unchecked (default): updates `user`'s shared *default* note, visible in all their chats, and removes any
    per-chat override here so this chat reflects it at once (resolve_slot/resolve_conversation_notes fall back
    to the default when there is no pin).

    Checked: forks/edits a private note pinned only to this conversation; the pre-checkbox behavior.

    Doesn't commit; the caller does with its other changes (routers/notes.py update_slot also clears
    disabled_default_notes).
    """
    pin = (
        await db.execute(
            select(NotePin).where(NotePin.conversation_id == conversation.id, NotePin.pin_type == pin_type),
        )
    ).scalar_one_or_none()

    # A channel's notes belong to the channel: its default notes are one stable member's (default_notes_owner_id),
    # so edits always pin to the channel.
    only_this_chat = only_this_chat or conversation.channel_id is not None
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
        # Shouldn't happen (seed_default_notes guarantees one per slot), but create it rather than silently do nothing.
        default_title, _ = DEFAULT_NOTE_SEEDS[pin_type]
        default_note = Note(owner_id=user.id, title=default_title, content=content, default_type=pin_type)
        db.add(default_note)
    else:
        default_note.content = content
    if pin is not None:
        await db.delete(pin)
