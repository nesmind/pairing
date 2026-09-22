"""
CRUD for a user's own Notes, including which of their conversations each
note is pinned to (as persona/rules/skill — see
app/services/note_service.py for how a pin actually reaches the model,
and app/services/chat_service.py for where that happens).

A note's text and its pins always save together in one request (see
note_service.sync_pins) rather than through a separate pin/unpin
endpoint per conversation — the Notes page's single Save button sends
the note's current title/content plus the full set of conversations it
should be pinned to, and the whole thing is reconciled server-side in
one go.

Every endpoint requires a logged-in user and every query is scoped to
that user's own notes/conversations — the same ownership pattern as
app/routers/conversations.py.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Note, NotePin, User
from app.schemas import NoteCreate, NoteOut, NoteSlotOut, NoteSlotUpdate, NoteUpdate, OkResponse
from app.services import channel_service, conversation_service, note_service, note_slot_service
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/notes", tags=["notes"])


async def _get_owned_note_or_404(db: AsyncSession, note_id: str, user: User) -> Note:
    """Mirrors app.services.conversation_service.get_accessible_conversation_or_404
    — 404 (not 403) whether the note doesn't exist or just isn't the
    caller's, so a caller can't distinguish the two and enumerate other
    users' notes."""
    note = (await db.execute(select(Note).where(Note.id == note_id, Note.owner_id == user.id))).scalar_one_or_none()
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


async def _require_slot_edit_rights(conversation, user: User) -> None:
    """The 3 slot-mutating endpoints below (update/use-default/turn-off)
    need more than read access: a personal conversation's owner always
    has full rights (unchanged from before channels existed), but a
    channel's shared conversation can only have its pinned persona/
    rules/skill changed by an admin or that channel's manager — a plain
    member can read the slots (get_slots below) but not edit them."""
    if conversation.channel_id is not None and not channel_service.can_manage_channel_conversation(conversation, user):
        raise HTTPException(
            status_code=403,
            detail="Only admins and this channel's managers can change its persona/rules/skill.",
        )


@router.get("", response_model=list[NoteOut])
async def list_notes(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Newest-first list of the current user's own notes, each with its
    pins — lets the Notes page show "pinned to <chat>" chips without a
    separate request per note."""
    notes = (
        (await db.execute(select(Note).where(Note.owner_id == user.id).order_by(Note.updated_at.desc())))
        .scalars()
        .all()
    )
    return [note_service.to_note_out(note) for note in notes]


@router.get("/slots/{conversation_id}", response_model=list[NoteSlotOut])
async def get_slots(conversation_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The 3 persona/rules/skill slots for one conversation, in a fixed
    order — powers the chat page's icons (see chat.js:
    loadPinnedBadges/openSlotEditor). `active: false` means no icon
    should show for that slot."""
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    return [
        await note_slot_service.resolve_slot(db, conversation, user, pin_type) for pin_type in note_service.NOTE_SLOTS
    ]


@router.put("/slots/{conversation_id}/{pin_type}", response_model=NoteSlotOut)
async def update_slot(
    conversation_id: str,
    pin_type: str,
    body: NoteSlotUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Edits this slot's content — see note_slot_service.update_slot_content
    for the only_this_chat=False (default: updates the shared note
    everywhere) vs. True (private per-chat fork, this endpoint's only
    behavior before that field existed) split. Also clears the slot
    from `disabled_default_notes` if it had been turned off, since
    saving new content here obviously means it's back on."""
    if pin_type not in note_service.NOTE_SLOTS:
        raise HTTPException(status_code=404, detail="Unknown slot")
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    await _require_slot_edit_rights(conversation, user)

    await note_slot_service.update_slot_content(db, conversation, user, pin_type, body.content, body.only_this_chat)

    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    if pin_type in disabled:
        disabled.discard(pin_type)
        note_slot_service.save_disabled_slots(conversation, disabled)

    await db.commit()
    return await note_slot_service.resolve_slot(db, conversation, user, pin_type)


@router.post("/slots/{conversation_id}/{pin_type}/use-default", response_model=NoteSlotOut)
async def use_default_slot(
    conversation_id: str,
    pin_type: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Reverts this slot to the shared default — removes any per-chat
    override pin and clears the slot from `disabled_default_notes` —
    regardless of which of those two states it's currently in."""
    if pin_type not in note_service.NOTE_SLOTS:
        raise HTTPException(status_code=404, detail="Unknown slot")
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    await _require_slot_edit_rights(conversation, user)

    pin = (
        await db.execute(
            select(NotePin).where(NotePin.conversation_id == conversation.id, NotePin.pin_type == pin_type),
        )
    ).scalar_one_or_none()
    if pin is not None:
        await db.delete(pin)

    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    if pin_type in disabled:
        disabled.discard(pin_type)
        note_slot_service.save_disabled_slots(conversation, disabled)

    await db.commit()
    return await note_slot_service.resolve_slot(db, conversation, user, pin_type)


@router.delete("/slots/{conversation_id}/{pin_type}", response_model=NoteSlotOut)
async def turn_off_slot(
    conversation_id: str,
    pin_type: str,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Turns this slot off for this conversation only — no icon, and it
    contributes nothing to the prompt (see
    app.services.note_service.resolve_conversation_notes) — until
    use_default_slot or update_slot brings it back."""
    if pin_type not in note_service.NOTE_SLOTS:
        raise HTTPException(status_code=404, detail="Unknown slot")
    conversation = await conversation_service.get_accessible_conversation_or_404(db, conversation_id, user)
    await _require_slot_edit_rights(conversation, user)

    pin = (
        await db.execute(
            select(NotePin).where(NotePin.conversation_id == conversation.id, NotePin.pin_type == pin_type),
        )
    ).scalar_one_or_none()
    if pin is not None:
        await db.delete(pin)

    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    disabled.add(pin_type)
    note_slot_service.save_disabled_slots(conversation, disabled)

    await db.commit()
    return await note_slot_service.resolve_slot(db, conversation, user, pin_type)


@router.post("", response_model=NoteOut)
async def create_note(
    body: NoteCreate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    note = Note(owner_id=user.id, title=body.title, content=body.content)
    db.add(note)
    await db.flush()  # assigns note.id so sync_pins can attach pins to it below
    await note_service.sync_pins(db, note, user, body.pins)
    await db.commit()
    await db.refresh(note)
    return note_service.to_note_out(note)


@router.patch("/{note_id}", response_model=NoteOut)
async def update_note(
    note_id: str,
    body: NoteUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    note = await _get_owned_note_or_404(db, note_id, user)
    if body.title is not None:
        note.title = body.title
    if body.content is not None:
        note.content = body.content
    if body.pins is not None:
        await note_service.sync_pins(db, note, user, body.pins)
    await db.commit()
    await db.refresh(note)
    return note_service.to_note_out(note)


@router.delete("/{note_id}", response_model=OkResponse)
async def delete_note(note_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    note = await _get_owned_note_or_404(db, note_id, user)
    if note.default_type is not None:
        raise HTTPException(status_code=400, detail="Default notes can't be deleted, only updated.")
    await db.delete(note)
    await db.commit()
    return OkResponse()
