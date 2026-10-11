"""CRUD for a user's own Notes, including which conversations each is pinned to as persona/rules/skill
(app/services/note_service.py, chat_service.py).

A note's text and pins save together in one request (note_service.sync_pins): the Notes page's single
Save sends the title/content and the full pin set. Scoped to the user's own notes, like conversations.py.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Note, NotePin, User
from app.schemas import (
    DefaultNotesEnabledConfig,
    NoteCreate,
    NoteOut,
    NoteSlotOut,
    NoteSlotUpdate,
    NoteUpdate,
    OkResponse,
)
from app.services import channel_service, conversation_service, default_notes_setting, note_service, note_slot_service
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/notes", tags=["notes"])


async def _get_owned_note_or_404(db: AsyncSession, note_id: str, user: User) -> Note:
    """404 (not 403) for missing or foreign notes, so ids can't be enumerated
    (as conversation_service.get_accessible_conversation_or_404).
    """
    note = (await db.execute(select(Note).where(Note.id == note_id, Note.owner_id == user.id))).scalar_one_or_none()
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


async def _require_slot_edit_rights(conversation, user: User) -> None:
    """Slot edits need more than read access: a personal chat's owner has full rights; a channel's slots
    change only by an admin or its manager (members can read via get_slots). Nobody can once the chat has a
    message: persona/rules/skill start the prompt, so changing them re-reads the whole conversation.
    """
    if conversation_service.visible_messages(conversation):
        raise HTTPException(
            status_code=409, detail="Persona, rules and skill can only change before the first message."
        )
    if conversation.channel_id is not None and not channel_service.can_manage_channel_conversation(conversation, user):
        raise HTTPException(
            status_code=403,
            detail="Only admins and this channel's managers can change its persona/rules/skill.",
        )


@router.get("/default-notes-enabled", response_model=DefaultNotesEnabledConfig)
async def get_default_notes_enabled_setting(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The user's own preference (Settings > Account) for whether new chats/channels start with persona/rules/skill
    defaults on. Open to all users; it only affects their own chats.
    """
    return DefaultNotesEnabledConfig(enabled=await default_notes_setting.get_default_notes_enabled(db, user))


@router.put("/default-notes-enabled", response_model=DefaultNotesEnabledConfig)
async def set_default_notes_enabled_setting(
    body: DefaultNotesEnabledConfig,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the user's own preference; never affects others or existing conversations
    (seeded once at creation, see note_service.seed_initial_disabled_notes).
    """
    await default_notes_setting.set_default_notes_enabled(db, user, body.enabled)
    return body


@router.get("", response_model=list[NoteOut])
async def list_notes(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The user's notes, newest first, with pins (avoids a request per note for "pinned to <chat>" chips)."""
    notes = (
        (await db.execute(select(Note).where(Note.owner_id == user.id).order_by(Note.updated_at.desc())))
        .scalars()
        .all()
    )
    return [note_service.to_note_out(note) for note in notes]


@router.get("/slots/{conversation_id}", response_model=list[NoteSlotOut])
async def get_slots(conversation_id: str, db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The 3 persona/rules/skill slots of a conversation in fixed order, for chat.js's icons
    (loadPinnedBadges/openSlotEditor). `active: false` = no icon.
    """
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
    """Edits the slot's content (note_slot_service.update_slot_content): only_this_chat=False (default) updates
    the shared note, True makes a per-chat fork. Also clears the slot from `disabled_default_notes`.
    """
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
    """Reverts the slot to the shared default: removes any per-chat pin and clears `disabled_default_notes`."""
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
    """Turns the slot off for this conversation only: no icon, nothing in the prompt
    (note_service.resolve_conversation_notes) until use_default_slot or update_slot.
    """
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
    await db.flush()  # assigns note.id for sync_pins
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
