"""A user's Notes (models/note.py: Note, NotePin): seeding and looking up each user's 3 protected default notes
(persona/rules/skill), resolving active notes per conversation, turning them into part of the engine's system
prompt, and the pin-syncing/CRUD behind routers/notes.py. The chat page's one-slot view is
note_slot_service.py (split for the file-size rule). Unrelated to document_ingest.py (embeddings/retrieval).

seed_initial_disabled_notes turns the per-user "default notes enabled" preference (Settings > Account, in
default_notes_setting.py) into a new chat/channel's initial disabled_default_notes. The getter/setter live
there only to avoid a circular import (this module imports channel_service).
"""

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models import Conversation, Note, NotePin, User
from app.schemas import NoteOut, NotePinCreate, NotePinOut
from app.services import channel_service
from app.services.default_notes_setting import get_default_notes_enabled

# Slot types in fixed order, a tuple (not a set) so persona comes first in the prompt (build_pinned_system_prompt)
# and the chat page's 3 icons keep their order.
NOTE_SLOTS: tuple[str, ...] = ("persona", "rules", "skill")

# Usable starting points per slot: editable but un-deletable from account creation, so Notes and the chat icons are
# never empty.
DEFAULT_NOTE_SEEDS: dict[str, tuple[str, str]] = {
    "persona": (
        "Persona",
        "Friendly and clear; adapts tone to the user.",
    ),
    "rules": (
        "Rules",
        "Never make up facts; say so if unsure.",
    ),
    "skill": (
        "Skill",
        "Writing, summarizing, brainstorming, general Q&A.",
    ),
}


async def seed_default_notes(db: AsyncSession, user: User) -> None:
    """Creates whichever of `user`'s 3 protected default notes are missing. Idempotent: safe for a new user
    (settings_service.create_user, auth_service.seed_default_users) and, at every startup, a backfill for
    pre-existing users (app/main.py on_startup). Doesn't commit.
    """
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


async def seed_initial_disabled_notes(db: AsyncSession, user: User, params: dict) -> dict:
    """Mutates and returns `params` (a new conversation's/channel's initial GenerationParams from
    settings_service.get_default_params) to start with all 3 default notes off when `user`'s
    default_notes_setting.get_default_notes_enabled is off; untouched (every slot on) when on.

    Seeds `disabled_default_notes` once at creation rather than having resolve_conversation_notes check this
    global setting live: the chat icons' "Enable"/"Turn off" already read/write that per-conversation list
    (routers/notes.py slot endpoints), and a live global gate would defeat "Enable" while the setting is off.
    """
    if not await get_default_notes_enabled(db, user):
        params["disabled_default_notes"] = list(NOTE_SLOTS)
    return params


def default_notes_owner_id(conversation: Conversation, user_id: str) -> str:
    """Whose default notes a conversation falls back to. Personal: the user's own. Channel: one stable member,
    since members' own defaults would make a different system prompt per speaker and discard the prompt cache.
    The first admin in the channel, else its first manager, else any member (lowest id).
    """
    channel = conversation.channel
    if channel is None or not channel.members:
        return user_id
    members = channel.members
    admins = [m.user_id for m in members if m.user is not None and m.user.role == "admin"]
    managers = [m.user_id for m in members if m.is_manager]
    return min(admins or managers or [m.user_id for m in members])


async def freeze_channel_notes(
    db: AsyncSession, conversation: Conversation, label: str, source_user_id: str, replace: bool = False
) -> None:
    """Configures a channel's notes once: copies `source_user_id`'s default notes into the channel as its own pinned
    notes (each slot not turned off), so its system prompt no longer depends on later edits or who is asking,
    keeping the prompt cache valid. Done at creation and reset; `replace` swaps already-pinned notes (clean
    start), else they stay. Doesn't commit.
    """
    disabled = set((conversation.params or {}).get("disabled_default_notes", []))
    for slot in NOTE_SLOTS:
        if slot in disabled:
            continue
        pin = (
            await db.execute(
                select(NotePin)
                .options(joinedload(NotePin.note))
                .where(NotePin.conversation_id == conversation.id, NotePin.pin_type == slot)
            )
        ).scalar_one_or_none()
        if pin is not None and not replace:
            continue
        default_note = await get_default_note(db, source_user_id, slot)
        if default_note is None:
            continue
        if pin is not None:
            old_note = pin.note
            await db.delete(pin)
            if old_note.default_type is None:  # a per-chat copy nobody else uses
                others = await db.scalar(
                    select(func.count()).select_from(NotePin).where(NotePin.note_id == old_note.id)
                )
                if others == 1:
                    await db.delete(old_note)
            await db.flush()
        title, _ = DEFAULT_NOTE_SEEDS[slot]
        note = Note(
            owner_id=source_user_id, title=f"{title} — {label}", content=default_note.content, default_type=None
        )
        db.add(note)
        await db.flush()
        db.add(NotePin(note_id=note.id, conversation_id=conversation.id, pin_type=slot))


async def get_default_note(db: AsyncSession, owner_id: str, pin_type: str) -> Note | None:
    """`owner_id`'s protected default note for a slot: chat_service's fallback when a conversation has no explicit
    pin (resolve_conversation_notes) and what the chat icon shows before any override (resolve_slot).
    """
    return (
        await db.execute(select(Note).where(Note.owner_id == owner_id, Note.default_type == pin_type))
    ).scalar_one_or_none()


async def resolve_conversation_notes(
    db: AsyncSession,
    conversation: Conversation,
    owner_id: str,
) -> dict[str, list[Note]]:
    """The *full* set of active notes per slot for one conversation, for prompt building
    (build_pinned_system_prompt). Notes pinned to this conversation (Notes page or chat icons) are used for their
    slot; a slot with no pin falls back to the user's default note, unless turned off
    (conversation.params.disabled_default_notes, schemas/common.py GenerationParams). Several pins in one slot still
    combine.
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
    owner_id = default_notes_owner_id(conversation, owner_id)
    for slot in NOTE_SLOTS:
        if not by_type[slot] and slot not in disabled:
            default_note = await get_default_note(db, owner_id, slot)
            if default_note:
                by_type[slot] = [default_note]
    return by_type


def build_pinned_system_prompt(notes_by_type: dict[str, list[Note]]) -> str:
    """Assembles the system prompt from active persona/rules/skill notes; the only source (the free-text
    per-conversation system prompt was dropped when Notes covered it). Persona first, as identity and tone
    frame the rest; rules and skills follow as situational guidance.
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
    """Makes `note`'s pins exactly match `desired`: adds missing, updates changed pin_type, removes the rest. A full
    replace, so the Notes page sends where the note should be pinned without a diff.

    Every target is validated first, so a bad id can't leave a partial sync. A target is valid if `user` owns it
    (personal) or can manage its channel (channel_service.can_manage_channel_conversation): plain members can't pin to
    the shared chat.
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

    # Queried explicitly, not via note.pins: a new note was only flushed, so `pins` was never populated and
    # touching it would lazy-load, which async SQLAlchemy can't do (MissingGreenlet).
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
    """Built by hand, not via NoteOut's from_attributes: NotePinOut.conversation_title is pin.conversation.title,
    with no matching attribute on the NotePin row.
    """
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
