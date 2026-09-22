"""Unit tests for app/services/note_slot_service.py — the chat page's
one-slot-at-a-time view: resolving a slot (default vs. override vs.
turned-off), and editing it via the "Only for this chat" checkbox split
(update_slot_content)."""

import pytest
from sqlalchemy import select

from app.models import Conversation, Note, NotePin
from app.schemas import ChannelCreate
from app.services import channel_service, note_slot_service


@pytest.mark.asyncio
async def test_save_disabled_slots_replaces_params_dict(db, user):
    """SQLAlchemy's JSON column only tracks reassignment, not in-place
    mutation — this test would still pass with a buggy in-place
    `.update()` implementation on sqlite's in-memory session, but
    documents the contract this function must uphold in production
    (MySQL/Postgres JSON columns behave the same way)."""
    conversation = Conversation(owner_id=user.id, params={"temperature": 0.8})
    note_slot_service.save_disabled_slots(conversation, {"persona"})
    assert conversation.params["disabled_default_notes"] == ["persona"]
    assert conversation.params["temperature"] == 0.8  # untouched


@pytest.mark.asyncio
async def test_resolve_slot_falls_back_to_default_note(db, user):
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    default_note = Note(owner_id=user.id, title="Persona", content="Be nice.", default_type="persona")
    db.add(default_note)
    await db.commit()
    await db.refresh(conversation)

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")

    assert slot.active is True
    assert slot.is_override is False
    assert slot.content == "Be nice."


@pytest.mark.asyncio
async def test_resolve_slot_respects_turned_off_default(db, user):
    conversation = Conversation(owner_id=user.id, params={"disabled_default_notes": ["persona"]})
    db.add(conversation)
    db.add(Note(owner_id=user.id, title="Persona", content="Be nice.", default_type="persona"))
    await db.commit()
    await db.refresh(conversation)

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")

    assert slot.active is False
    # Content still comes through so a faded icon's editor isn't blank.
    assert slot.content == "Be nice."


@pytest.mark.asyncio
async def test_resolve_slot_prefers_explicit_pin_over_default(db, user):
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    db.add(Note(owner_id=user.id, title="Default persona", content="Default.", default_type="persona"))
    override_note = Note(owner_id=user.id, title="Override", content="Custom persona.")
    db.add(override_note)
    await db.commit()
    await db.refresh(conversation)
    await db.refresh(override_note)

    db.add(NotePin(note_id=override_note.id, conversation_id=conversation.id, pin_type="persona"))
    await db.commit()

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")

    assert slot.is_override is True
    assert slot.content == "Custom persona."


@pytest.mark.asyncio
async def test_update_slot_content_global_updates_the_default_note(db, user):
    """The default behavior (checkbox unchecked): updates the shared
    default note directly, not a new per-chat note."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    default_note = Note(owner_id=user.id, title="Persona", content="Old.", default_type="persona")
    db.add(default_note)
    await db.commit()
    await db.refresh(conversation)

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "New global content.", only_this_chat=False
    )
    await db.commit()

    await db.refresh(default_note)
    assert default_note.content == "New global content."

    all_notes = (await db.execute(select(Note).where(Note.owner_id == user.id))).scalars().all()
    assert len(all_notes) == 1  # no new note forked

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.is_override is False
    assert slot.content == "New global content."


@pytest.mark.asyncio
async def test_update_slot_content_global_removes_an_existing_override(db, user):
    """Editing globally while this chat currently has its own override
    must drop that override — otherwise this chat would keep showing
    stale pinned content instead of the newly-updated global default."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    default_note = Note(owner_id=user.id, title="Persona", content="Default.", default_type="persona")
    db.add(default_note)
    override_note = Note(owner_id=user.id, title="Override", content="Old override.")
    db.add(override_note)
    await db.commit()
    await db.refresh(conversation)
    await db.refresh(override_note)
    db.add(NotePin(note_id=override_note.id, conversation_id=conversation.id, pin_type="persona"))
    await db.commit()

    await note_slot_service.update_slot_content(db, conversation, user, "persona", "New global.", only_this_chat=False)
    await db.commit()

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.is_override is False
    assert slot.content == "New global."

    await db.refresh(default_note)
    assert default_note.content == "New global."

    remaining_pins = (
        (await db.execute(select(NotePin).where(NotePin.conversation_id == conversation.id))).scalars().all()
    )
    assert remaining_pins == []


@pytest.mark.asyncio
async def test_update_slot_content_only_this_chat_forks_a_new_note(db, user):
    """Checked: creates a private per-chat note, leaving the shared
    default note untouched."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    default_note = Note(owner_id=user.id, title="Persona", content="Default.", default_type="persona")
    db.add(default_note)
    await db.commit()
    await db.refresh(conversation)

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "Just for this chat.", only_this_chat=True
    )
    await db.commit()

    await db.refresh(default_note)
    assert default_note.content == "Default."  # untouched

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.is_override is True
    assert slot.content == "Just for this chat."


@pytest.mark.asyncio
async def test_update_slot_content_only_this_chat_forks_note_titled_after_channel(
    db, admin_user, user, channel_manager_user
):
    """A channel's shared conversation always has title "New chat" (see
    Conversation.title) — its real display name lives on Channel.name
    instead. A forked per-chat note must use that channel name, not the
    generic conversation title."""
    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="Marketing",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "Channel persona.", only_this_chat=True
    )
    await db.commit()

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.title == "Persona — Marketing"


@pytest.mark.asyncio
async def test_update_slot_content_only_this_chat_edits_existing_override_in_place(db, user):
    """Checked, with an override already pinned: edits that note's
    content directly rather than forking yet another one."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    override_note = Note(owner_id=user.id, title="Override", content="Old.")
    db.add(override_note)
    await db.commit()
    await db.refresh(conversation)
    await db.refresh(override_note)
    db.add(NotePin(note_id=override_note.id, conversation_id=conversation.id, pin_type="persona"))
    await db.commit()

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "Updated override.", only_this_chat=True
    )
    await db.commit()

    await db.refresh(override_note)
    assert override_note.content == "Updated override."

    all_notes = (await db.execute(select(Note).where(Note.owner_id == user.id))).scalars().all()
    assert len(all_notes) == 1  # edited in place, not forked again


@pytest.mark.asyncio
async def test_update_slot_content_global_creates_default_note_if_somehow_missing(db, user):
    """Defensive fallback: seed_default_notes guarantees one default
    note per slot for every user in practice, but this must not
    silently no-op if that invariant is ever violated."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    # Deliberately no default "persona" note exists yet for this user.

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "First ever content.", only_this_chat=False
    )
    await db.commit()

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.content == "First ever content."
    assert slot.is_override is False
