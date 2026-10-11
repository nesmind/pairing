"""Tests for services/note_slot_service.py, the chat page's one-slot view: resolving a slot (default vs.
override vs. turned-off) and editing it via the "Only for this chat" checkbox (update_slot_content).
"""

import pytest
from sqlalchemy import select

from app.models import Conversation, Note, NotePin
from app.schemas import ChannelCreate
from app.services import channel_service, note_slot_service


@pytest.mark.asyncio
async def test_save_disabled_slots_replaces_params_dict(db, user):
    """SQLAlchemy's JSON column tracks reassignment, not in-place mutation. This would still pass with a buggy
    in-place `.update` on sqlite's in-memory session, but documents the contract for production (MySQL/Postgres JSON
    columns behave the same).
    """
    conversation = Conversation(owner_id=user.id, params={"temperature": 0.8})
    note_slot_service.save_disabled_slots(conversation, {"persona"})
    assert conversation.params["disabled_default_notes"] == ["persona"]
    assert conversation.params["temperature"] == 0.8


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
    """The default (checkbox unchecked): updates the shared default note directly, not a new per-chat note."""
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
    assert len(all_notes) == 1

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.is_override is False
    assert slot.content == "New global content."


@pytest.mark.asyncio
async def test_update_slot_content_global_removes_an_existing_override(db, user):
    """Editing globally while this chat has its own override must drop it, or the chat keeps showing stale pinned
    content."""
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
    """Checked: creates a private per-chat note, leaving the shared default untouched."""
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
    assert default_note.content == "Default."

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.is_override is True
    assert slot.content == "Just for this chat."


@pytest.mark.asyncio
async def test_update_slot_content_only_this_chat_forks_note_titled_after_channel(
    db, admin_user, user, channel_manager_user
):
    """A channel's conversation always has title "New chat" (Conversation.title); its display name is Channel.name.
    A forked per-chat note must use the channel name, not the generic title.
    """
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
    """Checked, with an override already pinned: edits that note's content rather than forking another."""
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
    assert len(all_notes) == 1


@pytest.mark.asyncio
async def test_update_slot_content_global_creates_default_note_if_somehow_missing(db, user):
    """Defensive: seed_default_notes guarantees one default note per slot, but this must not silently no-op if that is
    ever violated."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)
    # No default "persona" note exists yet for this user.

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "First ever content.", only_this_chat=False
    )
    await db.commit()

    slot = await note_slot_service.resolve_slot(db, conversation, user, "persona")
    assert slot.content == "First ever content."
    assert slot.is_override is False


async def _channel_with_different_default_notes(db, admin_user, user, channel_manager_user):
    from app.services import note_service

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="Shared",
            member_user_ids=[user.id, channel_manager_user.id, admin_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    channel.conversation.params = {**channel.conversation.params, "disabled_default_notes": []}
    for member, text in ((user, "alice's persona"), (channel_manager_user, "carol's persona"), (admin_user, "admin's")):
        await note_service.seed_default_notes(db, member)
        (await note_service.get_default_note(db, member.id, "persona")).content = text
    await db.commit()
    conversation_id = channel.conversation.id
    db.expunge_all()
    return await db.get(Conversation, conversation_id)


@pytest.mark.asyncio
async def test_a_channel_uses_the_same_default_notes_whoever_is_asking(db, admin_user, user, channel_manager_user):
    """Different system prompts per sender would make the model's prompt cache start over each time the speaker
    changes."""
    from app.services import note_service

    conversation = await _channel_with_different_default_notes(db, admin_user, user, channel_manager_user)

    prompts = []
    for asker in (user, channel_manager_user, admin_user):
        notes = await note_service.resolve_conversation_notes(db, conversation, asker.id)
        prompts.append(note_service.build_pinned_system_prompt(notes))
    slots = [(await note_slot_service.resolve_slot(db, conversation, u, "persona")).content for u in (user, admin_user)]

    assert len(set(prompts)) == 1 and "admin's" in prompts[0]
    assert slots == ["admin's"] * 2


@pytest.mark.asyncio
async def test_a_personal_chat_still_uses_its_own_users_default_notes(db, user, admin_user):
    from app.services import note_service

    await note_service.seed_default_notes(db, user)
    (await note_service.get_default_note(db, user.id, "persona")).content = "alice's persona"
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    notes = await note_service.resolve_conversation_notes(db, conversation, user.id)

    assert notes["persona"][0].content == "alice's persona"


@pytest.mark.asyncio
async def test_editing_a_channel_note_always_pins_it_to_the_channel(db, admin_user, user, channel_manager_user):
    from app.services import note_service

    conversation = await _channel_with_different_default_notes(db, admin_user, user, channel_manager_user)

    await note_slot_service.update_slot_content(
        db, conversation, user, "persona", "New channel text.", only_this_chat=False
    )
    await db.commit()

    assert (await note_service.get_default_note(db, user.id, "persona")).content == "alice's persona"
    assert (await note_service.get_default_note(db, admin_user.id, "persona")).content == "admin's"
    notes = await note_service.resolve_conversation_notes(db, conversation, admin_user.id)
    assert notes["persona"][0].content == "New channel text."


@pytest.mark.asyncio
async def test_a_channel_without_an_admin_member_uses_its_first_manager_notes(
    db, admin_user, user, channel_manager_user
):
    from app.services import note_service

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="NoAdmin",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation_id = channel.conversation.id
    db.expunge_all()
    conversation = await db.get(Conversation, conversation_id)

    assert note_service.default_notes_owner_id(conversation, user.id) == channel_manager_user.id
