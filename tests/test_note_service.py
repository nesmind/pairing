"""Tests for services/note_service.py: prompt building from active persona/rules/skill notes. The chat page's
one-slot resolve/edit logic (resolve_slot, update_slot_content, save_disabled_slots) is in note_slot_service.py,
tested in tests/test_note_slot_service.py.
"""

import pytest
from sqlalchemy import select

from app.models import Conversation, Note, NotePin
from app.schemas import NotePinCreate
from app.services import default_notes_setting, note_service


def test_build_pinned_system_prompt_orders_persona_first():
    persona_note = Note(content="Be helpful.")
    rules_note = Note(content="No lying.")
    prompt = note_service.build_pinned_system_prompt(
        {
            "persona": [persona_note],
            "rules": [rules_note],
            "skill": [],
        }
    )
    assert prompt.index("Persona") < prompt.index("Rules")


@pytest.mark.asyncio
async def test_seed_initial_disabled_notes_disables_all_slots_when_setting_is_off(db, user):
    """The default for a new account (default_notes_setting.get_default_notes_enabled): no saved preference means
    off, so a new conversation starts with all 3 slots disabled, not the old "every slot on".
    """
    params = await note_service.seed_initial_disabled_notes(db, user, {"temperature": 0.5})

    assert params["temperature"] == 0.5
    assert set(params["disabled_default_notes"]) == set(note_service.NOTE_SLOTS)


@pytest.mark.asyncio
async def test_seed_initial_disabled_notes_leaves_params_untouched_when_setting_is_on(db, user):
    await default_notes_setting.set_default_notes_enabled(db, user, True)

    params = await note_service.seed_initial_disabled_notes(db, user, {"temperature": 0.5})

    assert "disabled_default_notes" not in params


@pytest.mark.asyncio
async def test_sync_pins_works_on_a_just_flushed_note(db, user):
    """Regression: routers/notes.py's create_note only flushes a new Note before sync_pins (never selects it), so
    note.pins isn't populated by a query. sync_pins must not read that attribute: it raised
    sqlalchemy.exc.MissingGreenlet and crashed every POST /api/notes with a pin (services/note_service.py:sync_pins).
    """
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    note = Note(owner_id=user.id, title="New note", content="hi")
    db.add(note)
    await db.flush()  # assigns note.id, as create_note does

    await note_service.sync_pins(db, note, user, [NotePinCreate(conversation_id=conversation.id, pin_type="persona")])
    await db.commit()

    pins = (await db.execute(select(NotePin).where(NotePin.note_id == note.id))).scalars().all()
    assert len(pins) == 1
    assert pins[0].conversation_id == conversation.id
