"""Unit tests for app/services/note_service.py — prompt-building from
active persona/rules/skill notes. The chat page's one-slot-at-a-time
resolve/edit logic (resolve_slot, update_slot_content, save_disabled_slots)
now lives in app/services/note_slot_service.py, tested separately in
tests/test_note_slot_service.py."""

import pytest
from sqlalchemy import select

from app.models import Conversation, Note, NotePin
from app.schemas import NotePinCreate
from app.services import note_service


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
async def test_sync_pins_works_on_a_just_flushed_note(db, user):
    """Regression test: app/routers/notes.py's create_note only
    flush()es a new Note before calling sync_pins (never select()s it),
    so note.pins is never populated by a query. sync_pins must not read
    that relationship attribute directly — doing so used to raise
    sqlalchemy.exc.MissingGreenlet, crashing every POST /api/notes call
    that included a pin (see app/services/note_service.py:sync_pins)."""
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    note = Note(owner_id=user.id, title="New note", content="hi")
    db.add(note)
    await db.flush()  # assigns note.id — mirrors create_note's own flow exactly

    await note_service.sync_pins(db, note, user, [NotePinCreate(conversation_id=conversation.id, pin_type="persona")])
    await db.commit()

    pins = (await db.execute(select(NotePin).where(NotePin.note_id == note.id))).scalars().all()
    assert len(pins) == 1
    assert pins[0].conversation_id == conversation.id
