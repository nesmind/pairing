"""Persona/rules/skill of a chat can only change before its first message (app/routers/notes.py)."""

import pytest
from fastapi import HTTPException

from app.models import Conversation, Message
from app.routers.notes import _require_slot_edit_rights


@pytest.mark.asyncio
async def test_slots_are_editable_only_while_the_chat_has_no_messages(db, user):
    conversation = Conversation(owner_id=user.id, params={})
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation, ["messages"])

    await _require_slot_edit_rights(conversation, user)  # fresh chat: fine

    db.add(Message(conversation_id=conversation.id, role="user", content="hi", sender_id=user.id))
    await db.commit()
    await db.refresh(conversation, ["messages"])

    with pytest.raises(HTTPException) as locked:
        await _require_slot_edit_rights(conversation, user)
    assert locked.value.status_code == 409
