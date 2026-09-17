"""Unit tests for app/routers/chat.py's get_attachment endpoint — called
directly, not through a TestClient/ASGI app (see
tests/test_instance_proxy_http.py's own docstring on why this project
doesn't use that pattern). Covers a real regression found via live QA:
MessageAttachment.message has no lazy="selectin" (unlike most
relationships in this codebase), so reading attachment.message.
conversation_id directly crashed every attachment fetch with
sqlalchemy.exc.MissingGreenlet — fixed by looking the Message up by
attachment.message_id instead."""

from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse

from app.models import Message, MessageAttachment
from app.routers import chat as chat_router
from app.schemas import ConversationCreate
from app.services import chat_attachment_service, conversation_service


@pytest.mark.asyncio
async def test_get_attachment_serves_the_file_for_its_owner(db, user, tmp_path, monkeypatch):
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)
    monkeypatch.setattr(chat_router, "ATTACHMENTS_DIR", tmp_path)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    (tmp_path / conversation.id).mkdir()
    (tmp_path / conversation.id / "abc_notes.txt").write_text("hello")

    message = Message(conversation_id=conversation.id, role="user", content="see attached", sender_id=user.id)
    message.attachments = [
        MessageAttachment(path=f"{conversation.id}/abc_notes.txt", filename="notes.txt", type="text")
    ]
    db.add(message)
    await db.commit()
    attachment_id = message.attachments[0].id

    response = await chat_router.get_attachment(attachment_id, db=db, user=user)

    assert isinstance(response, FileResponse)
    assert Path(response.path).read_text() == "hello"


@pytest.mark.asyncio
async def test_get_attachment_404s_for_an_unknown_id(db, user):
    with pytest.raises(HTTPException) as exc_info:
        await chat_router.get_attachment("does-not-exist", db=db, user=user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_attachment_rejects_a_user_with_no_access_to_the_conversation(
    db, user, admin_user, tmp_path, monkeypatch
):
    """A real membership/ownership check, not just "does the row exist"
    — the whole point of routing every attachment fetch through
    get_accessible_conversation_or_404 rather than serving straight off
    the id."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)
    monkeypatch.setattr(chat_router, "ATTACHMENTS_DIR", tmp_path)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="user", content="see attached", sender_id=user.id)
    message.attachments = [MessageAttachment(path=f"{conversation.id}/x.txt", filename="x.txt", type="text")]
    db.add(message)
    await db.commit()
    attachment_id = message.attachments[0].id

    with pytest.raises(HTTPException) as exc_info:
        await chat_router.get_attachment(attachment_id, db=db, user=admin_user)
    assert exc_info.value.status_code == 404
