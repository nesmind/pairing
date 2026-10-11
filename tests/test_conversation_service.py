"""Tests for services/conversation_service.py: creation defaults, ownership lookups and the model-switch
validation closing the security review's §3.6 finding (a client could set a conversation's model to
any string, installed or not).
"""

import asyncio

import pytest
from fastapi import HTTPException

from app.models import Conversation, Message, MessageAttachment
from app.schemas import ChannelCreate, ConversationCreate, ConversationUpdate
from app.services import (
    channel_service,
    chat_attachment_service,
    conversation_service,
    default_notes_setting,
    reply_broadcast_service,
    reply_generation_service,
)


@pytest.mark.asyncio
async def test_create_conversation_uses_user_defaults(db, user, monkeypatch):
    async def fake_get_default_model(_db, _user):
        return "llama3:latest"

    async def fake_get_default_params(_db, _user):
        return {"temperature": 0.5, "rag_top_k": 4}

    monkeypatch.setattr(conversation_service, "get_default_model", fake_get_default_model)
    monkeypatch.setattr(conversation_service, "get_default_params", fake_get_default_params)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(), user)

    assert conversation.owner_id == user.id
    assert conversation.model == "llama3:latest"
    assert conversation.params["temperature"] == 0.5


@pytest.mark.asyncio
async def test_create_conversation_starts_with_notes_disabled_by_default(db, user, monkeypatch):
    """A new account has no saved default_notes_setting row, so a new conversation starts with all 3
    persona/rules/skill slots disabled (note_service.seed_initial_disabled_notes), not the old "every slot on".
    """

    async def fake_get_default_model(_db, _user):
        return "llama3:latest"

    monkeypatch.setattr(conversation_service, "get_default_model", fake_get_default_model)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(), user)

    assert set(conversation.params["disabled_default_notes"]) == {"persona", "rules", "skill"}


@pytest.mark.asyncio
async def test_create_conversation_leaves_notes_enabled_when_the_user_turned_it_on(db, user, monkeypatch):
    async def fake_get_default_model(_db, _user):
        return "llama3:latest"

    monkeypatch.setattr(conversation_service, "get_default_model", fake_get_default_model)
    await default_notes_setting.set_default_notes_enabled(db, user, True)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(), user)

    assert conversation.params.get("disabled_default_notes", []) == []


@pytest.mark.asyncio
async def test_create_conversation_explicit_model_wins_over_default(db, user, monkeypatch):
    async def fake_get_default_model(_db, _user):
        return "should-not-be-used"

    async def fake_get_default_params(_db, _user):
        return {}

    async def fake_installed():
        return ["gemma4:e2b"]

    monkeypatch.setattr(conversation_service, "get_default_model", fake_get_default_model)
    monkeypatch.setattr(conversation_service, "get_default_params", fake_get_default_params)
    monkeypatch.setattr(conversation_service, "installed_chat_models", fake_installed)

    conversation = await conversation_service.create_conversation(
        db,
        ConversationCreate(model="gemma4:e2b"),
        user,
    )
    assert conversation.model == "gemma4:e2b"


@pytest.mark.asyncio
async def test_get_accessible_conversation_or_404_rejects_other_users_conversation(db, user, admin_user):
    other_users_chat = Conversation(owner_id=admin_user.id, title="Not yours")
    db.add(other_users_chat)
    await db.commit()
    await db.refresh(other_users_chat)

    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.get_accessible_conversation_or_404(db, other_users_chat.id, user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_get_accessible_conversation_or_404_unknown_id_also_404s(db, user):
    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.get_accessible_conversation_or_404(db, "does-not-exist", user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_conversation_rejects_uninstalled_model(db, user, monkeypatch):
    conversation = Conversation(owner_id=user.id, title="Chat", model="llama3:latest")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    async def fake_installed():
        return ["llama3:latest", "gemma4:e2b"]

    monkeypatch.setattr(conversation_service, "installed_chat_models", fake_installed)

    with pytest.raises(HTTPException) as exc_info:
        await conversation_service.update_conversation(
            db,
            conversation,
            ConversationUpdate(model="not-a-real-model:latest"),
            user,
        )
    assert exc_info.value.status_code == 400
    # The rejected update must not take effect.
    assert conversation.model == "llama3:latest"


@pytest.mark.asyncio
async def test_update_conversation_accepts_installed_model(db, user, monkeypatch):
    conversation = Conversation(owner_id=user.id, title="Chat", model="llama3:latest")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    async def fake_installed():
        return ["llama3:latest", "gemma4:e2b"]

    monkeypatch.setattr(conversation_service, "installed_chat_models", fake_installed)

    updated = await conversation_service.update_conversation(
        db,
        conversation,
        ConversationUpdate(model="gemma4:e2b"),
        user,
    )
    assert updated.model == "gemma4:e2b"


@pytest.mark.asyncio
async def test_get_messages_since_none_returns_full_history(db, user):
    """`since=None` covers a poller's first tick: the full current history, like GET /{id}/messages."""
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()

    db.add(Message(conversation_id=conversation.id, role="user", content="hi"))
    await db.commit()
    # A directly built Conversation never populated its lazy="selectin" `messages`; refresh forces that load, as
    # elsewhere in this file for a fresh row's relationships.
    await db.refresh(conversation)

    result = await conversation_service.get_messages_since(db, conversation, None)
    assert [m.content for m in result.messages] == ["hi"]


@pytest.mark.asyncio
async def test_get_messages_since_only_returns_messages_newer_than_the_checkpoint(db, user):
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()

    db.add(Message(conversation_id=conversation.id, role="user", content="first"))
    await db.commit()
    await db.refresh(conversation)

    checkpoint = (await conversation_service.get_messages_since(db, conversation, None)).server_time
    await asyncio.sleep(0.01)

    db.add(Message(conversation_id=conversation.id, role="assistant", content="second"))
    await db.commit()

    result = await conversation_service.get_messages_since(db, conversation, checkpoint)
    assert [m.content for m in result.messages] == ["second"]


@pytest.mark.asyncio
async def test_get_messages_since_picks_up_a_message_whose_content_was_updated_in_place(db, user):
    """A streaming channel reply's Message row is updated in place (channel_generation_service), not just created;
    get_messages_since must surface that via Message.updated_at's onupdate, not only new rows.
    """
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()

    # attachments=[] set explicitly so it's loaded: db.refresh(conversation) reloads the *messages* collection but,
    # unlike a fresh query, not each message's nested relationships, and an untouched `attachments` would lazy-load
    # (unsupported outside a real request) when MessageOut validates.
    message = Message(conversation_id=conversation.id, role="assistant", content="", status="streaming", attachments=[])
    db.add(message)
    await db.commit()
    await db.refresh(conversation)

    checkpoint = (await conversation_service.get_messages_since(db, conversation, None)).server_time
    await asyncio.sleep(0.01)

    message.content = "now with content"
    await db.commit()

    result = await conversation_service.get_messages_since(db, conversation, checkpoint)
    assert [m.content for m in result.messages] == ["now with content"]


@pytest.mark.asyncio
async def test_get_messages_since_server_time_is_usable_as_the_next_poll_checkpoint(db, user):
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    first = await conversation_service.get_messages_since(db, conversation, None)
    second = await conversation_service.get_messages_since(db, conversation, first.server_time)
    assert second.messages == []


@pytest.mark.asyncio
async def test_has_active_reply_false_for_a_conversation_with_no_messages(db, user):
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    assert await conversation_service.has_active_reply(db, conversation) is False


@pytest.mark.asyncio
async def test_has_active_reply_true_while_an_assistant_message_is_streaming(db, user):
    """The authoritative guard behind "only one AI reply at a time per channel" (routers/chat.py stream_reply): a
    "streaming" assistant Message means generation is under way.
    """
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    db.add(Message(conversation_id=conversation.id, role="assistant", content="", status="streaming"))
    await db.commit()
    assert await conversation_service.has_active_reply(db, conversation) is True


@pytest.mark.asyncio
async def test_has_active_reply_false_once_every_reply_has_settled(db, user):
    """ "complete" and "error" are settled states; neither counts as active."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    db.add(Message(conversation_id=conversation.id, role="assistant", content="done", status="complete"))
    db.add(Message(conversation_id=conversation.id, role="assistant", content="oops", status="error"))
    await db.commit()
    assert await conversation_service.has_active_reply(db, conversation) is False


@pytest.mark.asyncio
async def test_mark_interrupted_messages_as_errored_sweeps_stale_streaming_rows(db, user):
    """Simulates a restart: a message left "streaming" by a task that can't still exist (asyncio.Task is in-process
    state; see the function) is swept to "error" on next startup, keeping its partial content.
    """
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    stuck = Message(conversation_id=conversation.id, role="assistant", content="got this far", status="streaming")
    already_done = Message(conversation_id=conversation.id, role="assistant", content="fine", status="complete")
    db.add_all([stuck, already_done])
    await db.commit()
    stuck_id, done_id = stuck.id, already_done.id

    swept = await conversation_service.mark_interrupted_messages_as_errored(db)
    assert swept == 1

    # This bulk UPDATE (default synchronize_session="evaluate") syncs some columns onto a loaded object (status) but
    # not others (error_message): a SQLAlchemy quirk, not a bug here. The row is written correctly (a fresh session
    # reads it); db.refresh forces the sync instead of depending on identity-map behavior.
    stuck_after = await db.get(Message, stuck_id)
    await db.refresh(stuck_after)
    assert stuck_after.status == "error"
    assert stuck_after.content == "got this far"
    assert stuck_after.error_message == "Interrupted by a server restart."

    done_after = await db.get(Message, done_id)
    assert done_after.status == "complete"


@pytest.mark.asyncio
async def test_visible_messages_excludes_deleted_ones(db, user):
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    db.add(Message(conversation_id=conversation.id, role="user", content="kept"))
    db.add(Message(conversation_id=conversation.id, role="user", content="removed", status="deleted"))
    await db.commit()
    await db.refresh(conversation)

    visible = conversation_service.visible_messages(conversation)
    assert [m.content for m in visible] == ["kept"]


@pytest.mark.asyncio
async def test_delete_message_soft_deletes_a_settled_message_without_touching_generation(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """A "complete"/"error" message has nothing running: delete_message must not call cancel_generation or publish."""

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cancel_generation/publish_deleted must not run for a non-streaming message")

    monkeypatch.setattr(reply_generation_service, "cancel_generation", fail_if_called)
    monkeypatch.setattr(reply_broadcast_service, "publish_deleted", fail_if_called)

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation
    message = Message(conversation_id=conversation.id, role="user", content="hello", sender_id=user.id)
    db.add(message)
    await db.commit()

    await conversation_service.delete_message(db, conversation, message)

    assert message.status == "deleted"
    assert message.content == ""
    assert message.sources is None


@pytest.mark.asyncio
async def test_delete_message_cancels_generation_for_a_streaming_message(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """Deleting a streaming reply stops generation (reply_generation_service.cancel_generation) and tells live
    watchers directly (reply_broadcast_service.publish_deleted) rather than leaving them waiting for a "done"/"error".
    """
    cancelled_ids = []
    published_conversation_ids = []
    monkeypatch.setattr(reply_generation_service, "cancel_generation", lambda mid: cancelled_ids.append(mid) or True)
    monkeypatch.setattr(reply_broadcast_service, "publish_deleted", lambda cid: published_conversation_ids.append(cid))

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation
    message = Message(conversation_id=conversation.id, role="assistant", content="partial", status="streaming")
    db.add(message)
    await db.commit()

    await conversation_service.delete_message(db, conversation, message)

    assert cancelled_ids == [message.id]
    assert published_conversation_ids == [conversation.id]
    assert message.status == "deleted"
    assert message.content == ""


@pytest.mark.asyncio
async def test_delete_message_also_cancels_a_still_streaming_reply_to_it(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """Deleting the user message still being answered takes the in-progress reply down too, found via
    Message.reply_to_message_id, not timestamps, so the model doesn't keep "thinking" about a vanished question.
    """
    cancelled_ids = []
    monkeypatch.setattr(reply_generation_service, "cancel_generation", lambda mid: cancelled_ids.append(mid) or True)
    monkeypatch.setattr(reply_broadcast_service, "publish_deleted", lambda _cid: None)

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation
    question = Message(conversation_id=conversation.id, role="user", content="hello?", sender_id=user.id)
    db.add(question)
    await db.commit()
    reply = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="thinking...",
        status="streaming",
        reply_to_message_id=question.id,
    )
    db.add(reply)
    await db.commit()
    # conversation.messages (selectin) was loaded in create_channel's return, before question/reply were added via
    # db.add: the staleness chat_service.build_reply_stream documents (expire_on_commit=False). delete_message
    # reads it to find the paired reply, so it needs a fresh load.
    await db.refresh(conversation, attribute_names=["messages"])

    await conversation_service.delete_message(db, conversation, question)

    assert cancelled_ids == [reply.id]
    assert question.status == "deleted"
    assert reply.status == "deleted"
    assert reply.content == ""


@pytest.mark.asyncio
async def test_delete_message_leaves_an_already_settled_reply_alone(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """The reverse: once a reply has settled ("complete" or "error"), deleting its question must NOT delete it; it's
    left for manual deletion.
    """

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cancel_generation must not run for an already-settled reply")

    monkeypatch.setattr(reply_generation_service, "cancel_generation", fail_if_called)
    monkeypatch.setattr(reply_broadcast_service, "publish_deleted", fail_if_called)

    channel = await channel_service.create_channel(
        db,
        ChannelCreate(
            name="general",
            member_user_ids=[user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
        admin_user,
    )
    conversation = channel.conversation
    question = Message(conversation_id=conversation.id, role="user", content="hello?", sender_id=user.id)
    db.add(question)
    await db.commit()
    reply = Message(
        conversation_id=conversation.id,
        role="assistant",
        content="already answered",
        status="complete",
        reply_to_message_id=question.id,
    )
    db.add(reply)
    await db.commit()
    await db.refresh(conversation, attribute_names=["messages"])

    await conversation_service.delete_message(db, conversation, question)

    assert question.status == "deleted"
    assert reply.status == "complete"
    assert reply.content == "already answered"


@pytest.mark.asyncio
async def test_delete_message_removes_its_attachment_files_and_rows(db, user, tmp_path, monkeypatch):
    """A soft-deleted message's attachments must not linger on disk (chat_attachment_service.delete_attachment_files,
    called from conversation_service._cancel_and_clear).
    """
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    attachment_dir = tmp_path / conversation.id
    attachment_dir.mkdir()
    attachment_path = attachment_dir / "abc123_notes.txt"
    attachment_path.write_text("some notes")

    message = Message(conversation_id=conversation.id, role="user", content="see attached", sender_id=user.id)
    message.attachments = [
        MessageAttachment(path=f"{conversation.id}/abc123_notes.txt", filename="notes.txt", type="text")
    ]
    db.add(message)
    await db.commit()
    await db.refresh(conversation, attribute_names=["messages"])
    attachment_id = message.attachments[0].id

    await conversation_service.delete_message(db, conversation, message)

    assert not attachment_path.exists()
    assert await db.get(MessageAttachment, attachment_id) is None


@pytest.mark.asyncio
async def test_delete_conversation_removes_its_attachments_folder(db, user, tmp_path, monkeypatch):
    """Hard-deleting a personal conversation takes every attachment with it
    (chat_attachment_service.delete_conversation_attachments, called before the ORM cascade removes Message rows).
    """
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    attachment_dir = tmp_path / conversation.id
    attachment_dir.mkdir()
    (attachment_dir / "abc123_photo.png").write_bytes(b"fake")

    await conversation_service.delete_conversation(db, conversation)

    assert not attachment_dir.exists()


@pytest.mark.asyncio
async def test_delete_conversation_with_no_attachments_does_not_raise(db, user, tmp_path, monkeypatch):
    """shutil.rmtree(..., ignore_errors=True) must not fail (or get ignore_errors=False) when a conversation never had
    an attachment folder: the common case."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)

    await conversation_service.delete_conversation(db, conversation)


@pytest.mark.asyncio
async def test_delete_conversation_cancels_a_still_streaming_reply(db, user, monkeypatch):
    """Deleting the whole conversation mid-reply stops generation like deleting that message
    (test_delete_message_cancels_generation_for_a_streaming_message). Otherwise the ORM cascade removes the Message
    row under the running task, which notices only seconds later via reply_cross_instance_service's watchdog.
    """
    cancelled_ids = []
    monkeypatch.setattr(reply_generation_service, "cancel_generation", lambda mid: cancelled_ids.append(mid) or True)
    monkeypatch.setattr(reply_generation_service, "mark_message_deleted", lambda mid: cancelled_ids.append(mid))

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="assistant", content="partial", status="streaming")
    db.add(message)
    await db.commit()
    await db.refresh(conversation, attribute_names=["messages"])

    await conversation_service.delete_conversation(db, conversation)

    assert cancelled_ids == [message.id, message.id]


@pytest.mark.asyncio
async def test_delete_conversation_does_not_cancel_a_settled_message(db, user, monkeypatch):
    """A "complete"/"error" message has nothing running: deleting its conversation must not call cancel_generation."""

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("cancel_generation must not run for a non-streaming message")

    monkeypatch.setattr(reply_generation_service, "cancel_generation", fail_if_called)
    monkeypatch.setattr(reply_generation_service, "mark_message_deleted", fail_if_called)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    message = Message(conversation_id=conversation.id, role="assistant", content="done", status="complete")
    db.add(message)
    await db.commit()
    await db.refresh(conversation, attribute_names=["messages"])

    await conversation_service.delete_conversation(db, conversation)
