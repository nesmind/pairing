"""Unit tests for app/services/conversation_service.py — conversation
creation defaults, ownership lookups, and the model-switch validation
that closes the 2026-09-08 security review's §3.6 finding (a client
could previously set a conversation's model to any string, installed or
not)."""

import asyncio

import pytest
from fastapi import HTTPException

from app.models import Conversation, Message, MessageAttachment
from app.schemas import ChannelCreate, ConversationCreate, ConversationUpdate
from app.services import (
    channel_service,
    chat_attachment_service,
    conversation_service,
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
async def test_create_conversation_explicit_model_wins_over_default(db, user, monkeypatch):
    async def fake_get_default_model(_db, _user):
        return "should-not-be-used"

    async def fake_get_default_params(_db, _user):
        return {}

    monkeypatch.setattr(conversation_service, "get_default_model", fake_get_default_model)
    monkeypatch.setattr(conversation_service, "get_default_params", fake_get_default_params)

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
    # The rejected update must not have taken effect.
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
    """`since=None` covers a poller's very first tick — the full current
    history, same as GET /{id}/messages."""
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()

    db.add(Message(conversation_id=conversation.id, role="user", content="hi"))
    await db.commit()
    # A directly-constructed Conversation (as opposed to one loaded via a
    # query) has never populated its lazy="selectin" `messages`
    # collection — refresh forces a real (selectin) load of it, same as
    # other tests in this file that touch a freshly-constructed row's
    # relationships.
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
    """A streaming channel reply's Message row is updated in place (see
    app.services.channel_generation_service), not only ever created
    once — get_messages_since must surface that as a change too, via
    Message.updated_at's onupdate, not just brand-new rows."""
    conversation = Conversation(owner_id=user.id, title="Chat", model="fake-model")
    db.add(conversation)
    await db.commit()

    # attachments=[] set explicitly so it's already loaded in memory —
    # db.refresh(conversation) below reloads the *messages* collection
    # but, unlike a real fresh query, doesn't cascade into reloading each
    # message's own nested relationships, so an untouched `attachments`
    # would otherwise need a genuine (here, unsupported outside a real
    # request) lazy load the moment MessageOut validation reads it.
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
    """The backend's authoritative guard behind "only one AI reply at a
    time per channel" (see app/routers/chat.py's stream_reply) — a
    "streaming" assistant Message means generation is already under
    way."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    db.add(Message(conversation_id=conversation.id, role="assistant", content="", status="streaming"))
    await db.commit()
    assert await conversation_service.has_active_reply(db, conversation) is True


@pytest.mark.asyncio
async def test_has_active_reply_false_once_every_reply_has_settled(db, user):
    """ "complete" and "error" are both settled states — neither should
    count as still active."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    db.add(Message(conversation_id=conversation.id, role="assistant", content="done", status="complete"))
    db.add(Message(conversation_id=conversation.id, role="assistant", content="oops", status="error"))
    await db.commit()
    assert await conversation_service.has_active_reply(db, conversation) is False


@pytest.mark.asyncio
async def test_mark_interrupted_messages_as_errored_sweeps_stale_streaming_rows(db, user):
    """Simulates a process restart: a message left "streaming" by a task
    that can't possibly still exist (asyncio.Task is pure in-process
    state — see this function's own docstring) must be swept to "error"
    on the next startup, keeping whatever partial content it already
    had."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    stuck = Message(conversation_id=conversation.id, role="assistant", content="got this far", status="streaming")
    already_done = Message(conversation_id=conversation.id, role="assistant", content="fine", status="complete")
    db.add_all([stuck, already_done])
    await db.commit()
    stuck_id, done_id = stuck.id, already_done.id

    swept = await conversation_service.mark_interrupted_messages_as_errored(db)
    assert swept == 1

    stuck_after = await db.get(Message, stuck_id)
    assert stuck_after.status == "error"
    assert stuck_after.content == "got this far"

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
    """A "complete"/"error" message has nothing running to stop — delete_message
    must not call cancel_generation or publish anything for one."""

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
    """Deleting a still-streaming reply must stop generation outright
    (see reply_generation_service.cancel_generation) and tell any live
    watcher directly (see reply_broadcast_service.publish_deleted) rather
    than leaving them waiting on a "done"/"error" that would otherwise
    never come."""
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
    """Deleting the user message that's still being answered must take
    the in-progress reply down with it too — found via
    Message.reply_to_message_id, not inferred from timestamps — so the
    model doesn't keep visibly "thinking" about a question that no
    longer exists."""
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
    # conversation.messages (lazy="selectin") was already loaded as part
    # of create_channel's return above, before question/reply were added
    # directly via db.add() — the same staleness gotcha
    # chat_service.build_reply_stream's own docstring documents, since
    # this app's AsyncSession is expire_on_commit=False. delete_message
    # reads this collection to find the paired reply, so it needs a
    # fresh load here to see either message at all.
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
    """The reverse of the above: once a reply has already finished
    ("complete" or "error"), deleting the question it answered must NOT
    also delete it — the user explicitly wants a settled reply left for
    manual deletion, not auto-removed."""

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
    await db.refresh(conversation, attribute_names=["messages"])  # see the other test's comment on this

    await conversation_service.delete_message(db, conversation, question)

    assert question.status == "deleted"
    assert reply.status == "complete"
    assert reply.content == "already answered"


@pytest.mark.asyncio
async def test_delete_message_removes_its_attachment_files_and_rows(db, user, tmp_path, monkeypatch):
    """A soft-deleted message's attachments must not linger on disk
    forever — see chat_attachment_service.delete_attachment_files, called
    from conversation_service._cancel_and_clear."""
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
    """Hard-deleting a personal conversation must take every attachment
    it ever held with it — see
    chat_attachment_service.delete_conversation_attachments, called from
    delete_conversation before the ORM cascade removes the Message rows."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    attachment_dir = tmp_path / conversation.id
    attachment_dir.mkdir()
    (attachment_dir / "abc123_photo.png").write_bytes(b"fake")

    await conversation_service.delete_conversation(db, conversation)

    assert not attachment_dir.exists()


@pytest.mark.asyncio
async def test_delete_conversation_with_no_attachments_does_not_raise(db, user, tmp_path, monkeypatch):
    """shutil.rmtree(..., ignore_errors=True) must not blow up (or get
    called with ignore_errors=False by mistake) when a conversation never
    had any attachment folder to begin with — the common case."""
    monkeypatch.setattr(chat_attachment_service, "ATTACHMENTS_DIR", tmp_path)
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)

    await conversation_service.delete_conversation(db, conversation)  # must not raise
