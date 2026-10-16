"""Tests for services/chat_service.py: build_reply_stream's history assembly and title-mode gating. Related:
tests/test_chat_history_service.py (trimming/merging), title_service's tests (the two title modes) and
tests/test_reply_generation_service.py (the detached generation used by personal and channel chats).
"""

import asyncio

import pytest
from sqlalchemy import select

from app.models import Conversation, Message, User
from app.schemas import ChannelCreate, ConversationCreate
from app.services import (
    channel_service,
    chat_service,
    chat_settings_service,
    conversation_service,
    model_catalog_service,
    reply_generation_service,
    reply_model_unload_service,
    title_service,
)
from app.services.auth_service import hash_password
from app.services.chat_attachment_service import AttachmentInfo
from app.services.inference_client import InferenceError


@pytest.fixture(autouse=True)
def _fake_model_is_installed(monkeypatch):
    """build_reply_stream calls model_catalog_service.resolve_installed_model before every reply, which calls
    installed_chat_models, a real call to the active engine. Tests here use "fake-model" and don't test that
    fallback (test_model_catalog_service.py and the dedicated tests below do), so it counts as installed and
    resolve_installed_model is a no-op without hitting an engine.
    """

    async def fake_installed_chat_models():
        return ["fake-model"]

    monkeypatch.setattr(model_catalog_service, "installed_chat_models", fake_installed_chat_models)


@pytest.mark.asyncio
async def test_build_reply_stream_yields_the_user_message_id_first(db, user, monkeypatch):
    """Regression: chat.js needs this dedicated event (not folded into "title" or the first "chunk") to tag the *user*
    message's optimistic bubble (chat.js readAssistantReplyStream). Without it that bubble has no data-message-id,
    and a later channel poll/live-watch catch-up (reply_generation_service.stream_reply's cancel_on_disconnect,
    chat.js ownSendInFlight) treats the same user message as unrecognized and duplicates it (a shipped bug).
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    first_event = await chat_service.build_reply_stream(db, conversation, user, "hello").__anext__()

    assert "user_message_id" in first_event
    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    user_message = next(m for m in messages if m.role == "user")
    assert first_event["user_message_id"] == user_message.id


@pytest.mark.asyncio
async def test_build_reply_stream_yields_the_user_message_id_first_for_a_channel_too(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """Same contract for a channel's shared conversation: not personal-chat-only like "title"."""

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

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
    first_event = await chat_service.build_reply_stream(db, conversation, user, "hello everyone").__anext__()

    assert "user_message_id" in first_event
    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    user_message = next(m for m in messages if m.role == "user")
    assert first_event["user_message_id"] == user_message.id


@pytest.mark.asyncio
async def test_build_reply_stream_sends_the_current_message_to_the_model(db, user, monkeypatch):
    """Regression: build_reply_stream handed _trim_history conversation.messages as is, assuming it included the new
    user message after `await db.commit`. It never did: AsyncSession is expire_on_commit=False (app/database.py),
    so committing doesn't invalidate the loaded collection, and db.add-ing a Message doesn't append it in memory
    (only conversation.messages.append(...) does). The model saw history only through the *previous* turn, never
    the current question: replies seemed to answer an earlier message.
    """
    captured: list[list[dict]] = []

    async def fake_chat_stream(_model, messages, _params):
        captured.append([dict(m) for m in messages])
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)

    async for _event in chat_service.build_reply_stream(db, conversation, user, "first question"):
        pass
    assert captured[0][-1] == {"role": "user", "content": "first question"}

    async for _event in chat_service.build_reply_stream(db, conversation, user, "second question"):
        pass
    assert captured[1][-1] == {"role": "user", "content": "second question"}


@pytest.mark.asyncio
async def test_build_reply_stream_always_sends_a_system_message_even_when_empty(db, user, monkeypatch):
    """A new conversation with default notes off, no RAG hits and no attachments has nothing for the system prompt,
    but the system message must still be sent empty, not omitted: some chat-template layers (e.g. Matricxon's
    VicunaPromptBuilder) treat "no system message" as license for their own default preamble, adding the unwanted
    tokens this prevents.
    """
    captured: list[list[dict]] = []

    async def fake_chat_stream(_model, messages, _params):
        captured.append([dict(m) for m in messages])
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)

    async for _event in chat_service.build_reply_stream(db, conversation, user, "hi"):
        pass

    assert captured[0][0] == {"role": "system", "content": ""}


@pytest.mark.asyncio
async def test_build_reply_stream_simple_mode_done_event_carries_the_final_title(db, user, monkeypatch):
    """ "Simple" mode (default; chat_settings_service.DEFAULT_TITLE_MODE) sets the title before streaming
    (title_service.maybe_set_title), so the "done" event's title is final with no follow-up.
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    events = [
        event
        async for event in chat_service.build_reply_stream(db, conversation, user, "What is the capital of France?")
    ]
    done_event = next(e for e in events if e.get("done"))
    assert done_event["title"] == "What is the capital of France?"
    assert conversation.title == "What is the capital of France?"


@pytest.mark.asyncio
async def test_build_reply_stream_yields_the_title_before_the_reply_starts(db, user, monkeypatch):
    """Regression: "simple" mode's title depends only on the user's message, but build_reply_stream sent it only in
    "done", behind the *entire reply*. A dedicated title event must reach the consumer before the first chunk.
    """
    events: list[str] = []

    async def fake_chat_stream(_model, _messages, _params):
        events.append("first chunk yielded")
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for event in chat_service.build_reply_stream(db, conversation, user, "What is the capital of France?"):
        if event.get("title") and not event.get("done"):
            events.append(f"title event: {event['title']}")

    assert events == ["title event: What is the capital of France?", "first chunk yielded"]


@pytest.mark.asyncio
async def test_build_reply_stream_smart_mode_schedules_title_after_done(db, user, monkeypatch):
    """Regression: title generation ran *before* "done", so the Send button (chat.js streamAssistantReply stops
    waiting right after "done") stayed disabled for the whole second model call. In "smart" mode
    (chat_settings_service.set_title_mode) title_service.schedule_smart_title_generation must run only *after*
    "done" (see tests/test_title_service.py for why it's scheduled, not awaited).
    """
    events: list[str] = []

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    def fake_schedule(_conversation_id, _first_user_message):
        events.append("smart title scheduled")

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(title_service, "schedule_smart_title_generation", fake_schedule)

    await chat_settings_service.set_title_mode(db, "smart")
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)

    async for event in chat_service.build_reply_stream(db, conversation, user, "hello"):
        if event.get("done"):
            events.append("done yielded")

    assert events == ["done yielded", "smart title scheduled"]


@pytest.mark.asyncio
async def test_build_reply_stream_never_titles_a_channel_conversation(
    db,
    admin_user,
    user,
    channel_manager_user,
    monkeypatch,
):
    """A channel conversation's title comes from the channel `name` (conversation_service.update_conversation), never
    chat_service, and its "done" has no "title" key (see test_build_reply_stream_persists_a_channel_reply).
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

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
    async for _event in chat_service.build_reply_stream(db, conversation, user, "Hello everyone"):
        pass
    assert conversation.title == "New chat"


@pytest.mark.asyncio
async def test_build_reply_stream_persists_a_channel_reply(
    db,
    admin_user,
    user,
    channel_manager_user,
    monkeypatch,
):
    """A channel reply goes through the same reply_generation_service.stream_reply as a personal one
    (tests/test_reply_generation_service.py). This checks the channel specifics: the assistant Message is
    persisted as status="complete" and its "done" carries no "title" (a personal chat's does, see
    test_build_reply_stream_simple_mode_done_event_carries_the_final_title).
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

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
    events = [event async for event in chat_service.build_reply_stream(db, conversation, user, "Hello everyone")]

    done_event = next(e for e in events if e.get("done"))
    assert "title" not in done_event

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    assistant_message = next(m for m in messages if m.role == "assistant")
    assert assistant_message.status == "complete"
    assert assistant_message.content == "ok"


@pytest.mark.asyncio
async def test_build_reply_stream_links_the_reply_to_its_triggering_user_message(db, user, monkeypatch):
    """conversation_service.delete_message uses Message.reply_to_message_id to find and cancel a streaming reply
    when its question is deleted, so build_reply_stream must pass the saved user_message's id to stream_reply.
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(db, conversation, user, "hello"):
        pass

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    user_message = next(m for m in messages if m.role == "user")
    assistant_message = next(m for m in messages if m.role == "assistant")
    assert assistant_message.reply_to_message_id == user_message.id


@pytest.mark.asyncio
async def test_build_reply_stream_persists_a_failed_personal_reply_instead_of_losing_it(db, user, monkeypatch):
    """Regression: before reply_generation_service unified generation, a personal chat's mid-stream InferenceError
    yielded {"error":...} and saved nothing, leaving no trace on reopening. Now the assistant Message is saved with
    status="error" and the partial content, like a channel's (test_build_reply_stream_persists_a_channel_reply).
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        raise InferenceError("boom")

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    events = [event async for event in chat_service.build_reply_stream(db, conversation, user, "hello")]
    assert events[-1]["error"] == "boom"

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    assistant_message = next(m for m in messages if m.role == "assistant")
    assert assistant_message.status == "error"
    assert assistant_message.content == "partial"


@pytest.mark.asyncio
async def test_build_reply_stream_cancels_a_personal_reply_on_disconnect(db, user, monkeypatch):
    """Regression: build_reply_stream must pass cancel_on_disconnect=True for a personal chat. Only its owner could
    watch, so leaving should stop generation (see tests/test_reply_generation_service.py
    test_cancel_on_disconnect_stops_generation_and_marks_it_errored), unlike a channel's, which keeps running.
    """
    never = asyncio.Event()

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        await never.wait()
        yield "unreachable"  # pragma: no cover

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    gen = chat_service.build_reply_stream(db, conversation, user, "hello")
    user_message_id_event = await gen.__anext__()
    assert "user_message_id" in user_message_id_event
    title_event = await gen.__anext__()
    assert "title" in title_event
    chunk_event = await gen.__anext__()
    assert chunk_event["chunk"] == "partial"

    # Simulate the owner navigating away mid-reply.
    await gen.aclose()

    # Waits for the cancelled task *and* the fresh task it spawns to persist the cancellation (a cancelled task can't
    # await its own cleanup; reply_model_unload_service.schedule_cancelled_write has its own tracking set).
    for _ in range(200):
        tasks = list(reply_generation_service._background_generation_tasks) + list(
            reply_model_unload_service._background_tasks
        )
        if not tasks:
            break
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        await asyncio.sleep(0)

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    assistant_message = next(m for m in messages if m.role == "assistant")
    assert assistant_message.status == "error"
    assert assistant_message.content == "partial"


@pytest.mark.asyncio
async def test_build_reply_stream_tags_the_user_message_with_its_sender(db, user, monkeypatch):
    """A "user" Message records who typed it (Message.sender_id) so a channel can label messages
    (MessageOut.sender_display_name, chat.js bubbleFor); a personal chat never shows it.
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(db, conversation, user, "hello"):
        pass

    # Queried fresh: conversation.messages has the same staleness as the regression test above (db.add never updates
    # it).
    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()

    # The `user` fixture has no first/last name, so the label is the bare username (the name-combining cases:
    # test_message_sender_display_name_combines_username_and_name).
    user_message = next(m for m in messages if m.role == "user")
    assert user_message.sender_id == user.id
    assert user_message.sender_display_name == user.username
    # No avatar (the fixture sets none): falls back to initials, the username's first letter.
    assert user_message.sender_avatar_url is None
    assert user_message.sender_initials == user.username[0].upper()

    assistant_message = next(m for m in messages if m.role == "assistant")
    assert assistant_message.sender_id is None
    assert assistant_message.sender_display_name is None
    assert assistant_message.sender_avatar_url is None
    assert assistant_message.sender_initials is None


@pytest.mark.asyncio
async def test_build_reply_stream_skips_generation_for_a_normal_channel_message(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """ask_ai=False ("post as a normal message", #ask-ai-btn in chat.js) never touches the model: no assistant
    Message is created and chat_stream isn't called.
    """

    def fake_chat_stream(*_args, **_kwargs):
        raise AssertionError("chat_stream must not be called for an ask_ai=False message")

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

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
    events = [
        event async for event in chat_service.build_reply_stream(db, conversation, user, "just chatting", ask_ai=False)
    ]

    assert events[-1] == {"done": True, "sources": []}
    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content == "just chatting"


@pytest.mark.asyncio
async def test_build_reply_stream_merges_consecutive_channel_messages_before_calling_the_model(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """Covers several members posting with ask_ai=False in a row before anyone asks the AI (see
    test_build_reply_stream_skips_generation_for_a_normal_channel_message): consecutive "user" messages, which a
    strict chat template (Mistral/Ministral, used by Ollama and Matricxon) rejects: "conversation roles must
    alternate user and assistant roles" (seen in Ollama's logs). chat_history_service must fold them into one
    turn, each line attributed to its sender, before the model sees them.
    """
    captured: list[list[dict]] = []

    async def fake_chat_stream(_model, messages, _params):
        captured.append([dict(m) for m in messages])
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

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

    async for _event in chat_service.build_reply_stream(db, conversation, user, "hi there", ask_ai=False):
        pass

    # Simulates a real second request, which re-fetches its Conversation from the DB (routers/conversations.py
    # Depends(get_db) plus a fresh query). Reusing the in-memory `conversation` keeps its loaded `.messages`
    # (conversation_service.visible_messages), which would miss the first message (expire_on_commit=False,
    # app/database.py).
    await db.refresh(conversation, attribute_names=["messages"])

    async for _event in chat_service.build_reply_stream(
        db, conversation, channel_manager_user, "how's it going", ask_ai=True
    ):
        pass

    sent_messages = captured[0]
    # A leading system message is now always sent, even empty (see chat_service.build_reply_stream); irrelevant to
    # this test's point (folding consecutive user messages), so only its role is asserted.
    assert [m["role"] for m in sent_messages] == ["system", "user"]
    assert sent_messages[1]["content"] == (
        f"{user.username}: hi there\n{channel_manager_user.username}: how's it going"
    )


@pytest.mark.asyncio
async def test_build_reply_stream_ask_ai_false_still_asks_ai_for_a_personal_chat(db, user, monkeypatch):
    """A personal chat has no other human, so it must always ask the AI even with ask_ai=False. The frontend never
    offers it; the service is the authoritative enforcement.
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    events = [event async for event in chat_service.build_reply_stream(db, conversation, user, "hello", ask_ai=False)]

    done_event = next(e for e in events if e.get("done"))
    assert "title" in done_event

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    assistant_message = next((m for m in messages if m.role == "assistant"), None)
    assert assistant_message is not None
    assert assistant_message.status == "complete"
    assert assistant_message.content == "ok"


@pytest.mark.asyncio
async def test_message_sender_display_name_combines_username_and_name(db, monkeypatch):
    """Message.sender_display_name is the username alone with no name, or "First (username)"/"Last (username)"/
    "First Last (username)": the username always stays visible to identify the account.
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    cases = [
        ("noname", None, None, "noname"),
        ("firstonly", "Alice", None, "Alice (firstonly)"),
        ("lastonly", None, "Smith", "Smith (lastonly)"),
        ("both", "Alice", "Smith", "Alice Smith (both)"),
    ]
    for username, first_name, last_name, expected in cases:
        sender = User(
            username=username,
            password_hash=hash_password("x"),
            role="user",
            first_name=first_name,
            last_name=last_name,
        )
        db.add(sender)
        await db.commit()

        conversation = await conversation_service.create_conversation(
            db, ConversationCreate(model="fake-model"), sender
        )
        async for _event in chat_service.build_reply_stream(db, conversation, sender, "hi"):
            pass
        message = (
            await db.execute(select(Message).where(Message.conversation_id == conversation.id, Message.role == "user"))
        ).scalar_one()
        assert message.sender_display_name == expected


@pytest.mark.asyncio
async def test_message_sender_avatar_url_reflects_the_senders_own_picture(db, monkeypatch):
    """The "avatar is set" branch only; the avatar-less case is in
    test_message_sender_display_name_combines_username_and_name (the `user` fixture has none)."""

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    sender = User(username="withavatar", password_hash=hash_password("x"), role="user", avatar_path="u123.png")
    db.add(sender)
    await db.commit()

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), sender)
    async for _event in chat_service.build_reply_stream(db, conversation, sender, "hi"):
        pass
    message = (
        await db.execute(select(Message).where(Message.conversation_id == conversation.id, Message.role == "user"))
    ).scalar_one()

    assert message.sender_avatar_url == f"/api/account/{sender.id}/avatar?v=u123.png"
    assert message.sender_initials == "W"


@pytest.mark.asyncio
async def test_build_reply_stream_records_the_attachments_on_the_user_message(db, user, tmp_path, monkeypatch):
    """Every attachment lands on the *user* message (never the assistant reply) as a MessageAttachment row
    (Message.attachments).
    """

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    # attachment.path must be under chat_service's ATTACHMENTS_DIR (build_reply_stream's relative_to call), patched to
    # tmp_path so tests don't write into the real attachments/ folder.
    monkeypatch.setattr(chat_service, "ATTACHMENTS_DIR", tmp_path)

    path = tmp_path / "notes.txt"
    path.write_text("some notes")
    path2 = tmp_path / "more.txt"
    path2.write_text("more notes")
    attachments = [
        AttachmentInfo(path=path, filename="notes.txt", type="text"),
        AttachmentInfo(path=path2, filename="more.txt", type="text"),
    ]

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(
        db, conversation, user, "see attached", attachments=attachments
    ):
        pass

    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()
    user_message = next(m for m in messages if m.role == "user")
    assistant_message = next(m for m in messages if m.role == "assistant")
    assert sorted(a.filename for a in user_message.attachments) == ["more.txt", "notes.txt"]
    assert all(a.type == "text" for a in user_message.attachments)
    assert assistant_message.attachments == []


@pytest.mark.asyncio
async def test_build_reply_stream_puts_a_text_attachment_in_front_of_the_newest_user_message(
    db, user, tmp_path, monkeypatch
):
    """A text/PDF attachment's extracted content reaches the model as part of the system prompt, with the conversation's
    model unchanged (no vision override; see the image test below)."""
    captured: dict = {}

    async def fake_chat_stream(model, messages, _params):
        captured["model"] = model
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    # attachment.path must be under chat_service's ATTACHMENTS_DIR (build_reply_stream's relative_to call), patched to
    # tmp_path so tests don't write into the real attachments/ folder.
    monkeypatch.setattr(chat_service, "ATTACHMENTS_DIR", tmp_path)

    path = tmp_path / "notes.txt"
    path.write_text("the launch code is 12345")
    attachment = AttachmentInfo(path=path, filename="notes.txt", type="text")

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(
        db, conversation, user, "what's the code?", attachments=[attachment]
    ):
        pass

    assert captured["model"] == "fake-model"
    system_message = next(m for m in captured["messages"] if m["role"] == "system")
    assert "the launch code is 12345" not in system_message["content"]  # keeps the system prompt cache-stable
    last_user = [m for m in captured["messages"] if m["role"] == "user"][-1]["content"]
    assert "the launch code is 12345" in last_user and "notes.txt" in last_user
    assert last_user.endswith("what's the code?")


@pytest.mark.asyncio
async def test_build_reply_stream_uses_the_default_vision_model_for_an_image_attachment(
    db, user, tmp_path, monkeypatch
):
    """An image attachment is answered by the admin's default vision model for that reply only; conversation.model is
    never written, so a later message without one reverts (checked by re-reading conversation.model).
    """
    captured: dict = {}

    async def fake_chat_stream(model, messages, _params):
        captured["model"] = model
        captured["messages"] = messages
        yield "ok"

    async def fake_get_default_vision_model(_db):
        return "llava:latest"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(model_catalog_service, "get_default_vision_model", fake_get_default_vision_model)
    monkeypatch.setattr(chat_service, "ATTACHMENTS_DIR", tmp_path)

    path = tmp_path / "photo.png"
    path.write_bytes(b"fake-image-bytes")
    attachment = AttachmentInfo(path=path, filename="photo.png", type="image")

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(
        db, conversation, user, "what is this?", attachments=[attachment]
    ):
        pass

    assert captured["model"] == "llava:latest"
    assert captured["messages"][-1]["role"] == "user"
    assert "images" in captured["messages"][-1]

    # conversation.model was never touched: the next (no-attachment) message reverts.
    await db.refresh(conversation)
    assert conversation.model == "fake-model"


@pytest.mark.asyncio
async def test_build_reply_stream_falls_back_to_conversation_model_with_no_vision_model_installed(
    db, user, tmp_path, monkeypatch
):
    """With no vision model configured/installed, an image attachment goes through as an ordinary message on the
    conversation's model instead of failing the request.
    """
    captured: dict = {}

    async def fake_chat_stream(model, messages, _params):
        captured["model"] = model
        captured["messages"] = messages
        yield "ok"

    async def fake_get_default_vision_model(_db):
        return None

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(model_catalog_service, "get_default_vision_model", fake_get_default_vision_model)
    monkeypatch.setattr(chat_service, "ATTACHMENTS_DIR", tmp_path)

    path = tmp_path / "photo.png"
    path.write_bytes(b"fake-image-bytes")
    attachment = AttachmentInfo(path=path, filename="photo.png", type="image")

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(
        db, conversation, user, "what is this?", attachments=[attachment]
    ):
        pass

    assert captured["model"] == "fake-model"
    assert "images" not in captured["messages"][-1]


@pytest.mark.asyncio
async def test_build_reply_stream_falls_back_when_the_conversation_model_is_not_installed(db, user, monkeypatch):
    """Covers: after an admin switches the active engine (engine_service), an *existing* conversation's model still
    names a tag that engine lacks (Ollama and Matricxon have separate catalogs), and sending used to fail at once
    with a raw 404. Now it falls back to an installed model and saves that correction on the conversation, so it
    isn't re-resolved on every message.
    """
    captured: dict = {}

    async def fake_chat_stream(model, messages, _params):
        captured["model"] = model
        captured["messages"] = messages
        yield "ok"

    async def fake_installed_chat_models():
        return ["actually-installed-model"]

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(model_catalog_service, "installed_chat_models", fake_installed_chat_models)

    # Built directly, not via create_conversation (which now rejects "stale-model"): it represents a conversation
    # that existed with a then-installed model before an engine switch or uninstall made it stale (create_conversation's
    # validation is the earlier defense; see conversation_service's tests).
    conversation = Conversation(owner_id=user.id, title="Chat", model="stale-model")
    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    async for _event in chat_service.build_reply_stream(db, conversation, user, "hello"):
        pass

    assert captured["model"] == "actually-installed-model"
    await db.refresh(conversation)
    assert conversation.model == "actually-installed-model"


@pytest.mark.asyncio
async def test_a_channel_can_hide_plain_messages_from_the_model(
    db, admin_user, user, channel_manager_user, monkeypatch
):
    """With the admin setting off, a message posted without asking the AI never reaches the model."""
    from app.services import channel_history_setting

    captured: list[list[dict]] = []

    async def fake_chat_stream(_model, messages, _params):
        captured.append([dict(m) for m in messages])
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
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
    async for _ in chat_service.build_reply_stream(db, conversation, user, "side chatter", ask_ai=False):
        pass

    for include in (True, False):
        await channel_history_setting.set_include_plain_messages(db, include)
        captured.clear()
        await db.refresh(conversation, ["messages"])  # a real request loads these fresh
        async for _ in chat_service.build_reply_stream(db, conversation, user, "now the real question"):
            pass
        sent = " ".join(m["content"] for m in captured[0])
        assert ("side chatter" in sent) is include
        assert "now the real question" in sent
