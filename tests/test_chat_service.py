"""Unit tests for app/services/chat_service.py: the pure history-trimming
helper plus build_reply_stream's own history-assembly and title-mode
gating (see app/services/title_service.py's own tests for the two title
modes themselves, and tests/test_reply_generation_service.py for the
detached-generation mechanism both a personal chat and a channel's
shared conversation go through)."""

import asyncio

import pytest
from sqlalchemy import select

from app.models import Message, User
from app.schemas import ChannelCreate, ConversationCreate
from app.services import (
    channel_service,
    chat_service,
    chat_settings_service,
    conversation_service,
    model_catalog_service,
    reply_generation_service,
    reply_termination_service,
    title_service,
)
from app.services.auth_service import hash_password
from app.services.chat_attachment_service import AttachmentInfo
from app.services.chat_service import _trim_history
from app.services.ollama_client import OllamaError


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def test_trim_history_keeps_everything_when_it_fits():
    messages = [_msg("user", "hi"), _msg("assistant", "hello")]
    trimmed = _trim_history(messages, num_ctx=4096)
    assert [m["content"] for m in trimmed] == ["hi", "hello"]


def test_trim_history_drops_oldest_first_when_over_budget():
    # num_ctx=256 (the function's own floor) minus 512 reserved clamps to
    # a 256-token floor -> 1024 characters of budget (256 * 4 chars/token).
    old = _msg("user", "x" * 900)
    newer = _msg("assistant", "y" * 900)
    newest = _msg("user", "z" * 100)
    trimmed = _trim_history([old, newer, newest], num_ctx=256)

    contents = [m["content"] for m in trimmed]
    assert old.content not in contents  # dropped: budget exceeded once it's included
    assert newest.content in contents  # newest is always kept, even alone
    assert contents.index(newer.content) < contents.index(newest.content)  # order preserved


def test_trim_history_always_keeps_at_least_the_newest_message():
    """Even a single message longer than the whole budget must not be
    dropped — an empty history would mean the model sees no prompt at
    all, which is worse than one slightly-over-budget message."""
    huge = _msg("user", "x" * 100_000)
    trimmed = _trim_history([huge], num_ctx=256)
    assert len(trimmed) == 1


@pytest.mark.asyncio
async def test_build_reply_stream_yields_the_user_message_id_first(db, user, monkeypatch):
    """Regression test: chat.js needs this — its own dedicated event,
    not folded into "title" or the first "chunk" — to tag the *user*
    message's own optimistically-rendered bubble (see
    app/static/js/chat.js's readAssistantReplyStream). Without it, that
    bubble never carries a data-message-id, and a later channel poll/
    live-watch catch-up (see reply_generation_service.stream_reply's
    cancel_on_disconnect and chat.js's ownSendInFlight) sees the same
    user message as unrecognized and creates a genuine duplicate for
    it — a real, shipped bug this event exists to fix."""

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
    """Same contract for a channel's shared conversation — this event
    isn't personal-chat-only the way "title" is."""

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
    """Regression test: build_reply_stream used to hand _trim_history
    conversation.messages as-is, trusting it to already include the
    just-added user message after `await db.commit()`. It never did —
    the app's AsyncSession is expire_on_commit=False (see
    app/database.py), so committing never invalidates an already-loaded
    relationship collection, and db.add()-ing a Message doesn't append
    it to conversation.messages in memory either (that only happens via
    the relationship itself, e.g. conversation.messages.append(...)).
    The practical effect: the model only ever saw history through the
    *previous* turn — never the question actually being asked — which
    is exactly why replies looked like they were answering an earlier
    message instead of the latest one."""
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
async def test_build_reply_stream_simple_mode_done_event_carries_the_final_title(db, user, monkeypatch):
    """ "Simple" mode (the default — see
    app.services.chat_settings_service.DEFAULT_TITLE_MODE) sets the title
    synchronously, before the reply even starts streaming (see
    title_service.maybe_set_title) — the "done" event's title is always
    already final, with no follow-up call needed."""

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
    """Regression test: "simple" mode's title is known instantly (it
    only depends on the user's own message), but build_reply_stream used
    to only ever send it as part of the "done" event — bundled behind
    the *entire reply*, however long that took to stream, even though
    nothing about the title was still waiting on it. A dedicated title
    event must reach the consumer before the first reply chunk does."""
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
    """Regression test: title generation used to unconditionally run
    *before* "done" was yielded, so the frontend's Send button (see
    chat.js: streamAssistantReply, which stops waiting on the response
    right after "done") stayed disabled for however long that second,
    slower model call took, on top of the reply itself already being
    fully shown. With "smart" mode configured (see
    app.services.chat_settings_service.set_title_mode),
    title_service.schedule_smart_title_generation must still only be
    called *after* "done" is yielded — see
    tests/test_title_service.py for why it's a scheduled background task
    rather than a plain `await` here in the first place."""
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
    """A channel conversation's title comes from the channel's own
    `name` (see conversation_service.update_conversation), never from
    chat_service — and its "done" event never carries a "title" key at
    all (see test_build_reply_stream_persists_a_channel_reply below for
    that assertion), unlike a personal chat's."""

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
    """A channel conversation's reply goes through the exact same
    reply_generation_service.stream_reply a personal chat's does (see
    tests/test_reply_generation_service.py for that mechanism itself) —
    this checks what's specific to being a channel: the assistant
    Message actually gets persisted as status="complete", and its "done"
    event carries no "title" (a personal chat's does — see
    test_build_reply_stream_simple_mode_done_event_carries_the_final_title
    above)."""

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
    """conversation_service.delete_message relies on Message.reply_to_message_id
    to find (and cancel) a still-streaming reply when the question that
    triggered it is deleted — build_reply_stream must pass the just-saved
    user_message's id through to reply_generation_service.stream_reply
    for this to work at all."""

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
    """Regression test: before reply_generation_service unified personal
    and channel generation, a personal chat's mid-stream OllamaError just
    yielded {"error": ...} with nothing ever saved — the failed attempt
    left no trace at all, and reopening the conversation later showed no
    sign a reply had even been attempted. Now the assistant Message is
    persisted with status="error" and whatever partial content came
    through, exactly like a channel's reply already does (see
    test_build_reply_stream_persists_a_channel_reply above)."""

    async def fake_chat_stream(_model, _messages, _params):
        yield "partial"
        raise OllamaError("boom")

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
    """Regression test: build_reply_stream must pass
    cancel_on_disconnect=True to reply_generation_service.stream_reply
    for a personal chat — unlike a channel's shared conversation, only
    its owner could ever be watching, so leaving should actually stop
    generation (see tests/test_reply_generation_service.py's
    test_cancel_on_disconnect_stops_generation_and_marks_it_errored for
    the mechanism itself), not let it keep running unseen the way a
    channel's does."""
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

    # Waits for the cancelled task *and* the fresh task it spawns to
    # actually persist that cancellation (a cancelled task can't safely
    # await its own cleanup — see
    # reply_termination_service.schedule_cancelled_write, in a *separate*
    # module's own tracking set from the main generation task's).
    for _ in range(200):
        tasks = list(reply_generation_service._background_generation_tasks) + list(
            reply_termination_service._background_tasks
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
    """A "user"-role Message records who actually typed it
    (Message.sender_id) — needed so a channel's shared conversation can
    label each message with its sender (see MessageOut.sender_display_name
    and chat.js's bubbleFor); a personal chat just never displays it."""

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)

    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    async for _event in chat_service.build_reply_stream(db, conversation, user, "hello"):
        pass

    # Queried fresh rather than read off conversation.messages: that
    # in-memory collection has the exact same staleness the regression
    # test above documents — db.add()-ing a Message never updates it —
    # so it still wouldn't include either message added just above.
    messages = (await db.execute(select(Message).where(Message.conversation_id == conversation.id))).scalars().all()

    # The `user` fixture has no first/last name set, so the label falls
    # back to the bare username — see
    # test_message_sender_display_name_combines_username_and_name below
    # for the name-combining cases.
    user_message = next(m for m in messages if m.role == "user")
    assert user_message.sender_id == user.id
    assert user_message.sender_display_name == user.username
    # No avatar uploaded (the `user` fixture never sets one) — falls
    # back to initials, the bare username's first letter here since it
    # has no first/last name either.
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
    """ask_ai=False ("post as a normal message" — see #ask-ai-btn in
    app/static/js/chat.js) must never touch the model at all: no
    assistant Message is created, and chat_stream is never even
    called."""

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
async def test_build_reply_stream_ask_ai_false_still_asks_ai_for_a_personal_chat(db, user, monkeypatch):
    """A personal chat has no "other human" to just message — it must
    always ask the AI, even if a caller explicitly passes ask_ai=False
    (the frontend never offers this choice there, but the service is the
    authoritative enforcement point, not the UI)."""

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
    """Message.sender_display_name is username-only with no name set,
    or "First (username)"/"Last (username)"/"First Last (username)" once
    either is set — the username stays visible in parentheses either
    way, so the label always still identifies *which account* sent it."""

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
    """See test_message_sender_display_name_combines_username_and_name
    above for the equivalent avatar-less case — this only covers the
    "an avatar is set" branch, since that one already exercises "none
    set" via the `user` fixture."""

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
    """Every attachment must land on the *user* message (never the
    assistant reply) as its own MessageAttachment row — see
    app.models.conversation.Message.attachments' own docstring."""

    async def fake_chat_stream(_model, _messages, _params):
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    # attachment.path must live under chat_service's own ATTACHMENTS_DIR
    # (see build_reply_stream's relative_to(ATTACHMENTS_DIR) call) —
    # patched to tmp_path so these tests don't write into the project's
    # real attachments/ folder.
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
async def test_build_reply_stream_folds_a_text_attachment_into_the_system_prompt(db, user, tmp_path, monkeypatch):
    """A text/PDF/etc. attachment's extracted content must reach the
    model as part of the system prompt, and the conversation's own model
    is used unchanged (no vision-model override — see the image test
    below for that case)."""
    captured: dict = {}

    async def fake_chat_stream(model, messages, _params):
        captured["model"] = model
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr(reply_generation_service, "chat_stream", fake_chat_stream)
    # attachment.path must live under chat_service's own ATTACHMENTS_DIR
    # (see build_reply_stream's relative_to(ATTACHMENTS_DIR) call) —
    # patched to tmp_path so these tests don't write into the project's
    # real attachments/ folder.
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
    assert "the launch code is 12345" in system_message["content"]
    assert "notes.txt" in system_message["content"]


@pytest.mark.asyncio
async def test_build_reply_stream_uses_the_default_vision_model_for_an_image_attachment(
    db, user, tmp_path, monkeypatch
):
    """An image attachment must answer via the admin-configured default
    vision model for this one reply only — conversation.model itself is
    never written, so a later message with no attachment reverts to it
    automatically (checked here by re-reading conversation.model after)."""
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

    # conversation.model was never touched — the next (no-attachment)
    # message reverts to it automatically.
    await db.refresh(conversation)
    assert conversation.model == "fake-model"


@pytest.mark.asyncio
async def test_build_reply_stream_falls_back_to_conversation_model_with_no_vision_model_installed(
    db, user, tmp_path, monkeypatch
):
    """If no vision model is configured/installed, an image attachment
    is silently sent through as an ordinary message on the conversation's
    own model, rather than failing the whole request."""
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
