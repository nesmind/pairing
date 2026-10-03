"""Sharing one message from a private chat into a channel (app/services/message_share_service.py and the
POST .../messages/{id}/share endpoint): a plain text copy posted as the sharer's own message, with no AI reply
and no link back to the private chat. Router functions are called directly, like tests/test_chat_router.py."""

import pytest
from fastapi import HTTPException

from app.models import Message, User
from app.routers import conversations as conversations_router
from app.schemas import ChannelCreate, ConversationCreate, MessageOut, ShareMessageRequest
from app.services import channel_service, conversation_service
from app.services.auth_service import hash_password


async def _private_chat(db, user, *messages: tuple[str, str, str]):
    """A private conversation with `(role, content, status)` messages; returns it plus their ids."""
    conversation = await conversation_service.create_conversation(db, ConversationCreate(model="fake-model"), user)
    rows = [
        Message(conversation_id=conversation.id, role=r, content=c, status=s, sender_id=None) for r, c, s in messages
    ]
    db.add_all(rows)
    await db.commit()
    await db.refresh(conversation)
    return conversation, [row.id for row in rows]


async def _channel(db, admin_user, manager, *members):
    body = ChannelCreate(
        name="general", member_user_ids=[manager.id, *(m.id for m in members)], manager_user_ids=[manager.id]
    )
    return await channel_service.create_channel(db, body, admin_user)


async def _share(db, user, conversation, message_id, channel_id) -> MessageOut:
    shared = await conversations_router.share_message(
        conversation.id, message_id, ShareMessageRequest(channel_id=channel_id), db=db, user=user
    )
    return MessageOut.model_validate(shared)


@pytest.mark.asyncio
async def test_sharing_your_own_message_posts_it_to_the_channel_as_you(db, user, admin_user, channel_manager_user):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, ("user", "what is RAG?", "complete"))

    out = await _share(db, user, conversation, message_id, channel.id)

    assert out.role == "user" and out.content == "what is RAG?"
    assert out.shared_from == "user" and out.status == "complete"
    assert out.sender_id == user.id and "alice" in out.sender_display_name
    await db.refresh(channel.conversation)
    posted = [m for m in channel.conversation.messages if m.id == out.id]
    assert len(posted) == 1 and posted[0].conversation_id == channel.conversation.id


@pytest.mark.asyncio
async def test_sharing_an_ai_answer_is_a_user_message_tagged_as_assistant_origin(
    db, user, admin_user, channel_manager_user
):
    """The copy must be a plain "user"-role message (an "assistant" one in a channel would read as the AI
    replying), remembering only where it came from."""
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (_, answer_id) = await _private_chat(
        db, user, ("user", "q", "complete"), ("assistant", "the answer", "complete")
    )

    out = await _share(db, user, conversation, answer_id, channel.id)

    assert (out.role, out.content, out.shared_from) == ("user", "the answer", "assistant")


@pytest.mark.asyncio
async def test_sharing_never_asks_the_ai_and_leaves_the_private_chat_untouched(
    db, user, admin_user, channel_manager_user
):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, ("user", "hello", "complete"))

    await _share(db, user, conversation, message_id, channel.id)

    await db.refresh(channel.conversation)
    assert [m.role for m in channel.conversation.messages] == ["user"]  # no assistant reply row
    await db.refresh(conversation)
    assert [m.id for m in conversation.messages] == [message_id]
    assert conversation.messages[0].shared_from is None


@pytest.mark.asyncio
async def test_the_channel_conversation_is_bumped_so_members_pick_the_message_up(
    db, user, admin_user, channel_manager_user
):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, ("user", "hi", "complete"))
    before = channel.conversation.updated_at

    await _share(db, user, conversation, message_id, channel.id)

    await db.refresh(channel.conversation, attribute_names=["updated_at"])  # re-read as stored (naive UTC)
    assert channel.conversation.updated_at > before


@pytest.mark.asyncio
async def test_a_non_member_cannot_share_into_a_channel(db, user, admin_user, channel_manager_user):
    channel = await _channel(db, admin_user, channel_manager_user)  # alice is not in it
    conversation, (message_id,) = await _private_chat(db, user, ("user", "hi", "complete"))

    with pytest.raises(HTTPException) as exc_info:
        await _share(db, user, conversation, message_id, channel.id)

    assert exc_info.value.status_code == 403
    await db.refresh(channel.conversation)
    assert channel.conversation.messages == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "status"), [("assistant", "streaming"), ("assistant", "error"), ("system", "complete")]
)
async def test_unfinished_failed_and_system_messages_cannot_be_shared(
    db, user, admin_user, channel_manager_user, role, status
):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, (role, "text", status))

    with pytest.raises(HTTPException) as exc_info:
        await _share(db, user, conversation, message_id, channel.id)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_an_empty_message_cannot_be_shared(db, user, admin_user, channel_manager_user):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, ("assistant", "   ", "complete"))

    with pytest.raises(HTTPException) as exc_info:
        await _share(db, user, conversation, message_id, channel.id)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_unknown_message_and_unknown_channel_are_404(db, user, admin_user, channel_manager_user):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    conversation, (message_id,) = await _private_chat(db, user, ("user", "hi", "complete"))

    for target_message, target_channel in (("nope", channel.id), (message_id, "nope")):
        with pytest.raises(HTTPException) as exc_info:
            await _share(db, user, conversation, target_message, target_channel)
        assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_a_channel_message_cannot_be_reshared_from_inside_a_channel(db, user, admin_user, channel_manager_user):
    channel = await _channel(db, admin_user, channel_manager_user, user)
    posted = Message(conversation_id=channel.conversation.id, role="user", content="hi", sender_id=user.id)
    db.add(posted)
    await db.commit()

    with pytest.raises(HTTPException) as exc_info:
        await _share(db, user, channel.conversation, posted.id, channel.id)

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_someone_elses_private_chat_cannot_be_shared_from(db, user, admin_user, channel_manager_user):
    """The endpoint opens the conversation through the normal access check, so another user's private chat is
    a 404 - its messages can never be pushed into a channel by anyone but its owner."""
    intruder = User(username="mallory", password_hash=hash_password("pw"), role="user")
    db.add(intruder)
    await db.commit()
    channel = await _channel(db, admin_user, channel_manager_user, intruder)
    conversation, (message_id,) = await _private_chat(db, user, ("user", "private", "complete"))

    with pytest.raises(HTTPException) as exc_info:
        await conversations_router.share_message(
            conversation.id, message_id, ShareMessageRequest(channel_id=channel.id), db=db, user=intruder
        )

    assert exc_info.value.status_code == 404
