"""Unit tests for app/services/channel_reset_service.py (admin "Reset channel")."""

import pytest
from sqlalchemy import func, select

from app.models import Message, MessageAttachment
from app.routers import channels as channels_router
from app.schemas import ChannelCreate, ChannelUpdate
from app.services import channel_service
from app.services.channel_reset_service import ChannelResetService


async def _channel_with_history(db, admin_user, user, channel_manager_user):
    body = ChannelCreate(
        name="general", member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[channel_manager_user.id]
    )
    channel = await channel_service.create_channel(db, body, admin_user)
    conv_id = channel.conversation.id
    first = Message(conversation_id=conv_id, role="user", content="hi", sender_id=user.id)
    db.add(first)
    await db.flush()
    db.add(Message(conversation_id=conv_id, role="assistant", content="hello", reply_to_message_id=first.id))
    db.add(MessageAttachment(message_id=first.id, path="x/y.png", filename="y.png", type="image"))
    channel.conversation.title = "Old topic"
    await db.commit()
    return channel


async def _count(db, model):
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


@pytest.mark.asyncio
async def test_reset_clears_history_but_keeps_channel_members_and_conversation(
    db, admin_user, user, channel_manager_user
):
    channel = await _channel_with_history(db, admin_user, user, channel_manager_user)
    await db.refresh(channel.conversation, ["messages"])

    removed = await ChannelResetService.reset(db, channel)

    assert removed == 2
    assert await _count(db, Message) == 0 and await _count(db, MessageAttachment) == 0
    assert channel.conversation.title == "New chat"
    assert len(channel.members) == 2 and channel.conversation.channel_id == channel.id


@pytest.mark.asyncio
async def test_router_reset_is_wired_to_the_service(db, admin_user, user, channel_manager_user):
    channel = await _channel_with_history(db, admin_user, user, channel_manager_user)

    response = await channels_router.reset_channel(channel.id, db=db, _admin=admin_user)

    assert response.ok is True and await _count(db, Message) == 0


@pytest.mark.asyncio
async def test_my_channels_flags_who_can_manage_them(db, admin_user, user, channel_manager_user):
    await _channel_with_history(db, admin_user, user, channel_manager_user)
    await channel_service.update_channel(
        db,
        (await channel_service.list_channels(db))[0],
        ChannelUpdate(
            member_user_ids=[admin_user.id, user.id, channel_manager_user.id],
            manager_user_ids=[channel_manager_user.id],
        ),
    )

    def flags(rows):
        return [(r.is_manager, r.can_manage) for r in rows]

    assert flags(await channels_router.list_my_channels(db=db, user=admin_user)) == [(False, True)]
    assert flags(await channels_router.list_my_channels(db=db, user=channel_manager_user)) == [(True, True)]
    assert flags(await channels_router.list_my_channels(db=db, user=user)) == [(False, False)]
