"""Unit tests for app/services/channel_reset_service.py (admin "Reset channel")."""

import pytest
from sqlalchemy import func, select

from app.models import Channel, Message, MessageAttachment
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
    return await _reload(db, channel)


async def _reload(db, channel):
    """The channel as a real request sees it: loaded fresh, with its conversation, messages and members."""
    channel_id = channel.id
    db.expunge_all()
    return await db.get(Channel, channel_id)


async def _count(db, model):
    return (await db.execute(select(func.count()).select_from(model))).scalar_one()


@pytest.mark.asyncio
async def test_reset_clears_history_but_keeps_channel_members_and_conversation(
    db, admin_user, user, channel_manager_user
):
    channel = await _channel_with_history(db, admin_user, user, channel_manager_user)
    await db.refresh(channel.conversation, ["messages"])

    removed = await ChannelResetService.reset(db, channel, admin_user)

    assert removed == 2
    assert await _count(db, Message) == 0 and await _count(db, MessageAttachment) == 0
    assert channel.conversation.title == "New chat"
    assert len(channel.members) == 2 and channel.conversation.channel_id == channel.id


@pytest.mark.asyncio
async def test_router_reset_is_wired_to_the_service(db, admin_user, user, channel_manager_user):
    channel = await _channel_with_history(db, admin_user, user, channel_manager_user)

    response = await channels_router.reset_channel(channel.id, db=db, admin=admin_user)

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


async def _admin_with_notes(db, admin_user, persona: str):
    from app.services import default_notes_setting, note_service

    await default_notes_setting.set_default_notes_enabled(db, admin_user, True)  # new channels start with notes on
    await note_service.seed_default_notes(db, admin_user)
    (await note_service.get_default_note(db, admin_user.id, "persona")).content = persona
    await db.commit()


@pytest.mark.asyncio
async def test_a_new_channel_gets_its_notes_once_and_later_edits_do_not_change_them(
    db, admin_user, user, channel_manager_user
):
    """The channel's system prompt must stay the same (so the model's cache stays valid) whoever asks and
    whatever anyone edits afterwards - its notes are copied in at creation."""
    from app.services import note_service

    await _admin_with_notes(db, admin_user, "first persona")
    body = ChannelCreate(
        name="notes", member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[channel_manager_user.id]
    )
    channel = await _reload(db, await channel_service.create_channel(db, body, admin_user))
    conversation = channel.conversation

    (await note_service.get_default_note(db, admin_user.id, "persona")).content = "edited later"
    await db.commit()

    for asker in (admin_user, user, channel_manager_user):
        notes = await note_service.resolve_conversation_notes(db, conversation, asker.id)
        assert notes["persona"][0].content == "first persona"


@pytest.mark.asyncio
async def test_reset_sets_the_notes_afresh_from_the_resetting_admins_defaults(
    db, admin_user, user, channel_manager_user
):
    from app.services import note_service

    await _admin_with_notes(db, admin_user, "first persona")
    body = ChannelCreate(
        name="notes2", member_user_ids=[user.id, channel_manager_user.id], manager_user_ids=[channel_manager_user.id]
    )
    channel = await _channel_with_history_for(db, admin_user, body)
    (await note_service.get_default_note(db, admin_user.id, "persona")).content = "second persona"
    await db.commit()

    await channels_router.reset_channel(channel.id, db=db, admin=admin_user)

    channel = await _reload(db, channel)
    notes = await note_service.resolve_conversation_notes(db, channel.conversation, user.id)
    assert [n.content for n in notes["persona"]] == ["second persona"]


async def _channel_with_history_for(db, admin_user, body):
    return await _reload(db, await channel_service.create_channel(db, body, admin_user))
