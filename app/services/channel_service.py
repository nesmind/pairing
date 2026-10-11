"""Logic for app/routers/channels.py: admin CRUD for channels and membership, plus
can_manage_channel_conversation, which conversation_service.py and note_service.py use to restrict a
channel's model/persona/rules/skill to admins and its managers.
"""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Channel, ChannelMember, Conversation, User
from app.schemas import ChannelCreate, ChannelMemberOut, ChannelOut, ChannelUpdate
from app.services import chat_attachment_service
from app.services.model_catalog_service import resolve_installed_model
from app.services.settings_service import get_default_model, get_default_params


def can_manage_channel_conversation(conversation: Conversation, user: User) -> bool:
    """True if `user` may change the conversation's model/params or pinned notes: admins, or managers
    (is_manager) of its channel. Assumes `conversation.channel` is loaded (lazy="selectin").
    """
    if user.role == "admin":
        return True
    if conversation.channel is None:
        return False
    member = next((m for m in conversation.channel.members if m.user_id == user.id), None)
    return bool(member and member.is_manager)


async def _validate_membership(db: AsyncSession, member_ids: list[str], manager_ids: set[str]) -> list[User]:
    """Shared create/update validation: at least one manager, every member a real user, every manager a member
    holding the "channel_manager" role (eligibility; flagging on the channel grants the rights). Run on create,
    and on update only when member_user_ids was sent, so rename-only edits skip it.
    """
    if not manager_ids:
        raise HTTPException(status_code=400, detail="A channel needs at least one manager.")
    if manager_ids - set(member_ids):
        raise HTTPException(status_code=400, detail="Managers must also be members.")

    members = (await db.execute(select(User).where(User.id.in_(member_ids)))).scalars().all()
    found_ids = {u.id for u in members}
    missing = set(member_ids) - found_ids
    if missing:
        raise HTTPException(status_code=400, detail="One or more selected users don't exist.")

    ineligible = [u.username for u in members if u.id in manager_ids and u.role != "channel_manager"]
    if ineligible:
        raise HTTPException(
            status_code=400,
            detail=f"Only users with the channel_manager role can be a channel manager: {', '.join(ineligible)}",
        )
    return members


async def _assert_name_available(db: AsyncSession, name: str, exclude_channel_id: str | None = None) -> None:
    stmt = select(Channel.id).where(Channel.name == name)
    if exclude_channel_id:
        stmt = stmt.where(Channel.id != exclude_channel_id)
    if (await db.execute(stmt)).scalar_one_or_none() is not None:
        raise HTTPException(status_code=400, detail=f'A channel named "{name}" already exists.')


async def list_channels(db: AsyncSession) -> list[Channel]:
    """Every channel, for the admin Channels tab."""
    return (await db.execute(select(Channel).order_by(Channel.created_at))).scalars().all()


async def list_channels_for_user(db: AsyncSession, user: User) -> list[Channel]:
    """The channels `user` belongs to (sidebar). Membership only: non-member admins don't see it."""
    return (
        (
            await db.execute(
                select(Channel).join(ChannelMember).where(ChannelMember.user_id == user.id).order_by(Channel.name),
            )
        )
        .scalars()
        .all()
    )


async def get_channel_or_404(db: AsyncSession, channel_id: str) -> Channel:
    channel = await db.get(Channel, channel_id)
    if channel is None:
        raise HTTPException(status_code=404, detail="Channel not found")
    return channel


async def create_channel(db: AsyncSession, body: ChannelCreate, creator: User) -> Channel:
    """Creates the channel, its shared Conversation (with `creator`'s default model/params, like
    conversation_service.create_conversation) and initial membership in one transaction.
    """
    manager_ids = set(body.manager_user_ids)
    members = await _validate_membership(db, body.member_user_ids, manager_ids)
    await _assert_name_available(db, body.name)

    channel = Channel(name=body.name)
    db.add(channel)
    await db.flush()  # assigns channel.id

    # Deferred import: note_service imports this module (can_manage_channel_conversation), so a top-level import would
    # be circular.
    from app.services import note_service

    # get_default_model is a stored preference with no installed-check; apply the same fallback as
    # conversation_service.create_conversation's resolve_installed_model (defaults go stale after an engine
    # switch or deleted model).
    params = await note_service.seed_initial_disabled_notes(db, creator, await get_default_params(db, creator))
    db.add(
        Conversation(
            channel_id=channel.id,
            model=await resolve_installed_model(db, creator, await get_default_model(db, creator)),
            params=params,
        )
    )
    for member in members:
        db.add(ChannelMember(channel_id=channel.id, user_id=member.id, is_manager=member.id in manager_ids))

    await db.commit()
    await db.refresh(channel)
    # The channel's notes are configured once here, copied from the creator's defaults (freeze_channel_notes).
    await note_service.freeze_channel_notes(db, channel.conversation, body.name, creator.id)
    await db.commit()
    return channel


async def update_channel(db: AsyncSession, channel: Channel, body: ChannelUpdate) -> Channel:
    """Renames the channel and/or replaces membership wholesale (see ChannelUpdate)."""
    if body.name is not None and body.name != channel.name:
        await _assert_name_available(db, body.name, exclude_channel_id=channel.id)
        channel.name = body.name

    if body.member_user_ids is not None:
        manager_ids = set(body.manager_user_ids or [])
        members = await _validate_membership(db, body.member_user_ids, manager_ids)
        found_ids = {u.id for u in members}

        existing_by_user = {m.user_id: m for m in channel.members}
        for member in members:
            existing = existing_by_user.get(member.id)
            if existing is None:
                db.add(ChannelMember(channel_id=channel.id, user_id=member.id, is_manager=member.id in manager_ids))
            else:
                existing.is_manager = member.id in manager_ids
        for user_id, row in existing_by_user.items():
            if user_id not in found_ids:
                await db.delete(row)

    await db.commit()
    # attribute_names=["members"] rather than a bare refresh: that expires all relationships (e.g. `conversation`)
    # and under AsyncSession the next access (channel_service.to_channel_out) would lazy-load outside an await
    # (MissingGreenlet). Naming the one changed relationship avoids it.
    await db.refresh(channel, attribute_names=["members"])
    return channel


async def delete_channel(db: AsyncSession, channel: Channel) -> None:
    """Permanently deletes the channel, its conversation (messages, note_pins) and membership via cascades
    (models/channel.py, models/conversation.py). No undo.

    Attachment files are removed explicitly first (the cascade ignores disk), as in
    conversation_service.delete_conversation. `channel.conversation` is selectin-loaded.
    """
    chat_attachment_service.delete_conversation_attachments(channel.conversation.id)
    await db.delete(channel)
    await db.commit()


def to_channel_out(channel: Channel) -> ChannelOut:
    # `m.user` is None for a membership whose user is gone (user_service.delete_user now cleans these; one such
    # row once crashed the admin Channels list). Skip it rather than show a fake member.
    return ChannelOut(
        id=channel.id,
        name=channel.name,
        conversation_id=channel.conversation.id,
        created_at=channel.created_at,
        members=[
            ChannelMemberOut(user_id=m.user_id, username=m.user.username, is_manager=m.is_manager)
            for m in channel.members
            if m.user is not None
        ],
    )
