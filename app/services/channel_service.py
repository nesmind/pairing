"""
Business logic behind app/routers/channels.py: admin CRUD for channels
and their membership, plus the permission check
(can_manage_channel_conversation) that app/services/conversation_service.py
and app/services/note_service.py use to gate a channel conversation's
model/persona/rules/skill against everyone but admins and that specific
channel's managers.
"""

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Channel, ChannelMember, Conversation, User
from app.schemas import ChannelCreate, ChannelMemberOut, ChannelOut, ChannelUpdate
from app.services import chat_attachment_service
from app.services.settings_service import get_default_model, get_default_params


def can_manage_channel_conversation(conversation: Conversation, user: User) -> bool:
    """True if `user` may change `conversation`'s model/params or its
    pinned persona/rules/skill notes — global admins always can; anyone
    else needs to be flagged is_manager on the specific channel this
    conversation belongs to. Assumes `conversation.channel` is already
    loaded (true for anything fetched via
    conversation_service.get_accessible_conversation_or_404, since
    Conversation.channel and Channel.members are both lazy="selectin")."""
    if user.role == "admin":
        return True
    if conversation.channel is None:
        return False
    member = next((m for m in conversation.channel.members if m.user_id == user.id), None)
    return bool(member and member.is_manager)


async def _validate_membership(db: AsyncSession, member_ids: list[str], manager_ids: set[str]) -> list[User]:
    """Shared validation for create/update: at least one manager is
    required, every member id must be a real user, every manager id must
    also be a member, and every manager must currently hold the
    "channel_manager" role (see app/schemas/user.py) — that role is what
    makes a user *eligible* to be flagged as a manager; being flagged on
    a specific channel is what actually grants the rights (see
    can_manage_channel_conversation). Only reached when membership is
    actually being set/changed (create always calls this; update only
    calls it when body.member_user_ids was sent), so a rename-only PATCH
    never re-triggers this requirement against a channel's existing,
    already-valid membership."""
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
    """The channels `user` is a member of, for the sidebar (see
    app/routers/channels.py's GET /api/channels). Membership-only — a
    global admin who isn't a member of a given channel doesn't see it
    here, by design."""
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
    """Creates the channel, its one shared Conversation (started with
    `creator`'s own default model/params — same convention
    conversation_service.create_conversation uses for a brand-new
    personal chat), and its initial membership, all in one transaction."""
    manager_ids = set(body.manager_user_ids)
    members = await _validate_membership(db, body.member_user_ids, manager_ids)
    await _assert_name_available(db, body.name)

    channel = Channel(name=body.name)
    db.add(channel)
    await db.flush()  # assigns channel.id for the conversation/members below

    db.add(
        Conversation(
            channel_id=channel.id,
            model=await get_default_model(db, creator),
            params=await get_default_params(db, creator),
        )
    )
    for member in members:
        db.add(ChannelMember(channel_id=channel.id, user_id=member.id, is_manager=member.id in manager_ids))

    await db.commit()
    await db.refresh(channel)
    return channel


async def update_channel(db: AsyncSession, channel: Channel, body: ChannelUpdate) -> Channel:
    """Renames the channel and/or replaces its membership wholesale (see
    ChannelUpdate's docstring on why member/manager updates are a full
    replace, not a diff)."""
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
    # attribute_names=["members"] (not a bare refresh()): a bare
    # db.refresh() expires *all* of channel's relationship attributes,
    # including ones this function never touched (like `conversation`)
    # and, critically, doesn't safely repopulate them afterward under
    # AsyncSession — the very next access (channel_service.to_channel_out,
    # called by the router right after this returns) would then attempt
    # a synchronous lazy load outside of an awaited context and crash
    # with sqlalchemy.exc.MissingGreenlet. Naming exactly the one
    # relationship this function actually changed avoids expiring
    # anything else, and IS safely reloaded here since it's awaited.
    await db.refresh(channel, attribute_names=["members"])
    return channel


async def delete_channel(db: AsyncSession, channel: Channel) -> None:
    """Permanently deletes the channel, its shared conversation (and that
    conversation's messages/note_pins), and its membership rows — all via
    cascades declared on Channel/Conversation (see app/models/channel.py,
    app/models/conversation.py). There's no undo.

    The ORM cascade has no idea attachment files exist on disk at all —
    removed explicitly first, same as
    conversation_service.delete_conversation's identical treatment for a
    personal conversation. `channel.conversation` is lazy="selectin", so
    this is safe to read with no extra query."""
    chat_attachment_service.delete_conversation_attachments(channel.conversation.id)
    await db.delete(channel)
    await db.commit()


def to_channel_out(channel: Channel) -> ChannelOut:
    # `m.user` reads as None for a membership row whose user no longer exists — user_service.delete_user cleans
    # these up on delete now, but a stray one (confirmed live: this crashed the entire admin Channels list over
    # one bad row, not just that one channel) shouldn't take the whole list down with it. Skipped rather than
    # shown with a fake placeholder username — it's not a real member of anything anymore.
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
