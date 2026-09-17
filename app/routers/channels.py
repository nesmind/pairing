"""
Channel administration (create/edit/delete, admin-only — see
app/services/channel_service.py) plus the one member-facing endpoint
that lists the channels the current user belongs to, for the sidebar.
This file only handles HTTP routing, status codes, and Depends()
injection; see channel_service for the actual business logic.

Two routers live here since the two concerns have different audiences
and prefixes: `router` (admin-only, under /api/settings/channels,
mirrors app/routers/users.py) and `member_router` (any logged-in user,
under /api/channels).
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ChannelCreate, ChannelOut, ChannelSummary, ChannelUpdate, OkResponse
from app.services import channel_service
from app.services.auth_service import get_current_user, require_admin

router = APIRouter(prefix="/api/settings/channels", tags=["channels"])
member_router = APIRouter(prefix="/api/channels", tags=["channels"])


@router.get("", response_model=list[ChannelOut])
async def list_channels(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    channels = await channel_service.list_channels(db)
    return [channel_service.to_channel_out(c) for c in channels]


@router.post("", response_model=ChannelOut, status_code=201)
async def create_channel(
    body: ChannelCreate,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    channel = await channel_service.create_channel(db, body, admin)
    return channel_service.to_channel_out(channel)


@router.patch("/{channel_id}", response_model=ChannelOut)
async def update_channel(
    channel_id: str,
    body: ChannelUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    channel = await channel_service.get_channel_or_404(db, channel_id)
    channel = await channel_service.update_channel(db, channel, body)
    return channel_service.to_channel_out(channel)


@router.delete("/{channel_id}", response_model=OkResponse)
async def delete_channel(
    channel_id: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    channel = await channel_service.get_channel_or_404(db, channel_id)
    await channel_service.delete_channel(db, channel)
    return OkResponse()


@member_router.get("", response_model=list[ChannelSummary])
async def list_my_channels(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The channels `user` is currently a member of — powers the
    sidebar's Channels section. Membership-only, no admin override (see
    channel_service.list_channels_for_user's docstring)."""
    channels = await channel_service.list_channels_for_user(db, user)
    return [
        ChannelSummary(
            id=c.id,
            name=c.name,
            conversation_id=c.conversation.id,
            is_manager=next(m.is_manager for m in c.members if m.user_id == user.id),
        )
        for c in channels
    ]
