"""Request/response shapes for app/routers/channels.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChannelMemberOut(BaseModel):
    user_id: str
    username: str
    is_manager: bool


class ChannelOut(BaseModel):
    """One channel, for the admin Channels tab — full membership included
    so the edit form can pre-populate its member/manager pickers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    conversation_id: str
    created_at: datetime
    members: list[ChannelMemberOut]


class ChannelSummary(BaseModel):
    """One channel the current user belongs to, for the sidebar (see
    app/routers/channels.py's GET /api/channels) — just enough to list
    and open it, plus whether *this* viewer can manage it."""

    id: str
    name: str
    conversation_id: str
    is_manager: bool


class ChannelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    member_user_ids: list[str] = Field(..., min_length=1)
    # Must be a subset of member_user_ids, and every id in it must belong
    # to a user whose role is "channel_manager" — validated in
    # app.services.channel_service.create_channel.
    manager_user_ids: list[str] = []


class ChannelUpdate(BaseModel):
    """Every field is optional so a save can rename a channel without
    touching membership. member_user_ids/manager_user_ids, when given,
    are a full replace (not a diff) of the current membership — the same
    "resend everything that should be true" convention as
    app.services.note_service.sync_pins — so manager_user_ids should
    always be resent alongside member_user_ids on a membership change,
    even if the manager set itself didn't change."""

    name: str | None = Field(None, min_length=1, max_length=255)
    member_user_ids: list[str] | None = None
    manager_user_ids: list[str] | None = None
