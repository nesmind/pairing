"""Request/response shapes for app/routers/channels.py."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ChannelMemberOut(BaseModel):
    user_id: str
    username: str
    is_manager: bool


class ChannelOut(BaseModel):
    """One channel for the admin Channels tab, with full membership for the edit form's pickers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    conversation_id: str
    created_at: datetime
    members: list[ChannelMemberOut]


class ChannelSummary(BaseModel):
    """A channel the user belongs to, for the sidebar (GET /api/channels): enough to list and open it, plus
    whether this viewer can manage it.
    """

    id: str
    name: str
    conversation_id: str
    is_manager: bool
    # Admins and the channel's managers may change its settings (Behavior tab).
    can_manage: bool = False


class ChannelCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    member_user_ids: list[str] = Field(..., min_length=1)
    # Subset of member_user_ids, all with role "channel_manager" (checked in channel_service.create_channel).
    manager_user_ids: list[str] = []


class ChannelUpdate(BaseModel):
    """All fields optional. member_user_ids/manager_user_ids replace the membership wholesale (like
    note_service.sync_pins), so resend manager_user_ids with member_user_ids on any membership change.
    """

    name: str | None = Field(None, min_length=1, max_length=255)
    member_user_ids: list[str] | None = None
    manager_user_ids: list[str] | None = None
