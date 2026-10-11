"""Shapes for account management (Settings > Users, admin-only) and self-service actions in routers/settings.py."""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, field_validator


class UserOut(BaseModel):
    """One row of the admin user list, without the password hash. `id` is included because the Channels
    member picker (settings.js buildChannelForm) needs real ids for member_user_ids/manager_user_ids.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    status: str
    created_at: datetime


class UserCreate(BaseModel):
    """Body for POST /api/settings/users (admin-only); there is no self-registration."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=6, max_length=200)
    # Descriptive only (models/user.py); optional.
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    role: Literal["admin", "user", "channel_manager"] = "user"
    status: Literal["active", "disabled"] = "active"


class UserUpdate(BaseModel):
    """Body for PATCH /api/settings/users/{username}. All fields optional; username is the identity and not editable."""

    password: str | None = Field(None, min_length=6, max_length=200)
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    role: Literal["admin", "user", "channel_manager"] | None = None
    status: Literal["active", "disabled"] | None = None


class AccountProfileOut(BaseModel):
    """Responses of.../account/*: the user's own name and picture. `avatar_url`/`initials` mirror User's properties."""

    model_config = ConfigDict(from_attributes=True)

    first_name: str | None = None
    last_name: str | None = None
    avatar_url: str | None = None
    initials: str
    timezone: str


class AccountProfileUpdate(BaseModel):
    """Body for PATCH /api/account/profile (self-service; UserUpdate is admin-only). Both fields always sent; blank
    clears."""

    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    # Optional: omitted leaves the timezone unchanged.
    timezone: str | None = Field(None, max_length=64)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str | None) -> str | None:
        """Must be a real IANA name (the UI passes it to Intl.DateTimeFormat)."""
        if value is None:
            return None
        try:
            ZoneInfo(value)
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError("Unknown timezone.") from exc
        return value


class PublicProfileOut(BaseModel):
    """GET /api/account/{user_id}: profile fields any user may see about another, already visible via a channel's
    sender_display_name, structured for chat.js's avatar modal. Excludes role/status/email; room to grow.
    """

    model_config = ConfigDict(from_attributes=True)

    username: str
    first_name: str | None = None
    last_name: str | None = None
    avatar_url: str | None = None
    initials: str


class ChangePasswordRequest(BaseModel):
    """Body for POST /api/settings/change-password (self-service). Requires the current password."""

    current_password: str
    new_password: str = Field(..., min_length=6, max_length=200)
