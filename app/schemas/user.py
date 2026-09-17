"""Request/response shapes for account management (Settings > Users,
admin-only) and self-service account actions in app/routers/settings.py."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class UserOut(BaseModel):
    """One row in the admin System tab's user list — never includes the
    password hash, just enough to see and manage an account. `id` is
    included (even though the Users tab itself keys off `username`, its
    stable identity) because the Channels tab's member/manager picker
    (see app/static/js/settings.js: buildChannelForm) reuses this same
    endpoint and needs real user ids to send as
    ChannelCreate/ChannelUpdate's member_user_ids/manager_user_ids."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    username: str
    first_name: str | None = None
    last_name: str | None = None
    role: str
    status: str
    created_at: datetime


class UserCreate(BaseModel):
    """Body for POST /api/settings/users (admin-only). There's no public
    self-registration in this app, so this is the only way to add an
    account short of editing the database directly."""

    username: str = Field(..., min_length=3, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=6, max_length=200)
    # Purely descriptive (see app/models/user.py) — optional, unlike
    # username/password, since nothing in the app depends on either
    # being set.
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    role: Literal["admin", "user", "channel_manager"] = "user"
    status: Literal["active", "disabled"] = "active"


class UserUpdate(BaseModel):
    """Body for PATCH /api/settings/users/{username} (admin-only).
    Every field is optional so a save can touch just one of
    password/role/status without needing to resend the others; the
    username itself isn't editable here since it's the account's
    identity (create a new account instead of renaming one)."""

    password: str | None = Field(None, min_length=6, max_length=200)
    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)
    role: Literal["admin", "user", "channel_manager"] | None = None
    status: Literal["active", "disabled"] | None = None


class AccountProfileOut(BaseModel):
    """GET/PATCH/PUT/DELETE .../account/* responses — the current user's
    own name + picture, for the Settings > Account > Profile section.
    `avatar_url`/`initials` mirror app.models.user.User's identically
    named properties directly (from_attributes reads them like any other
    attribute)."""

    model_config = ConfigDict(from_attributes=True)

    first_name: str | None = None
    last_name: str | None = None
    avatar_url: str | None = None
    initials: str


class AccountProfileUpdate(BaseModel):
    """Body for PATCH /api/account/profile — self-service only, unlike
    UserUpdate above (admin-only, can also touch password/role/status).
    Both fields optional but always sent by the frontend (blank clears
    the name), same convention as UserUpdate's own first_name/last_name."""

    first_name: str | None = Field(None, max_length=100)
    last_name: str | None = Field(None, max_length=100)


class PublicProfileOut(BaseModel):
    """GET /api/account/{user_id} — the small set of profile fields any
    logged-in user can see about any *other* user: whatever's already
    visible via a channel's sender_display_name label, just structured
    (rather than pre-formatted into one string) for chat.js's "click an
    avatar to see more" modal. Deliberately excludes anything not
    already effectively public within this app's own trust boundary —
    no role/status/email/etc. A natural place to grow more fields into
    later, per the modal's own explicit "more details later" design."""

    model_config = ConfigDict(from_attributes=True)

    username: str
    first_name: str | None = None
    last_name: str | None = None
    avatar_url: str | None = None
    initials: str


class ChangePasswordRequest(BaseModel):
    """Body for POST /api/settings/change-password — self-service,
    unlike UserUpdate above (admin-only, can set anyone's password
    without knowing it). Requires the account's *current* password as
    proof of ownership before setting the new one."""

    current_password: str
    new_password: str = Field(..., min_length=6, max_length=200)
