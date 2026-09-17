"""The login account model. See app/models/__init__.py for the package
this belongs to and app/services/auth_service.py for the business logic
built on top of it."""

from sqlalchemy import Column, DateTime, String

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow


class User(Base):
    """A login account. Every conversation and per-user setting belongs
    to exactly one User (see Conversation.owner_id and
    AppSetting.owner_id) — the shared knowledge base is the one
    exception, deliberately visible to every user (see
    app/services/document_service.py)."""

    __tablename__ = "users"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Matches the max_length on schemas.user.UserCreate.username — the
    # app never lets a longer value in, this just gives MySQL a bound too.
    username = Column(String(50), nullable=False, unique=True)
    # Purely descriptive — never used for login/identity (username is),
    # and nullable since every account that existed before this field
    # was added has neither, and nothing in the app requires either to
    # be set going forward either.
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    # Filename only (not a full path) under app.config.AVATARS_DIR — see
    # app/services/avatar_service.py, the only writer. Null means "no
    # picture uploaded"; the UI falls back to `initials` below in that
    # case (see also Message.sender_avatar_url/sender_initials in
    # app/models/conversation.py, which read this same pair per message
    # sender).
    avatar_path = Column(String(255), nullable=True)
    # bcrypt hash (see app/services/auth_service.py) — the plaintext
    # password is never stored or logged anywhere.
    password_hash = Column(String(255), nullable=False)
    # "admin" unlocks the Settings page's System tab (see
    # app/routers/settings.py); "user" is everyone else. Nothing about a
    # user's own conversations/settings differs by role.
    role = Column(String(20), nullable=False, default="user")
    # "active" or "disabled". A disabled account can't log in, and an
    # already-open session for it stops working on its very next request
    # (see app.services.auth_service.get_current_user) — an admin
    # flipping this doesn't have to also force a logout separately.
    # Deliberately a soft deactivation rather than a delete: it keeps the
    # account's conversation history intact and reversible.
    status = Column(String(20), nullable=False, default="active")
    created_at = Column(DateTime, default=utcnow)

    @property
    def avatar_url(self) -> str | None:
        """Same "URL only once the file actually exists" contract as
        app.models.conversation.MessageAttachment.url — null until this
        account has uploaded a picture (see
        app.services.avatar_service.save_avatar), in which case the UI
        falls back to `initials` below instead.

        The `?v=<avatar_path>` suffix is a cache-buster: avatar_path
        itself now includes a random token generated fresh on every
        upload (see save_avatar's own docstring on why — a re-upload
        that kept the exact same URL left browsers showing the stale
        cached picture, confirmed live), so reusing it here as the query
        value is enough on its own — no extra state, no filesystem call,
        and no reliance on a timestamp's resolution being fine enough to
        tell two fast re-uploads apart (an earlier version of this used
        the file's own mtime for that and, confirmed live, two uploads in
        quick succession could land on the same one)."""
        if not self.avatar_path:
            return None
        return f"/api/account/{self.id}/avatar?v={self.avatar_path}"

    @property
    def initials(self) -> str:
        """Round-avatar fallback text shown wherever no picture has been
        uploaded (see app.services.avatar_service and chat.js's
        renderAvatar) — the first letter of first_name and of last_name
        (whichever are set), or just the username's own first letter if
        neither is. Always at least one character: every account has a
        username."""
        letters = [name[0] for name in (self.first_name, self.last_name) if name]
        return "".join(letters).upper() if letters else self.username[0].upper()
