"""The login account model (logic: app/services/auth_service.py)."""

from sqlalchemy import Column, DateTime, String

from app.database import Base
from app.models._base import ID_LEN, new_id, utcnow

DEFAULT_TIMEZONE = "UTC"


class User(Base):
    """A login account. Conversations and per-user settings belong to one User; the knowledge base is shared."""

    __tablename__ = "users"

    id = Column(String(ID_LEN), primary_key=True, default=new_id)
    # Matches schemas.user.UserCreate.username max_length; bounds MySQL too.
    username = Column(String(50), nullable=False, unique=True)
    # Descriptive only (username is the identity); nullable for older accounts.
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    # Filename under AVATARS_DIR (written only by app/services/avatar_service.py). Null = no picture; the UI
    # uses `initials` (see Message.sender_avatar_url / sender_initials).
    avatar_path = Column(String(255), nullable=True)
    # bcrypt hash (auth_service.py); plaintext is never stored or logged.
    password_hash = Column(String(255), nullable=False)
    # "admin" unlocks Settings > System (app/routers/settings.py); "user" is everyone else.
    role = Column(String(20), nullable=False, default="user")
    # "active" or "disabled". A disabled account cannot log in and an open session stops on its next request
    # (auth_service.get_current_user). A soft deactivation that keeps history.
    status = Column(String(20), nullable=False, default="active")
    # IANA timezone (e.g. "Asia/Jerusalem") for message timestamps (app/static/js/timezone.js). Starts as
    # DEFAULT_TIMEZONE; changed in Settings > Account.
    timezone = Column(String(64), nullable=False, default=DEFAULT_TIMEZONE, server_default=DEFAULT_TIMEZONE)
    created_at = Column(DateTime, default=utcnow)

    @property
    def avatar_url(self) -> str | None:
        """URL only once the file exists (like MessageAttachment.url); null falls back to `initials`
        (avatar_service.save_avatar).

        `?v=<avatar_path>` is a cache-buster: avatar_path holds a fresh random token per upload (a same-URL
        re-upload showed stale pictures; mtime was too coarse for quick re-uploads).
        """
        if not self.avatar_path:
            return None
        return f"/api/account/{self.id}/avatar?v={self.avatar_path}"

    @property
    def initials(self) -> str:
        """Avatar fallback text: first letters of first/last name, else the username's first letter (never empty)."""
        letters = [name[0] for name in (self.first_name, self.last_name) if name]
        return "".join(letters).upper() if letters else self.username[0].upper()
