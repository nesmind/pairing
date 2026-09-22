"""
Validation and on-disk storage for a user's own profile picture (Settings
> Account) — see app.config.AVATARS_DIR/MAX_AVATAR_MB. One file per user,
always overwritten in place rather than accumulating like
chat_attachment_service's per-message attachments do, since only the most
recently uploaded picture is ever shown.

Display-side fallback (initials, or the username's first letter, when no
picture is set) lives on User.initials/avatar_url in app/models — this
module only owns the file itself.
"""

import secrets
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AVATARS_DIR, MAX_AVATAR_MB
from app.models import User

SUPPORTED_AVATAR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


async def save_avatar(db: AsyncSession, user: User, upload: UploadFile) -> User:
    """Validates `upload` (extension + size) and writes it to
    AVATARS_DIR/<user.id>-<random>.<ext>, replacing whatever was there
    before — deletes the old file first so a jpg-then-png re-upload
    doesn't leave an orphaned jpg behind. The random component (not just
    <user.id><ext>, which this used before) means a re-upload always gets
    a genuinely new filename/URL even when the extension doesn't change —
    see User.avatar_url's own docstring on why that's load-bearing, not
    cosmetic: reusing the exact same filename left browsers showing the
    stale cached picture after a re-upload, confirmed live. Raises
    HTTPException(400) for anything unsupported or oversized, without
    touching the existing picture."""
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in SUPPORTED_AVATAR_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported picture type: {suffix or 'unknown'}")

    raw = await upload.read()
    max_bytes = MAX_AVATAR_MB * 1024 * 1024
    if len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Picture exceeds the {MAX_AVATAR_MB}MB limit.")

    await delete_avatar(user)
    stored_name = f"{user.id}-{secrets.token_hex(4)}{suffix}"
    (AVATARS_DIR / stored_name).write_bytes(raw)
    user.avatar_path = stored_name
    await db.commit()
    return user


async def delete_avatar(user: User) -> None:
    """Removes the current picture file, if any, and clears
    `user.avatar_path` — used both by DELETE .../avatar (revert to
    initials) and by save_avatar above (before writing a replacement).
    Caller is responsible for committing `user`'s own session; this only
    touches the filesystem and the in-memory attribute."""
    if user.avatar_path:
        (AVATARS_DIR / user.avatar_path).unlink(missing_ok=True)
        user.avatar_path = None
