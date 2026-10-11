"""Validation and storage of a user's profile picture (Settings > Account; AVATARS_DIR, MAX_AVATAR_MB).
One file per user, overwritten rather than accumulated.

The initials fallback lives on User.initials/avatar_url; this module owns only the file.
"""

import secrets
from pathlib import Path

from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AVATARS_DIR, MAX_AVATAR_MB
from app.models import User

SUPPORTED_AVATAR_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


async def save_avatar(db: AsyncSession, user: User, upload: UploadFile) -> User:
    """Validates `upload` (extension, size) and writes AVATARS_DIR/<user.id>-<random>.<ext>, deleting the old
    file first. The random part gives a new URL even when the extension is unchanged (see User.avatar_url:
    reused names showed stale cached pictures). Raises HTTPException(400) for unsupported or oversized
    files, leaving the existing picture.
    """
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
    """Removes the picture file, if any, and clears `user.avatar_path`. The caller commits; only the file and attribute
    change."""
    if user.avatar_path:
        (AVATARS_DIR / user.avatar_path).unlink(missing_ok=True)
        user.avatar_path = None
