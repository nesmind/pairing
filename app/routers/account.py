"""Self-service account endpoints (Settings > Account > Profile): own name and picture. Admin management
of other accounts is app/routers/users.py.

Any logged-in user may fetch any avatar (GET.../{user_id}/avatar): channel messages show others'
pictures (Message.sender_avatar_url), and members already see each other's names.
"""

import mimetypes

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import AVATARS_DIR
from app.database import get_db
from app.models import User
from app.schemas import AccountProfileOut, AccountProfileUpdate, PublicProfileOut
from app.services import avatar_service
from app.services.auth_service import get_current_user

router = APIRouter(prefix="/api/account", tags=["account"])


@router.get("/profile", response_model=AccountProfileOut)
async def get_profile(user: User = Depends(get_current_user)):
    return user


@router.patch("/profile", response_model=AccountProfileOut)
async def update_profile(
    body: AccountProfileUpdate,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    # "or None": blank names are stored as NULL (as in user_service.update_user).
    user.first_name = body.first_name or None
    user.last_name = body.last_name or None
    if body.timezone is not None:
        user.timezone = body.timezone
    await db.commit()
    return user


@router.put("/avatar", response_model=AccountProfileOut)
async def upload_avatar(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    return await avatar_service.save_avatar(db, user, file)


@router.delete("/avatar", response_model=AccountProfileOut)
async def remove_avatar(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Reverts to the initials fallback (User.initials)."""
    await avatar_service.delete_avatar(user)
    await db.commit()
    return user


@router.get("/{user_id}/avatar")
async def get_avatar(user_id: str, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    target = await db.get(User, user_id)
    if target is None or not target.avatar_path:
        raise HTTPException(status_code=404, detail="No picture set.")
    media_type = mimetypes.guess_type(target.avatar_path)[0] or "application/octet-stream"
    return FileResponse(AVATARS_DIR / target.avatar_path, media_type=media_type)


@router.get("/{user_id}", response_model=PublicProfileOut)
async def get_public_profile(user_id: str, db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Backs chat.js's avatar-click modal. Registered after GET /profile: FastAPI resolves by route order,
    so "profile" must not match as a user_id.
    """
    target = await db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return target
