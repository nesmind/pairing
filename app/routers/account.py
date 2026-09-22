"""
Self-service account endpoints (Settings > Account > Profile): the
current user's own first/last name and profile picture. Distinct from
app/routers/users.py, which is admin-only management of *other*
accounts (password reset, role/status, no current-password check).

Also serves any user's avatar image (GET .../{user_id}/avatar), open to
any logged-in user, not just its owner — rendering a channel message's
sender avatar means fetching *someone else's* picture (see
app.models.conversation.Message.sender_avatar_url). There's nothing
sensitive about a profile picture within this app's own trust boundary:
every member of a channel already sees the sender's display name.
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
    # "or None" — same normalize-blank-to-NULL reasoning as
    # user_service.update_user's identical pair, so a name cleared from
    # this form doesn't just store an empty string.
    user.first_name = body.first_name or None
    user.last_name = body.last_name or None
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
    """Reverts to the initials fallback — see app.models.user.User.initials."""
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
    """Backs chat.js's "click a message's avatar to see more" modal.
    Registered after GET /profile above (both are technically GET
    /api/account/<one segment>) so a literal "profile" is never matched
    as though it were a user_id here instead — route order is what
    FastAPI actually uses to resolve that, not any kind of specificity
    ranking."""
    target = await db.get(User, user_id)
    if target is None:
        raise HTTPException(status_code=404, detail="User not found.")
    return target
