"""Settings > System > Context window: the single `num_ctx` every chat uses (admin-only, system-wide)."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas.settings import ContextWindow
from app.services import context_window_setting
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/context-window", tags=["settings"])


@router.get("", response_model=ContextWindow)
async def get_setting(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ContextWindow(num_ctx=await context_window_setting.get_num_ctx(db))


@router.put("", response_model=ContextWindow)
async def set_setting(body: ContextWindow, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    await context_window_setting.set_num_ctx(db, body.num_ctx)
    return body
