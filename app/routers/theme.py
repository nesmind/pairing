"""
GET/PUT for the current user's app-wide UI theme (see
app/services/theme_service.py and app/static/css/themes.css). A
separate router from app/routers/settings.py — that file is already at
CLAUDE.md's line cap — but shares the same "/api/settings" prefix
since, from the Settings page's point of view, this is just another
per-user setting living next to the code theme.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import UiTheme
from app.services import theme_service
from app.services.auth_service import get_current_user
from app.theme_config import UI_THEMES

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/theme", response_model=UiTheme)
async def get_theme(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The current user's own app-wide UI theme — shown/settable from Settings' Account tab."""
    return UiTheme(theme=await theme_service.get_ui_theme(db, user))


@router.put("/theme", response_model=UiTheme)
async def set_theme(
    body: UiTheme,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the current user's own UI theme — never affects any other user's view."""
    valid_ids = {t["id"] for t in UI_THEMES}
    if body.theme not in valid_ids:
        raise HTTPException(status_code=400, detail=f'Unknown theme "{body.theme}".')
    await theme_service.set_ui_theme(db, user, body.theme)
    return body
