"""GET/PUT for the user's app-wide UI theme (services/theme_service.py, static/css/themes.css). Separate
from settings.py (file-size cap) but under the same "/api/settings" prefix.
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
    """The user's UI theme, settable from Settings' Account tab."""
    return UiTheme(theme=await theme_service.get_ui_theme(db, user))


@router.put("/theme", response_model=UiTheme)
async def set_theme(
    body: UiTheme,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the user's own UI theme."""
    valid_ids = {t["id"] for t in UI_THEMES}
    if body.theme not in valid_ids:
        raise HTTPException(status_code=400, detail=f'Unknown theme "{body.theme}".')
    await theme_service.set_ui_theme(db, user, body.theme)
    return body
