"""
The app-wide UI theme (backgrounds, text, accent colors, fonts — see
app/static/css/themes.css) a user has picked. Split out of
settings_service.py (already at CLAUDE.md's line cap) rather than added
there; mirrors that file's other get_x/set_x per-user setting pairs
(e.g. get_default_model/set_default_model) in shape.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AppSetting, User
from app.theme_config import DEFAULT_UI_THEME, UI_THEMES

THEME_KEY = "ui_theme"


async def get_ui_theme(db: AsyncSession, user: User) -> str:
    """Returns `user`'s own app-wide UI theme (see app.theme_config.UI_THEMES), falling back to
    app.theme_config.DEFAULT_UI_THEME if they haven't picked one or their saved value is no longer a valid
    theme id (e.g. a theme that was later removed from the catalog)."""
    row = await db.get(AppSetting, (user.id, THEME_KEY))
    if row and row.value["theme"] in {t["id"] for t in UI_THEMES}:
        return row.value["theme"]
    return DEFAULT_UI_THEME


async def set_ui_theme(db: AsyncSession, user: User, theme: str) -> None:
    row = await db.get(AppSetting, (user.id, THEME_KEY))
    value = {"theme": theme}
    if row is None:
        db.add(AppSetting(owner_id=user.id, key=THEME_KEY, value=value))
    else:
        row.value = value
    await db.commit()
