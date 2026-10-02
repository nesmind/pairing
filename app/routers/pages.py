"""
The HTML page routes (chat, notes, stats, connectors, images, settings) — split out
of app/main.py purely to keep that file under CLAUDE.md's file-size rule. Every JSON
API endpoint lives in its own resource-specific router elsewhere; these are the only
routes that render a full HTML page via app.templates_env.templates.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.services.auth_service import PageAuth
from app.services.theme_service import get_ui_theme
from app.templates_env import templates
from app.theme_config import UI_THEMES

router = APIRouter()


async def _profile_context(db: AsyncSession, user: User) -> dict:
    """Shared <body>/<html> data-attributes every page but Settings passes to base.html — first/last name +
    avatar_url seed the header's account panel badge (see app/templates/_account_panel.html/account_panel.js) with
    no round trip needed just to render it; theme the app-wide UI theme <html data-theme="..."> renders with,
    ui_themes the catalog _account_panel.html's own Appearance section renders its swatch picker from (mirrors
    settings_page's own ui_themes/theme pair)."""
    return {
        "username": user.username,
        "first_name": user.first_name,
        "last_name": user.last_name,
        "avatar_url": user.avatar_url,
        "timezone": user.timezone,
        "theme": await get_ui_theme(db, user),
        "ui_themes": UI_THEMES,
    }


@router.get("/", response_class=HTMLResponse)
async def chat_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The main chat screen — a single page; conversations are switched
    client-side via the JSON API rather than separate server routes."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    context = {"is_admin": auth.is_admin, "user_id": user.id, **await _profile_context(db, user)}
    return templates.TemplateResponse(request, "chat.html", context)


@router.get("/notes", response_class=HTMLResponse)
async def notes_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Notes screen — create/edit notes and pin them to specific
    conversations as a persona/rules/skill (see app/routers/notes.py and
    app/services/note_service.py)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "notes.html", await _profile_context(db, user))


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Stats screen — a real usage dashboard by default, with "Visual chat workflow", "Telemetry", and
    "System" options on its own page-local left nav (see app/templates/stats.html; unlike notes.html/settings.html's
    centered-column pages, this one is full-bleed with a second sidebar). Admin-only — see app/routers/stats.py's
    own docstring for why; a non-admin who somehow lands here (the sidebar link is already hidden for them) is
    bounced to chat rather than shown a bare 403 page."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_admin(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "stats.html", await _profile_context(db, user))


@router.get("/connectors", response_class=HTMLResponse)
async def connectors_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Connectors screen — configure/enable pluggable third-party engines (see
    app.services.connectors, app/routers/connectors.py). Admin-only, same reasoning
    and non-admin redirect-to-chat behavior as /stats."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_admin(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "connectors.html", await _profile_context(db, user))


@router.get("/images", response_class=HTMLResponse)
async def image_generation_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Image generation screen — type a prompt, get a ComfyUI-backed
    text-to-image result, browse past generations (see
    app/routers/image_generation.py and
    app/services/image_generation_service.py)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "image_generation.html", await _profile_context(db, user))


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The settings/fine-tuning screen. `is_admin` decides whether the
    template renders the System tab (see app/templates/settings.html)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "is_admin": auth.is_admin,
            "username": user.username,
            "ui_themes": UI_THEMES,
            "theme": await get_ui_theme(db, user),
        },
    )
