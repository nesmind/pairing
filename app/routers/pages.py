"""HTML page routes (chat, notes, stats, connectors, images, settings), split from app/main.py for the
file-size rule. JSON endpoints live in resource routers; these render full pages via templates_env.templates.
"""

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.services.auth_service import PageAuth
from app.services.engines.registry import registry
from app.services.theme_service import get_ui_theme
from app.templates_env import templates
from app.theme_config import UI_THEMES

router = APIRouter()


async def _profile_context(db: AsyncSession, user: User) -> dict:
    """Data-attributes every page except Settings passes to base.html: names and avatar_url seed the header
    account badge (_account_panel.html/account_panel.js); theme sets <html data-theme>; ui_themes feeds the
    Appearance swatch picker (like settings_page).
    """
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
    """The chat screen: one page; conversations switch client-side via the JSON API."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    context = {"is_admin": auth.is_admin, "user_id": user.id, **await _profile_context(db, user)}
    return templates.TemplateResponse(request, "chat.html", context)


@router.get("/notes", response_class=HTMLResponse)
async def notes_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Notes screen: create/edit notes and pin them as persona/rules/skill (routers/notes.py, note_service.py)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "notes.html", await _profile_context(db, user))


@router.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Stats screen: usage dashboard, with "Visual chat workflow", "Telemetry" and "System" in its own
    left nav (full-bleed, stats.html). Admin-only (see routers/stats.py); non-admins are bounced to chat.
    """
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_admin(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "stats.html", await _profile_context(db, user))


@router.get("/cluster", response_class=HTMLResponse)
async def cluster_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Cluster screen: full-bleed like /stats with its own nav; cluster mode and readiness checklist
    (routers/cluster_admin.py). Admin-only, same redirect as /stats.
    """
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_admin(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "cluster.html", await _profile_context(db, user))


@router.get("/connectors", response_class=HTMLResponse)
async def connectors_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Connectors screen: configure pluggable third-party engines (services.connectors,
    routers/connectors.py). Admin-only, same redirect as /stats.
    """
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_admin(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "connectors.html", await _profile_context(db, user))


@router.get("/images", response_class=HTMLResponse)
async def image_generation_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Image generation screen: prompt, active image engine, past generations
    (routers/image_generation.py, services/image_generation_service.py).
    """
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "image_generation.html", await _profile_context(db, user))


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    """User-level settings, same page for every account (admin functions are on /admin-settings).
    `is_admin` is always False here (templates and settings.js use it for tabs/controls); `can_admin` only drives the
    admin link.
    """
    return await _settings_response(request, db, admin_page=False)


@router.get("/admin-settings", response_class=HTMLResponse)
async def admin_settings_page(request: Request, db: AsyncSession = Depends(get_db)):
    """Admin settings (System, External servers, MCP servers, Users, Channels, admin Models), same layout as
    /settings. Non-admins go back to /settings.
    """
    return await _settings_response(request, db, admin_page=True)


async def _settings_response(request: Request, db: AsyncSession, admin_page: bool):
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    if admin_page and not auth.is_admin:
        return RedirectResponse("/settings", status_code=303)
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "is_admin": admin_page,
            "admin_page": admin_page,
            "can_admin": auth.is_admin,
            "username": user.username,
            "ui_themes": UI_THEMES,
            "theme": await get_ui_theme(db, user),
            # A connector engine manages its own models: its download sections start hidden, no flash on load.
            "remote_models": not registry.active().capabilities.model_management,
        },
    )
