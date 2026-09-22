"""
FastAPI application factory: wires together the database, API routers, static files, and the HTML pages. This is
the single file `run.py` imports to start the server.
"""

import logging
import time

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.middleware.sessions import SessionMiddleware

from app.config import (
    APP_NAME,
    APP_VERSION,
    BASE_DIR,
    INSTANCE_INDEX,
    SECRET_KEY,
    SESSION_COOKIE_SECURE,
)
from app.database import get_db
from app.models import User
from app.routers import (
    account,
    auth,
    channels,
    chat,
    comfyui_admin,
    conversations,
    db_admin,
    documents,
    engine_admin,
    health,
    http_proxy_admin,
    image_generation,
    instances_admin,
    matricxon_admin,
    model_catalog_admin,
    notes,
    ollama_admin,
    proxy_admin,
    settings,
    stats,
    system_metrics,
    telemetry,
    theme,
    users,
)
from app.services import startup_service
from app.services.auth_service import PageAuth
from app.services.instance_proxy import InstanceProxyMiddleware
from app.services.theme_service import get_ui_theme
from app.templates_env import templates
from app.theme_config import UI_THEMES

logger = logging.getLogger("llama_chat")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title=APP_NAME, description="On-prem, multi-user chat UI for local Ollama models")


class InstanceIndexHeaderMiddleware:
    """Stamps every response with X-Instance-Index: <this process's own app.config.INSTANCE_INDEX> — on every
    process, primary and sibling alike, always (not gated on proxy_mode). Useful on its own (which instance
    answered a given request, matching /health's own transparency style), and is what makes "Local pAIring server
    mode"'s load-balancing actually observable from outside."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Registered outermost, so this also wraps InstanceProxyMiddleware — a proxied response already
                # carries the *serving* instance's own header (stamped before the reply left its own process).
                # Only add one here if missing, or a proxied response would carry two: the correct one from
                # whichever instance served it, plus a wrong one naming the primary that merely forwarded it.
                if not any(k.lower() == b"x-instance-index" for k, _ in message["headers"]):
                    message["headers"].append((b"x-instance-index", str(INSTANCE_INDEX).encode()))
            await send(message)

        await self.app(scope, receive, send_wrapper)


# Adds the signed session cookie every login relies on (see app/services/auth_service.py and app/routers/auth.py)
# — must be registered before any route that reads request.session. https_only defaults to False (see
# SESSION_COOKIE_SECURE's own docstring) so a plain-HTTP local deployment isn't silently broken out of the box.
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, https_only=SESSION_COOKIE_SECURE)

# Starlette's add_middleware() prepends to its own list and builds the stack in reverse, so whichever is registered
# *last* ends up outermost (sees the request first) — these two must come after SessionMiddleware, so a request
# InstanceProxyMiddleware forwards away never touches this process's own session handling (the target instance
# decodes the shared-secret, stateless session cookie independently — see instance_proxy.py's own docstring).
app.add_middleware(InstanceProxyMiddleware)
app.add_middleware(InstanceIndexHeaderMiddleware)

# Serves everything under app/static/ at /static/... (CSS, JS) — plain files, no build step, nothing to compile
# when the project is copied to a new server. Not behind a login check: CSS/JS aren't sensitive, and the login page
# needs them. Absolute path via BASE_DIR (not a relative "app/static" string) since a relative path only resolves
# if the process's working directory happens to be the project root — not guaranteed for every launch method (once
# crashed a systemd-launched instance with "Directory 'app/static' does not exist").
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "app" / "static")), name="static")

# `templates` (imported above) is shared with app/routers/auth.py's login page — see app/templates_env.py for why
# this must be one instance rather than each file creating its own (a global registered on one wouldn't exist on
# the other's).

# Appended as `?v=...` on every /static/ URL in the templates. Changes on every restart, which is exactly when a
# code change might have shipped — without this, a browser that already cached an old script can end up running
# mismatched old/new JS files together, failing in confusing ways instead of just fetching the current one.
STATIC_VERSION = str(int(time.time()))
templates.env.globals["static_version"] = STATIC_VERSION
templates.env.globals["app_version"] = APP_VERSION
templates.env.globals["app_name"] = APP_NAME

# All JSON API endpoints live under their own routers, grouped by resource — see each module's docstring for what it
# covers. Every endpoint in these (besides the auth router itself) requires a logged-in user via
# Depends(get_current_user)/Depends(require_admin).
app.include_router(account.router)
app.include_router(auth.router)
app.include_router(channels.router)
app.include_router(channels.member_router)
app.include_router(conversations.router)
app.include_router(chat.router)
app.include_router(settings.router)
app.include_router(model_catalog_admin.router)
app.include_router(users.router)
app.include_router(db_admin.router)
app.include_router(documents.router)
app.include_router(notes.router)
app.include_router(health.router)
app.include_router(instances_admin.router)
app.include_router(image_generation.router)
app.include_router(comfyui_admin.router)
app.include_router(proxy_admin.router)
app.include_router(http_proxy_admin.router)
app.include_router(ollama_admin.router)
app.include_router(matricxon_admin.router)
app.include_router(engine_admin.router)
app.include_router(stats.router)
app.include_router(telemetry.router)
app.include_router(system_metrics.router)
app.include_router(theme.router)


@app.on_event("startup")
async def on_startup():
    """See app.services.startup_service.run_startup_tasks for the actual work (split out to keep this file compact)."""
    await startup_service.run_startup_tasks()


@app.on_event("shutdown")
async def on_shutdown():
    """See app.services.startup_service.run_shutdown_tasks."""
    await startup_service.run_shutdown_tasks()


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
        "theme": await get_ui_theme(db, user),
        "ui_themes": UI_THEMES,
    }


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The main chat screen — a single page; conversations are switched
    client-side via the JSON API rather than separate server routes."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    context = {"is_admin": auth.is_admin, "user_id": user.id, **await _profile_context(db, user)}
    return templates.TemplateResponse(request, "chat.html", context)


@app.get("/notes", response_class=HTMLResponse)
async def notes_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Notes screen — create/edit notes and pin them to specific
    conversations as a persona/rules/skill (see app/routers/notes.py and
    app/services/note_service.py)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "notes.html", await _profile_context(db, user))


@app.get("/stats", response_class=HTMLResponse)
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


@app.get("/images", response_class=HTMLResponse)
async def image_generation_page(request: Request, db: AsyncSession = Depends(get_db)):
    """The Image generation screen — type a prompt, get a ComfyUI-backed
    text-to-image result, browse past generations (see
    app/routers/image_generation.py and
    app/services/image_generation_service.py)."""
    auth = await PageAuth(request, db).resolve()
    if isinstance(user := auth.require_login(), RedirectResponse):
        return user
    return templates.TemplateResponse(request, "image_generation.html", await _profile_context(db, user))


@app.get("/settings", response_class=HTMLResponse)
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
