"""
Settings > External servers > Ollama admin endpoints: local-mode
parameters (num_parallel/keep_alive/max_loaded_models/context_length) or
remote-mode host list, and starting/stopping/observing the actual local
`ollama serve` subprocess. See app/services/ollama_process.py for the
supervisor logic — mirrors app/routers/comfyui_admin.py's identical
shape for ComfyUI's own half of the same tab.

Named ollama_admin.py for the Settings-page-admin-endpoint convention
this project uses (db_admin.py, instances_admin.py, comfyui_admin.py) —
distinct from app/services/ollama_admin.py (model pull/delete), an
unrelated existing module this doesn't touch. Replaces the old, narrower
"Ollama concurrency" endpoints (num_parallel only, always restarted
Ollama immediately on save) — that one field is now part of the fuller
config below.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import OLLAMA_GITHUB_REPO, OLLAMA_PINNED_VERSION
from app.database import get_db
from app.models import User
from app.schemas import (
    AvailableVersions,
    HostHealthCheck,
    InstallDefaults,
    OkResponse,
    OllamaAutoDetectedPath,
    OllamaServerConfig,
    OllamaServerStatus,
)
from app.services import (
    github_releases,
    http_proxy_service,
    ollama_installer,
    ollama_pool,
    ollama_process,
    server_pool_broadcast,
    settings_service,
)
from app.services.auth_service import require_admin

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/ollama", tags=["ollama"])


async def _save_and_broadcast(db: AsyncSession, body: OllamaServerConfig) -> None:
    """Persists `body`, applies it to this instance's own live pool
    immediately, and best-effort propagates the same refresh to every
    other local instance (see app.services.server_pool_broadcast — each
    instance independently routes its own chat requests through
    app.services.ollama_pool, so every one of them needs this, not just
    whichever received this request). A sibling that couldn't be
    refreshed just keeps routing through its own stale host list until
    its next save or a restart — logged, not raised, since this
    process's own change (already applied above) is the one that
    matters most and did succeed."""
    await settings_service.set_ollama_server_config(db, body)
    ollama_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/ollama/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new Ollama config: %s", "; ".join(failures))


@router.get("/config", response_model=OllamaServerConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await settings_service.get_ollama_server_config(db)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """Pings `host` directly (see app.services.ollama_pool.check_host) —
    not limited to a host already saved to the live pool, so Settings >
    External servers can let an admin verify a remote-mode candidate URL
    is reachable before ever clicking Save on the form."""
    return HostHealthCheck(host=host, healthy=await ollama_pool.check_host(host))


@router.get("/auto-detected-path", response_model=OllamaAutoDetectedPath)
async def auto_detected_path(_admin: User = Depends(require_admin)):
    """What ollama_process.auto_detect_binary() (PATH, then the usual
    fixed fallback locations) resolves to right now, ignoring any saved
    binary_path override — the "Ollama binary path" field's own
    placeholder shows this real value so "leave it blank" is concrete,
    not just a description."""
    return OllamaAutoDetectedPath(
        path=ollama_process.auto_detect_binary(), models_path=ollama_process.auto_detect_models_path()
    )


@router.put("/config", response_model=OllamaServerConfig)
async def update_config(
    body: OllamaServerConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Persists the new config and refreshes the live host pool on every
    instance — does not itself restart a currently-running local Ollama
    (see POST /apply for that, since restarting interrupts every
    in-flight reply and the frontend should warn before triggering it)."""
    await _save_and_broadcast(db, body)
    return body


@router.post("/apply", response_model=OllamaServerStatus)
async def apply_config(
    body: OllamaServerConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Saves `body` and, if Ollama is running locally, restarts it with
    the new local-mode parameters applied — the one action in this tab
    that can't apply live. The frontend is expected to confirm with the
    admin before calling this (same as the old "Ollama concurrency"
    save button already did)."""
    await _save_and_broadcast(db, body)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        return await ollama_process.apply_local_config(body, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's own
    app.services.server_pool_broadcast.broadcast_refresh right after an
    admin's config save — never by a browser, so no require_admin
    session here. Deliberately placed under this router's own
    "/api/settings/ollama" prefix, already covered by
    app.services.instance_proxy_http.EXEMPT_PREFIXES, so this can never
    get load-balanced to a random *different* instance — see
    app/routers/db_admin.py's internal_switch_database for the identical
    reasoning (a proxied request always looks loopback-sourced, which
    would defeat this check if this path could ever be proxied)."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    ollama_pool.refresh_from_config(await settings_service.get_ollama_server_config(db))
    return OkResponse(ok=True)


@router.get("/status", response_model=OllamaServerStatus)
async def get_status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_ollama_server_config(db)
    return await ollama_process.get_status(config.binary_path)


@router.post("/start", response_model=OllamaServerStatus)
async def start(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_ollama_server_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        return await ollama_process.start(config, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop", response_model=OllamaServerStatus)
async def stop(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_ollama_server_config(db)
    try:
        return await ollama_process.stop(config.binary_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/install-defaults", response_model=InstallDefaults)
async def install_defaults(_admin: User = Depends(require_admin)):
    """The maintainer-picked repo/version app.services.ollama_installer
    falls back to when OllamaServerConfig.install_repo/install_version
    haven't been overridden — Settings > External servers shows this next
    to the Install button and pre-fills the override fields' placeholders
    with it."""
    return InstallDefaults(repo=OLLAMA_GITHUB_REPO, version=OLLAMA_PINNED_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """Real tags fetched live from GitHub (see app.services.github_releases) for `repo` — the admin's own
    not-yet-saved install_repo override if they're previewing one, else OLLAMA_GITHUB_REPO's own default.
    Backs the version picker's datalist next to the plain install_version override field; see that schema's own
    docstring for why this is always a convenience, never a hard requirement."""
    return AvailableVersions(versions=await github_releases.list_tags(repo or OLLAMA_GITHUB_REPO))


@router.post("/install")
async def install(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Downloads and installs Ollama's pinned release (or the admin's own
    install_repo/install_version override, if set — see
    app.services.ollama_installer.install_stream), streaming progress
    over SSE — same wire format as /api/settings/pull-model. A real,
    multi-GB download; see app.services.ollama_installer's own docstring
    for exactly what it does and why."""
    config = await settings_service.get_ollama_server_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)

    async def event_stream():
        async for progress in ollama_installer.install_stream(
            config.install_repo, config.install_version, proxy_url=proxy_config.proxy_url()
        ):
            yield f"data: {json.dumps(progress)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
