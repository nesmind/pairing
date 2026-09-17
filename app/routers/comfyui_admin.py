"""
Settings > External servers > ComfyUI admin endpoints: how to launch
ComfyUI on this machine, and starting/stopping/observing the actual
subprocess. See app/services/comfyui_service.py for the supervisor
logic — mirrors app/routers/instances_admin.py's shape, and
app/routers/ollama_admin.py's identical shape for Ollama's own half of
the same "External servers" tab.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import COMFYUI_GITHUB_REPO, COMFYUI_PINNED_VERSION
from app.database import get_db
from app.models import User
from app.schemas import ComfyUIProcessConfig, ComfyUIStatus, HostHealthCheck, InstallDefaults, OkResponse
from app.services import (
    comfyui_installer,
    comfyui_pool,
    comfyui_service,
    http_proxy_service,
    server_pool_broadcast,
    settings_service,
)
from app.services.auth_service import require_admin

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/comfyui", tags=["comfyui"])


@router.get("/config", response_model=ComfyUIProcessConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await settings_service.get_comfyui_config(db)


@router.put("/config", response_model=ComfyUIProcessConfig)
async def update_config(
    body: ComfyUIProcessConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Persists the new config, applies it to this instance's own live
    pool immediately, and best-effort propagates the same refresh to
    every other local instance (see app.services.server_pool_broadcast —
    each instance independently routes its own image-generation requests
    through app.services.comfyui_pool, so every one of them needs this,
    not just whichever received this request)."""
    await settings_service.set_comfyui_config(db, body)
    comfyui_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/comfyui/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new ComfyUI config: %s", "; ".join(failures))
    return body


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's own
    app.services.server_pool_broadcast.broadcast_refresh — see
    app/routers/ollama_admin.py's identical endpoint for the full
    reasoning (loopback-only, never proxied since this path already
    falls under this router's own EXEMPT_PREFIXES entry)."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    comfyui_pool.refresh_from_config(await settings_service.get_comfyui_config(db))
    return OkResponse(ok=True)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """See app/routers/ollama_admin.py's identical endpoint — pings
    `host` directly (app.services.comfyui_pool.check_host), not limited
    to a host already saved to the live pool."""
    return HostHealthCheck(host=host, healthy=await comfyui_pool.check_host(host))


@router.get("/status", response_model=ComfyUIStatus)
async def get_status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await comfyui_service.get_status(db)


@router.post("/start", response_model=ComfyUIStatus)
async def start(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        return await comfyui_service.start(db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop", response_model=ComfyUIStatus)
async def stop(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        return await comfyui_service.stop(db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/install-defaults", response_model=InstallDefaults)
async def install_defaults(_admin: User = Depends(require_admin)):
    """The maintainer-picked repo/version app.services.comfyui_installer
    falls back to when ComfyUIProcessConfig.install_repo/install_version
    haven't been overridden — see app/routers/ollama_admin.py's identical
    endpoint."""
    return InstallDefaults(repo=COMFYUI_GITHUB_REPO, version=COMFYUI_PINNED_VERSION)


@router.post("/install")
async def install(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Clones ComfyUI's pinned tag (or the admin's own install_repo/
    install_version override, if set) and installs its dependencies,
    streaming progress over SSE — same wire format as
    /api/settings/pull-model. See app.services.comfyui_installer's own
    docstring for exactly what it does and why."""
    config = await settings_service.get_comfyui_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)

    async def event_stream():
        async for progress in comfyui_installer.install_stream(
            config.install_repo, config.install_version, proxy_url=proxy_config.proxy_url()
        ):
            yield f"data: {json.dumps(progress)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
