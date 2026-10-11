"""Settings > External servers > ComfyUI admin: launch config and start/stop/status of its subprocess
(app/services/comfyui_service.py). Same shape as instances_admin.py and ollama_admin.py.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import COMFYUI_GITHUB_REPO, COMFYUI_PINNED_VERSION
from app.database import get_db
from app.models import User
from app.schemas import (
    AvailableVersions,
    ComfyUIProcessConfig,
    ComfyUIStatus,
    HostHealthCheck,
    InstallDefaults,
    OkResponse,
)
from app.services import (
    comfyui_installer,
    comfyui_pool,
    comfyui_service,
    github_releases,
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
    """Persists the config, applies it to this instance's pool and best-effort broadcasts it to the other local
    instances (server_pool_broadcast; each routes image requests through comfyui_pool).
    """
    await settings_service.set_comfyui_config(db, body)
    comfyui_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/comfyui/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new ComfyUI config: %s", "; ".join(failures))
    return body


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh; see ollama_admin.py's
    identical endpoint (loopback-only, never proxied: path is in EXEMPT_PREFIXES).
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    comfyui_pool.refresh_from_config(await settings_service.get_comfyui_config(db))
    return OkResponse(ok=True)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """Pings `host` directly (comfyui_pool.check_host), even if not in the live pool. See ollama_admin.py."""
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
    """The repo/version comfyui_installer uses unless install_repo/install_version are overridden."""
    return InstallDefaults(repo=COMFYUI_GITHUB_REPO, version=COMFYUI_PINNED_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """See ollama_admin.py's identical endpoint."""
    return AvailableVersions(versions=await github_releases.list_tags(repo or COMFYUI_GITHUB_REPO))


@router.post("/install")
async def install(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Clones ComfyUI's pinned tag (or the admin's override) and installs dependencies, streaming progress over
    SSE like /api/settings/pull-model (see comfyui_installer).
    """
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
