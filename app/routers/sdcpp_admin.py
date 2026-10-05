"""
Settings > Image > stable-diffusion.cpp admin endpoints: how to launch `sd-server` on this machine, and
starting/stopping/observing the subprocess. See app/services/sdcpp_service.py — same shape as
app/routers/comfyui_admin.py.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import SDCPP_GITHUB_REPO, SDCPP_PINNED_VERSION
from app.database import get_db
from app.models import User
from app.schemas import (
    AvailableVersions,
    HostHealthCheck,
    InstallDefaults,
    OkResponse,
    SdCppBuild,
    SdCppConfig,
    SdCppStatus,
)
from app.services import (
    github_releases,
    http_proxy_service,
    image_engine_service,
    sdcpp_installer,
    sdcpp_pool,
    sdcpp_service,
    server_pool_broadcast,
)
from app.services.auth_service import require_admin

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/sdcpp", tags=["sdcpp"])


@router.get("/config", response_model=SdCppConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await image_engine_service.get_sdcpp_config(db)


@router.put("/config", response_model=SdCppConfig)
async def update_config(
    body: SdCppConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Persists the new config, applies it to this instance's own live
    pool immediately, and best-effort propagates the same refresh to
    every other local instance (see app.services.server_pool_broadcast —
    each instance independently routes its own image-generation requests
    through app.services.sdcpp_pool, so every one of them needs this,
    not just whichever received this request)."""
    await image_engine_service.set_sdcpp_config(db, body)
    sdcpp_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/sdcpp/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new stable-diffusion.cpp config: %s", "; ".join(failures))
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
    sdcpp_pool.refresh_from_config(await image_engine_service.get_sdcpp_config(db))
    return OkResponse(ok=True)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """See app/routers/ollama_admin.py's identical endpoint — pings
    `host` directly (app.services.sdcpp_pool.check_host), not limited
    to a host already saved to the live pool."""
    return HostHealthCheck(host=host, healthy=await sdcpp_pool.check_host(host))


@router.get("/status", response_model=SdCppStatus)
async def get_status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await sdcpp_service.get_status(db)


@router.post("/start", response_model=SdCppStatus)
async def start(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        return await sdcpp_service.start(db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/stop", response_model=SdCppStatus)
async def stop(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    try:
        return await sdcpp_service.stop(db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/install-defaults", response_model=InstallDefaults)
async def install_defaults(_admin: User = Depends(require_admin)):
    """The maintainer-picked repo/version app.services.sdcpp_installer
    falls back to when SdCppConfig.install_repo/install_version
    haven't been overridden — see app/routers/ollama_admin.py's identical
    endpoint."""
    return InstallDefaults(repo=SDCPP_GITHUB_REPO, version=SDCPP_PINNED_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """See app/routers/ollama_admin.py's identical endpoint."""
    return AvailableVersions(versions=await github_releases.list_tags(repo or SDCPP_GITHUB_REPO))


@router.post("/install")
async def install(
    build: SdCppBuild | None = None,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Clones stable-diffusion.cpp's pinned tag (or the admin's own install_repo/
    install_version override, if set) and installs its dependencies,
    streaming progress over SSE — same wire format as
    /api/settings/pull-model. See app.services.sdcpp_installer's own
    docstring for exactly what it does and why."""
    config = await image_engine_service.get_sdcpp_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)

    async def event_stream():
        async for progress in sdcpp_installer.install_stream(
            config.install_repo, config.install_version, proxy_url=proxy_config.proxy_url(), build=build or config.build
        ):
            yield f"data: {json.dumps(progress)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
