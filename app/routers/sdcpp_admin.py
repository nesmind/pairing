"""Settings > Image > stable-diffusion.cpp admin: launch config for `sd-server` and start/stop/status
(app/services/sdcpp_service.py). Same shape as comfyui_admin.py.
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
from app.schemas.cluster import ClusterCheck
from app.services import (
    github_releases,
    http_proxy_service,
    image_engine_service,
    sdcpp_installer,
    sdcpp_pool,
    sdcpp_service,
    server_pool_broadcast,
    settings_service,
)
from app.services.auth_service import require_admin
from app.services.cluster_service import ClusterService

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/sdcpp", tags=["sdcpp"])


@router.get("/config", response_model=SdCppConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await image_engine_service.get_sdcpp_config(db)


async def _save_and_broadcast(db: AsyncSession, body: SdCppConfig) -> None:
    """Persists `body`, applies it to this instance's pool and best-effort broadcasts it to the other local
    instances (server_pool_broadcast; each routes image requests through sdcpp_pool).
    """
    await image_engine_service.set_sdcpp_config(db, body)
    sdcpp_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/sdcpp/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new stable-diffusion.cpp config: %s", "; ".join(failures))


def _check_images_folder(body: SdCppConfig) -> None:
    try:
        image_engine_service.check_images_dir(body.images_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/images-check", response_model=ClusterCheck | None)
async def get_images_check(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> ClusterCheck | None:
    """Warning for the Images folder field: with several instances (cluster mode) it must be on shared
    storage, or an image made on one 404s on the others. None when nothing to say.
    """
    if (await settings_service.get_matricxon_server_config(db)).cluster_role == "off":
        return None
    check = await ClusterService(db).images_check()
    return None if check.status == "ok" else check


@router.put("/config", response_model=SdCppConfig)
async def update_config(
    body: SdCppConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Persists the config; does not restart a running sd-server (POST /apply does). 400 if the images folder is missing
    or not writable."""
    _check_images_folder(body)
    await _save_and_broadcast(db, body)
    return body


@router.post("/apply", response_model=SdCppStatus)
async def apply_config(
    body: SdCppConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Restarts a running sd-server on the new config and saves only if that works; a model that cannot
    load is a 400 with the reason, nothing saved, the previous model back.
    """
    _check_images_folder(body)
    try:
        status = await sdcpp_service.apply(db, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await _save_and_broadcast(db, body)
    return status


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh; see
    ollama_admin.internal_refresh."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    sdcpp_pool.refresh_from_config(await image_engine_service.get_sdcpp_config(db))
    return OkResponse(ok=True)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """See ollama_admin.py; pings `host` directly (sdcpp_pool.check_host)."""
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
    """The repo/version sdcpp_installer uses unless SdCppConfig.install_repo/install_version are overridden (see
    ollama_admin.py)."""
    return InstallDefaults(repo=SDCPP_GITHUB_REPO, version=SDCPP_PINNED_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """See ollama_admin.py's identical endpoint."""
    return AvailableVersions(versions=await github_releases.list_tags(repo or SDCPP_GITHUB_REPO))


@router.post("/install")
async def install(
    build: SdCppBuild | None = None,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Clones stable-diffusion.cpp's pinned tag (or the override) and installs it, streaming SSE like
    /api/settings/pull-model (sdcpp_installer).
    """
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
