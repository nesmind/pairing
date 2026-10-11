"""Settings > External servers > Matricxon admin: local-mode params (max_loaded_models) or remote host list,
and start/stop/status of the local process via its scripts. Same shape as ollama_admin.py, except
Matricxon is a sibling checkout plus scripts, not a binary (matricxon_process.py).

Not to be confused with app/services/matricxon_admin.py (model pull/delete).
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import MATRICXON_DEFAULT_VERSION, MATRICXON_GITHUB_REPO
from app.database import get_db
from app.models import User
from app.schemas import (
    AvailableVersions,
    HostHealthCheck,
    InstallDefaults,
    MatricxonAutoDetectedPath,
    MatricxonServerConfig,
    MatricxonServerStatus,
    OkResponse,
)
from app.services import (
    engine_service,
    github_releases,
    http_proxy_service,
    matricxon_installer,
    matricxon_pool,
    matricxon_process,
    server_pool_broadcast,
    settings_service,
)
from app.services.auth_service import require_admin
from app.services.cluster_hosts import ClusterHosts
from app.services.extended_model_catalog_service import ExtendedModelCatalog

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/matricxon", tags=["matricxon"])


def _without_secret(config: MatricxonServerConfig) -> MatricxonServerConfig:
    """The admin secret never goes to the browser."""
    return config.model_copy(update={"admin_secret": None})


async def _save_and_broadcast(db: AsyncSession, body: MatricxonServerConfig) -> None:
    """Persists `body`, writes Matricxon's.env (matricxon_process.write_env_file, so it applies however the
    process starts) and, only if Matricxon is the active engine, refreshes this instance's matricxon_pool and
    best-effort the other local instances. An inactive engine's config is saved but has no live effect until
    switched to (engine_admin.py); ollama_admin does it unconditionally.
    """
    previous = await settings_service.get_matricxon_server_config(db)
    # Nodes added on the Cluster page are not part of this form; saving must not wipe them (it once did).
    ClusterHosts.keep_through_a_save(previous, body)
    await settings_service.set_matricxon_server_config(db, body)
    if previous.mode != body.mode:
        await ExtendedModelCatalog(db).clear()
    matricxon_process.write_env_file(body)
    if await engine_service.get_active_engine(db) == "matricxon":
        matricxon_pool.refresh_from_config(body)
        failures = await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
        if failures:
            logger.warning("Some instances did not pick up the new Matricxon config: %s", "; ".join(failures))


@router.get("/config", response_model=MatricxonServerConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return _without_secret(await settings_service.get_matricxon_server_config(db))


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    return HostHealthCheck(host=host, healthy=await matricxon_pool.check_host(host))


@router.get("/auto-detected-path", response_model=MatricxonAutoDetectedPath)
async def auto_detected_path(_admin: User = Depends(require_admin)):
    return MatricxonAutoDetectedPath(
        path=matricxon_process.auto_detect_project_dir(), models_path=matricxon_process.auto_detect_models_dir()
    )


@router.put("/config", response_model=MatricxonServerConfig)
async def update_config(
    body: MatricxonServerConfig, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    """Persists the config; does not restart a running local Matricxon (POST /apply does), as
    ollama_admin.update_config."""
    await _save_and_broadcast(db, body)
    return _without_secret(body)


@router.post("/apply", response_model=MatricxonServerStatus)
async def apply_config(
    body: MatricxonServerConfig, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    await _save_and_broadcast(db, body)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        await ExtendedModelCatalog(db).clear()
        return await matricxon_process.apply_local_config(body, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh; see
    ollama_admin.internal_refresh."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    matricxon_pool.refresh_from_config(await settings_service.get_matricxon_server_config(db))
    return OkResponse(ok=True)


@router.get("/status", response_model=MatricxonServerStatus)
async def get_status(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_matricxon_server_config(db)
    return await matricxon_process.get_status(config.project_dir)


@router.post("/start", response_model=MatricxonServerStatus)
async def start(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_matricxon_server_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        await ExtendedModelCatalog(db).clear()
        return await matricxon_process.start(config, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop", response_model=MatricxonServerStatus)
async def stop(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_matricxon_server_config(db)
    try:
        await ExtendedModelCatalog(db).clear()
        return await matricxon_process.stop(config.project_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/install-defaults", response_model=InstallDefaults)
async def install_defaults(_admin: User = Depends(require_admin)):
    return InstallDefaults(repo=MATRICXON_GITHUB_REPO, version=MATRICXON_DEFAULT_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """See ollama_admin.py's identical endpoint."""
    return AvailableVersions(versions=await github_releases.list_tags(repo or MATRICXON_GITHUB_REPO))


@router.post("/install")
async def install(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Same SSE shape as ollama_admin's install; matricxon_installer always yields a single {"error":...} today."""
    config = await settings_service.get_matricxon_server_config(db)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)

    async def event_stream():
        async for progress in matricxon_installer.install_stream(
            config.install_repo, config.install_version, proxy_url=proxy_config.proxy_url()
        ):
            yield f"data: {json.dumps(progress)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
