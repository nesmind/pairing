"""Settings > Inference engines > Ollama admin: local-mode params (num_parallel/keep_alive/max_loaded_models/
context_length) or remote host list, and start/stop/status of the local `ollama serve`
(app/services/ollama_process.py). Same shape as sdcpp_admin.py.

Not to be confused with app/services/ollama_admin.py (model pull/delete). Replaces the old
"Ollama concurrency" endpoints (num_parallel only, restarted on save).
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
from app.services.extended_model_catalog_service import ExtendedModelCatalog

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/ollama", tags=["ollama"])


async def _save_and_broadcast(db: AsyncSession, body: OllamaServerConfig) -> None:
    """Persists `body`, applies it to this instance's pool and best-effort broadcasts it to the other local
    instances (server_pool_broadcast; each routes chat through ollama_pool). A sibling that fails keeps its
    stale host list until the next save or restart: logged, not raised.
    """
    previous = await settings_service.get_ollama_server_config(db)
    await settings_service.set_ollama_server_config(db, body)
    if previous.mode != body.mode:
        await ExtendedModelCatalog(db).clear()
    ollama_pool.refresh_from_config(body)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/ollama/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new Ollama config: %s", "; ".join(failures))


@router.get("/config", response_model=OllamaServerConfig)
async def get_config(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await settings_service.get_ollama_server_config(db)


@router.get("/check-host", response_model=HostHealthCheck)
async def check_host(host: str, _admin: User = Depends(require_admin)):
    """Pings `host` directly (ollama_pool.check_host), even if not in the live pool, so an admin can verify a
    remote URL before saving.
    """
    return HostHealthCheck(host=host, healthy=await ollama_pool.check_host(host))


@router.get("/auto-detected-path", response_model=OllamaAutoDetectedPath)
async def auto_detected_path(_admin: User = Depends(require_admin)):
    """What ollama_process.auto_detect_binary resolves to now (PATH, then fixed locations), ignoring any
    saved override; shown as the binary-path field's placeholder.
    """
    return OllamaAutoDetectedPath(
        path=ollama_process.auto_detect_binary(), models_path=ollama_process.auto_detect_models_path()
    )


@router.put("/config", response_model=OllamaServerConfig)
async def update_config(
    body: OllamaServerConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Persists the config and refreshes the pool on every instance; does not restart a running local Ollama
    (POST /apply does, since that interrupts in-flight replies and the frontend warns first).
    """
    await _save_and_broadcast(db, body)
    return body


@router.post("/apply", response_model=OllamaServerStatus)
async def apply_config(
    body: OllamaServerConfig,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Saves `body` and, if Ollama runs locally, restarts it with the new params: the one action here that
    cannot apply live. The frontend confirms first.
    """
    await _save_and_broadcast(db, body)
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        await ExtendedModelCatalog(db).clear()
        return await ollama_process.apply_local_config(body, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh, never by a browser.
    Under "/api/settings/ollama", covered by instance_proxy_http.EXEMPT_PREFIXES, so it is never
    load-balanced to another instance (see db_admin.internal_switch_database: a proxied request looks
    loopback-sourced and would defeat the check).
    """
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
        await ExtendedModelCatalog(db).clear()
        return await ollama_process.start(config, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/stop", response_model=OllamaServerStatus)
async def stop(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    config = await settings_service.get_ollama_server_config(db)
    try:
        await ExtendedModelCatalog(db).clear()
        return await ollama_process.stop(config.binary_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/install-defaults", response_model=InstallDefaults)
async def install_defaults(_admin: User = Depends(require_admin)):
    """The repo/version ollama_installer uses unless OllamaServerConfig.install_repo/install_version are
    overridden; shown next to Install and as the override placeholders.
    """
    return InstallDefaults(repo=OLLAMA_GITHUB_REPO, version=OLLAMA_PINNED_VERSION)


@router.get("/available-versions", response_model=AvailableVersions)
async def available_versions(repo: str | None = None, _admin: User = Depends(require_admin)):
    """Tags fetched live from GitHub (github_releases) for `repo` (the unsaved override, else
    OLLAMA_GITHUB_REPO) for the version picker's datalist; a convenience, never required.
    """
    return AvailableVersions(versions=await github_releases.list_tags(repo or OLLAMA_GITHUB_REPO))


@router.post("/install")
async def install(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Downloads and installs Ollama's pinned release (or the override; ollama_installer.install_stream),
    streaming SSE like /api/settings/pull-model. A multi-GB download.
    """
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
