"""stable-diffusion.cpp process supervision (Settings > Image): start/stop/status of `sd-server`. Same design as
comfyui_service — explicit imperative start/stop, primary instance only, never auto-started/stopped with the app
(see that module's docstring for why)."""

import asyncio
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IS_PRIMARY, SDCPP_HOST
from app.schemas import SdCppConfig, SdCppStatus
from app.services import image_engine_service, sdcpp_installer, sdcpp_process, sdcpp_source_build
from app.services.image_model_service import ImageModelStore

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_lock = asyncio.Lock()


async def _ping_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            return (await client.get(SDCPP_HOST)).status_code == 200
    except httpx.HTTPError:
        return False


async def get_status(db: AsyncSession) -> SdCppStatus:
    tracked = sdcpp_process.read_tracking()
    config = await image_engine_service.get_sdcpp_config(db)
    # A saved path counts only while the file is still there (a removed install left it behind).
    installed = sdcpp_installer.is_installed() or bool(config.binary_path and Path(config.binary_path).is_file())
    missing = sdcpp_source_build.missing_tools()
    if tracked is None or not config.binary_path or not sdcpp_process.is_alive(tracked["pid"], config.binary_path):
        return SdCppStatus(running=False, installed=installed, source_build_missing=missing)
    return SdCppStatus(
        running=True,
        installed=installed,
        pid=tracked["pid"],
        healthy=await _ping_health(),
        source_build_missing=missing,
    )


async def _only_installed_model(db: AsyncSession) -> str | None:
    """With exactly one model in the models folder there's nothing to choose — use it. None otherwise."""
    models = (await ImageModelStore.open(db)).list_files()
    return models[0].path if len(models) == 1 else None


async def start(db: AsyncSession) -> SdCppStatus:
    """Raises ValueError (router -> 400) if unconfigured, not the primary instance, or the model fails to load."""
    if not IS_PRIMARY:
        raise ValueError("stable-diffusion.cpp can only be managed from the primary instance.")
    config = await image_engine_service.get_sdcpp_config(db)
    if not config.binary_path:
        raise ValueError("Set the sd-server path before starting stable-diffusion.cpp.")
    if not config.model_path:
        config.model_path = await _only_installed_model(db)
        if not config.model_path:
            raise ValueError(
                "Choose a model in the Model dropdown first (stable-diffusion.cpp serves one model at a time) — "
                "download one under Settings > Model > Browse more models if the list is empty."
            )
        await image_engine_service.set_sdcpp_config(db, config)

    await _launch(config)
    return await get_status(db)


async def _launch(config: SdCppConfig) -> None:
    """Spawns sd-server for `config` unless one already runs; StartupError if it can't come up."""
    async with _lock:
        tracked = sdcpp_process.read_tracking()
        if tracked and sdcpp_process.is_alive(tracked["pid"], config.binary_path):
            return
        try:  # blocks until sd-server is listening - keep it off the event loop
            pid = await asyncio.to_thread(sdcpp_process.spawn, config.binary_path, config.model_path, config.extra_args)
        except sdcpp_process.StartupError:
            sdcpp_process.write_tracking(None)
            raise
        sdcpp_process.write_tracking({"pid": pid} if pid is not None else None)


async def stop(db: AsyncSession) -> SdCppStatus:
    if not IS_PRIMARY:
        raise ValueError("stable-diffusion.cpp can only be managed from the primary instance.")
    config = await image_engine_service.get_sdcpp_config(db)

    async with _lock:
        tracked = sdcpp_process.read_tracking()
        if tracked and config.binary_path:
            sdcpp_process.terminate(tracked["pid"], config.binary_path)
        sdcpp_process.write_tracking(None)
    return await get_status(db)


async def apply(db: AsyncSession, new: SdCppConfig) -> SdCppStatus:
    """Validates `new` by really starting it, before the caller saves anything: a running sd-server is
    restarted on `new`; if that fails the previous config is brought back and ValueError says why. A stopped
    server is left stopped (nothing to try it on - `start` reports the problem later)."""
    if not (await get_status(db)).running:
        return await get_status(db)
    old = await image_engine_service.get_sdcpp_config(db)
    await stop(db)
    try:
        await _launch(new)
    except sdcpp_process.StartupError as exc:
        restored = await _restore(old)
        raise ValueError(
            f"{exc} Nothing was saved{'; the previous model is running again' if restored else ''}."
        ) from exc
    return await get_status(db)


async def _restore(old: SdCppConfig) -> bool:
    try:
        await _launch(old)
    except (sdcpp_process.StartupError, ValueError, OSError):
        return False
    return True
