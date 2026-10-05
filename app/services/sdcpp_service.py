"""stable-diffusion.cpp process supervision (Settings > Image): start/stop/status of `sd-server`. Same design as
comfyui_service — explicit imperative start/stop, primary instance only, never auto-started/stopped with the app
(see that module's docstring for why)."""

import asyncio

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IS_PRIMARY, SDCPP_HOST
from app.schemas import SdCppStatus
from app.services import image_engine_service, sdcpp_installer, sdcpp_process
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
    installed = sdcpp_installer.is_installed() or bool(config.binary_path)
    if tracked is None or not config.binary_path or not sdcpp_process.is_alive(tracked["pid"], config.binary_path):
        return SdCppStatus(running=False, installed=installed)
    return SdCppStatus(running=True, installed=installed, pid=tracked["pid"], healthy=await _ping_health())


async def _only_installed_model(db: AsyncSession) -> str | None:
    """With exactly one model in the models folder there's nothing to choose — use it. None otherwise."""
    models = (await ImageModelStore.open(db)).list_files()
    return models[0].path if len(models) == 1 else None


async def start(db: AsyncSession) -> SdCppStatus:
    """Raises ValueError (router -> 400) if unconfigured or not the primary instance."""
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

    async with _lock:
        tracked = sdcpp_process.read_tracking()
        if tracked and sdcpp_process.is_alive(tracked["pid"], config.binary_path):
            return await get_status(db)
        pid = sdcpp_process.spawn(config.binary_path, config.model_path, config.extra_args)
        sdcpp_process.write_tracking({"pid": pid} if pid is not None else None)
    return await get_status(db)


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
