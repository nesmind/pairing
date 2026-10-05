"""Persistence for the image-engine settings: which engine serves the Images page, and the stable-diffusion.cpp
config (see settings_service for ComfyUI's). Split out of settings_service to keep it under the file-size rule."""

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import DEFAULT_IMAGE_ENGINE, ImageEngineName, SdCppConfig

ACTIVE_IMAGE_ENGINE_KEY = "active_image_engine"
SDCPP_CONFIG_KEY = "sdcpp_config"
# Engines that may be selected/used. ComfyUI is switched off ahead of its removal: its code is still here, but it
# can't be chosen, and an install that had saved it falls back to the default instead of staying stuck on it.
ENABLED_IMAGE_ENGINES: tuple[str, ...] = ("sdcpp",)


async def _put(db: AsyncSession, key: str, value: dict) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, key))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=key, value=value))
    else:
        row.value = value
    await db.commit()


async def get_active_image_engine(db: AsyncSession) -> ImageEngineName:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, ACTIVE_IMAGE_ENGINE_KEY))
    engine = row.value["engine"] if row else DEFAULT_IMAGE_ENGINE
    return engine if engine in ENABLED_IMAGE_ENGINES else DEFAULT_IMAGE_ENGINE


async def set_active_image_engine(db: AsyncSession, engine: ImageEngineName) -> None:
    """ValueError (-> 400) for an engine that is switched off."""
    if engine not in ENABLED_IMAGE_ENGINES:
        raise ValueError(f"The {engine} image engine is no longer available.")
    await _put(db, ACTIVE_IMAGE_ENGINE_KEY, {"engine": engine})


async def get_sdcpp_config(db: AsyncSession) -> SdCppConfig:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, SDCPP_CONFIG_KEY))
    return SdCppConfig(**row.value) if row else SdCppConfig()


async def set_sdcpp_config(db: AsyncSession, config: SdCppConfig) -> None:
    await _put(db, SDCPP_CONFIG_KEY, config.model_dump())


def check_images_dir(path: str | None) -> None:
    """ValueError (-> 400) unless `path` is blank or an existing, writable, absolute folder (never created)."""
    if not path:
        return
    folder = Path(path).expanduser()
    if not folder.is_absolute():
        raise ValueError(f"The images folder must be an absolute path: {path}")
    if not folder.is_dir():
        raise ValueError(f"The images folder does not exist: {path}")
    if not os.access(folder, os.W_OK | os.X_OK):
        raise ValueError(f"The images folder is not writable: {path}")


async def images_root(db: AsyncSession, default: Path) -> Path:
    """Where generated images are saved: the configured folder, else `default` (app.config.IMAGES_DIR)."""
    path = (await get_sdcpp_config(db)).images_path
    return Path(path).expanduser() if path else default


async def find_image(db: AsyncSession, default: Path, relative: str) -> Path | None:
    """A saved image's file: under the configured root, else under `default` (images made before the folder was
    changed stay where they were - changing it never moves them). None if neither has it."""
    for root in (await images_root(db, default), default):
        if (root / relative).is_file():
            return root / relative
    return None
