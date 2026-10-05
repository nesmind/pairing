"""Persistence for the image-engine settings: which engine serves the Images page, and the stable-diffusion.cpp
config (see settings_service for ComfyUI's). Split out of settings_service to keep it under the file-size rule."""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import DEFAULT_IMAGE_ENGINE, ImageEngineName, SdCppConfig

ACTIVE_IMAGE_ENGINE_KEY = "active_image_engine"
SDCPP_CONFIG_KEY = "sdcpp_config"


async def _put(db: AsyncSession, key: str, value: dict) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, key))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=key, value=value))
    else:
        row.value = value
    await db.commit()


async def get_active_image_engine(db: AsyncSession) -> ImageEngineName:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, ACTIVE_IMAGE_ENGINE_KEY))
    return row.value["engine"] if row else DEFAULT_IMAGE_ENGINE


async def set_active_image_engine(db: AsyncSession, engine: ImageEngineName) -> None:
    await _put(db, ACTIVE_IMAGE_ENGINE_KEY, {"engine": engine})


async def get_sdcpp_config(db: AsyncSession) -> SdCppConfig:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, SDCPP_CONFIG_KEY))
    return SdCppConfig(**row.value) if row else SdCppConfig()


async def set_sdcpp_config(db: AsyncSession, config: SdCppConfig) -> None:
    await _put(db, SDCPP_CONFIG_KEY, config.model_dump())
