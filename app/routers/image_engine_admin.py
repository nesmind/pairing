"""GET/PUT for Settings > Image's "Active image engine" picker — which engine the Images page uses."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ImageEngineConfig
from app.services import image_engine_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/image-engine", tags=["image-engine"])


@router.get("", response_model=ImageEngineConfig)
async def get_image_engine(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ImageEngineConfig(active_image_engine=await image_engine_service.get_active_image_engine(db))


@router.put("", response_model=ImageEngineConfig)
async def set_image_engine(
    body: ImageEngineConfig, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    await image_engine_service.set_active_image_engine(db, body.active_image_engine)
    return body
