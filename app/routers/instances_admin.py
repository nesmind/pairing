"""
Settings > System > Local instances admin endpoints: how many local
app-process instances should be running (1-8), and their current
observed status. See app/services/instance_service.py for the
supervisor logic — changing the count here takes effect immediately
(siblings spawned/terminated live), no restart required.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import InstancesConfig, InstancesUpdate
from app.services import instance_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/settings/instances", tags=["instances"])


@router.get("", response_model=InstancesConfig)
async def get_instances(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return await instance_service.get_status(db)


@router.put("", response_model=InstancesConfig)
async def update_instances(
    body: InstancesUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    try:
        return await instance_service.set_instance_count(db, body.count)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
