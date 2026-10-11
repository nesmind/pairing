"""GET /api/system-metrics/summary: CPU/RAM/disk for the Stats "System" view (system_metrics_service.py).
Admin-only, like stats.py/telemetry.py.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import SystemMetricsSummary
from app.services import system_metrics_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/system-metrics", tags=["system-metrics"])


@router.get("/summary", response_model=SystemMetricsSummary)
async def get_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await system_metrics_service.get_system_metrics_summary(db)
