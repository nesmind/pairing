"""GET /api/system-metrics/summary — local server CPU/RAM/disk usage for
the Stats page's "System" view (see app/services/system_metrics_service.py
for the query). Admin-only, same access-control decision as
app/routers/stats.py/telemetry.py (see stats.py's own docstring)."""

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
