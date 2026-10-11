"""GET /api/stats/summary: usage numbers for the Stats dashboard (services/stats_service.py, stats.html).
Admin-only: the page also covers Telemetry and System views, which are operational.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import EngineSupportResponse, StatsSummary
from app.services import stats_service
from app.services.auth_service import require_admin
from app.services.engine_support_service import EngineSupportService

router = APIRouter(prefix="/api/stats", tags=["stats"])


@router.get("/summary", response_model=StatsSummary)
async def get_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await stats_service.get_stats_summary(db)


@router.get("/engine-support", response_model=EngineSupportResponse)
async def get_engine_support(_user: User = Depends(require_admin)) -> EngineSupportResponse:
    return await EngineSupportService.get()
