"""GET /api/stats/summary — real usage numbers for the Stats page's
default dashboard view (see app/services/stats_service.py for the
aggregation queries and app/templates/stats.html for the page this
serves). Admin-only — unlike the Notes/Images links it used to sit
alongside in chat.html's sidebar, this page also covers the Telemetry
and System views (ML engine call data, this machine's own CPU/RAM/disk),
which read as more operational/infra-facing than something every
regular user should see by default.
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
