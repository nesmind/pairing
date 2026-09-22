"""GET /api/telemetry/{ollama,matricxon}/summary — per-engine call telemetry for the Stats page's "Telemetry"
view (see app/services/telemetry_service.py for the aggregation queries). Admin-only, same access-control
decision as app/routers/stats.py's own summary endpoint (see that file's own docstring). Two literal routes
(not one route with an `engine` path/query param) matches this codebase's existing convention of a real,
named endpoint per engine rather than a validated-string param — same as ollama_admin.py/matricxon_admin.py
being separate routers instead of one parameterized by engine name."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import OllamaTelemetrySummary
from app.services import telemetry_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/ollama/summary", response_model=OllamaTelemetrySummary)
async def get_ollama_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await telemetry_service.get_engine_telemetry_summary(db, "ollama")


@router.get("/matricxon/summary", response_model=OllamaTelemetrySummary)
async def get_matricxon_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await telemetry_service.get_engine_telemetry_summary(db, "matricxon")
