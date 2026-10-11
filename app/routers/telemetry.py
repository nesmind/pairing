"""GET /api/telemetry/{ollama,matricxon}/summary: per-engine call telemetry for the Stats "Telemetry" view
(telemetry_service.py). Admin-only, like stats.py.

The two literal routes are the codebase's convention (one named endpoint per engine) and keep existing
paths working. The generic `/{engine}/summary` route is additive, for a third engine (stats.html/telemetry.js);
telemetry_service.get_engine_telemetry_summary already takes `engine`, so it is routing only.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import EngineName, OllamaTelemetrySummary
from app.services import telemetry_service
from app.services.auth_service import require_admin

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/ollama/summary", response_model=OllamaTelemetrySummary)
async def get_ollama_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await telemetry_service.get_engine_telemetry_summary(db, "ollama")


@router.get("/matricxon/summary", response_model=OllamaTelemetrySummary)
async def get_matricxon_summary(db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)):
    return await telemetry_service.get_engine_telemetry_summary(db, "matricxon")


@router.get("/{engine}/summary", response_model=OllamaTelemetrySummary)
async def get_engine_summary(
    engine: EngineName, db: AsyncSession = Depends(get_db), _user: User = Depends(require_admin)
):
    return await telemetry_service.get_engine_telemetry_summary(db, engine)
