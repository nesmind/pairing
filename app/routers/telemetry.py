"""GET /api/telemetry/{ollama,matricxon}/summary — per-engine call telemetry for the Stats page's "Telemetry"
view (see app/services/telemetry_service.py for the aggregation queries). Admin-only, same access-control
decision as app/routers/stats.py's own summary endpoint (see that file's own docstring).

The two literal routes are this codebase's existing convention (a real, named endpoint per engine rather than a
validated-string param — same as ollama_admin.py/matricxon_admin.py being separate routers instead of one
parameterized by engine name) and stay, so nothing already depending on their exact paths breaks. The generic
`/{engine}/summary` route below is additive: the frontend (see stats.html/telemetry.js) now calls it directly
once a third engine's name isn't one of these two literals, without this router needing a new route added for
it — `telemetry_service.get_engine_telemetry_summary` already takes `engine` as a plain parameter, so this is
routing-only, no new aggregation logic."""

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
