"""
GET /health — a reverse-proxy-facing readiness check, not part of the
app's own UI. Reports whether *this* instance can actually serve a real
chat request right now: at least one of the currently *active* engine's
effective host(s) reachable (see app.services.engines.registry, resolved
via app.services.engine_service — local or remote per Settings >
External servers), and the database reachable — not just "is the
process alive," which a plain TCP/HTTP check to any other route would
already tell you.

Deliberately unauthenticated (infrastructure polls this, not logged-in
users) and cheap (a short-timeout GET per host, a plain `SELECT 1`), so
it's safe for a proxy to hit every few seconds. Returns 200 when
healthy, 503 otherwise — the status code a reverse proxy actually looks
at to decide whether to keep routing here; the body is for a human
debugging why.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import HealthStatus
from app.services.engines.registry import registry

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    engine = registry.active()
    active_engine = engine.name
    engine_hosts = await engine.check_health()
    engine_ok = any(engine_hosts.values())

    try:
        await db.execute(text("SELECT 1"))
        database_ok = True
    except SQLAlchemyError:
        database_ok = False

    healthy = engine_ok and database_ok
    body = HealthStatus(
        status="ok" if healthy else "unhealthy",
        database=database_ok,
        active_engine=active_engine,
        engine_hosts=engine_hosts,
    )
    return JSONResponse(content=body.model_dump(), status_code=200 if healthy else 503)
