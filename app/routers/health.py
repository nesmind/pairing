"""GET /health: reverse-proxy readiness. Reports whether this instance can serve a chat now: at least one host
of the active engine reachable (engines.registry, engine_service) and the database reachable.

Unauthenticated and cheap (short-timeout GET per host, `SELECT 1`) so proxies can poll often.
200 when healthy, 503 otherwise (what proxies act on); the body is for humans.
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
