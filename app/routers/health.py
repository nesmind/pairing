"""
GET /health — a reverse-proxy-facing readiness check, not part of the
app's own UI. Reports whether *this* instance can actually serve a real
chat request right now: at least one of its currently effective Ollama
host(s) reachable (see app.services.ollama_pool.get_effective_hosts —
local or remote, per Settings > External servers), and the database
reachable — not just "is the process alive," which a plain TCP/HTTP
check to any other route would already tell you.

Deliberately unauthenticated (infrastructure polls this, not logged-in
users) and cheap (a short-timeout GET per Ollama host, a plain
`SELECT 1`), so it's safe for a proxy to hit every few seconds. Returns
200 when healthy, 503 otherwise — the status code a reverse proxy
actually looks at to decide whether to keep routing here; the body is
for a human debugging why.
"""

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import HealthStatus
from app.services import ollama_pool

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthStatus)
async def health(db: AsyncSession = Depends(get_db)) -> JSONResponse:
    ollama_hosts = await ollama_pool.check_hosts()
    ollama_ok = any(ollama_hosts.values())

    try:
        await db.execute(text("SELECT 1"))
        database_ok = True
    except SQLAlchemyError:
        database_ok = False

    healthy = ollama_ok and database_ok
    body = HealthStatus(status="ok" if healthy else "unhealthy", database=database_ok, ollama_hosts=ollama_hosts)
    return JSONResponse(content=body.model_dump(), status_code=200 if healthy else 503)
