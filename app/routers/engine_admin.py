"""
GET/PUT for Settings > External servers' "Active engine" picker — see app.services.engine_service for what this
actually controls (which engine app.services.inference_client dispatches chat/embedding/model-management calls
to). Deliberately its own tiny router rather than folded into ollama_admin.py or matricxon_admin.py: this choice
belongs to neither engine specifically.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ActiveEngineConfig, OkResponse
from app.services import engine_service, engine_switch_service, server_pool_broadcast
from app.services.auth_service import require_admin

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/engine", tags=["engine"])


@router.get("", response_model=ActiveEngineConfig)
async def get_active_engine(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ActiveEngineConfig(active_engine=await engine_service.get_active_engine(db))


@router.put("", response_model=ActiveEngineConfig)
async def set_active_engine(
    body: ActiveEngineConfig, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    previous_engine = await engine_service.get_active_engine(db)
    await engine_service.set_active_engine(db, body.active_engine)
    if body.active_engine != previous_engine:
        # Only when the engine actually changed — a no-op save (re-picking the same engine) must never touch
        # anyone's already-valid default model. See engine_switch_service's own docstring for why this can't
        # live inside engine_service.set_active_engine itself (a circular import).
        await engine_switch_service.clear_stale_default_models(db)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/engine/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new active engine: %s", "; ".join(failures))
    return body


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's own server_pool_broadcast.broadcast_refresh right after an
    admin's engine switch — see app/routers/ollama_admin.py's internal_refresh for the identical loopback-only
    guard and reasoning."""
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await engine_service.load_cache_from_db(db)
    return OkResponse(ok=True)
