"""GET/PUT for Settings > External servers' "Active engine" picker: which engine app.services.inference_client
dispatches chat, embedding and model management to (engine_service). A separate router since it belongs to neither
engine.
"""

import dataclasses
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import ActiveEngineConfig, EngineCapabilitiesOut, EngineOption, EngineOptionsResponse, OkResponse
from app.services import engine_service, server_pool_broadcast
from app.services.auth_service import get_current_user, require_admin
from app.services.engines.registry import registry
from app.services.extended_model_catalog_service import ExtendedModelCatalog

logger = logging.getLogger("llama_chat")

router = APIRouter(prefix="/api/settings/engine", tags=["engine"])


@router.get("", response_model=ActiveEngineConfig)
async def get_active_engine(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    return ActiveEngineConfig(active_engine=await engine_service.get_active_engine(db))


@router.get("/options", response_model=EngineOptionsResponse)
async def get_engine_options(_user: User = Depends(get_current_user)):
    """Every registered engine (engines.registry), for pickers and capability checks without hardcoded names.
    Open to any logged-in user: static metadata with no secrets, needed by regular users' views (e.g. the RAG embedding
    picker).
    """
    return EngineOptionsResponse(
        engines=[
            EngineOption(
                name=engine.name,
                display_name=engine.display_name,
                capabilities=EngineCapabilitiesOut(**dataclasses.asdict(engine.capabilities)),
                ready=await engine.is_ready(),
            )
            for engine in registry.all()
        ]
    )


@router.put("", response_model=ActiveEngineConfig)
async def set_active_engine(
    body: ActiveEngineConfig, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    # A Connector-backed engine (e.g. "runpod") may be registered but not ready (InferenceEngine.is_ready), so
    # it cannot be activated until configured and enabled on the Connectors page. Ollama/Matricxon are always ready.
    engine = registry.get(body.active_engine)
    if not await engine.is_ready():
        raise HTTPException(
            status_code=400,
            detail=f"{engine.display_name} isn't configured and enabled yet — set it up on the Connectors page first.",
        )
    # Each engine keeps its default model under its own key (settings_service.default_model_key), so switching
    # need not touch it (the old shared key had to be wiped on every switch).
    if await engine_service.get_active_engine(db) != body.active_engine:
        await ExtendedModelCatalog(db).clear()
    await engine_service.set_active_engine(db, body.active_engine)
    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/engine/internal-refresh")
    if failures:
        logger.warning("Some instances did not pick up the new active engine: %s", "; ".join(failures))
    return body


@router.post("/internal-refresh", response_model=OkResponse)
async def internal_refresh(request: Request, db: AsyncSession = Depends(get_db)):
    """Called only by another local instance's server_pool_broadcast.broadcast_refresh after an engine switch;
    see ollama_admin.internal_refresh for the loopback-only guard.
    """
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="Internal endpoint — not reachable externally.")
    await engine_service.load_cache_from_db(db)
    return OkResponse(ok=True)
