"""Admin settings > System: which model reranks Knowledge base results (document_rerank.py)."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas.rerank import RerankModelCatalogResponse, RerankModelSetting, RerankModelUpdate
from app.services.auth_service import require_admin
from app.services.default_model_settings import DefaultModelSettings
from app.services.engines.registry import registry
from app.services.inference_client import InferenceError
from app.services.rerank_model_catalog_service import RerankModelCatalogService

router = APIRouter(prefix="/api/settings", tags=["rerank"])


async def _current(settings: DefaultModelSettings) -> RerankModelSetting:
    engine_supports = registry.active().capabilities.rerank
    installed = await settings.installed_rerankers() if engine_supports else []
    return RerankModelSetting(
        model=await settings.rerank() if engine_supports else None,
        enabled=await settings.rerank_enabled(),
        installed=installed,
        engine_supports=engine_supports,
    )


@router.get("/rerank-model", response_model=RerankModelSetting)
async def get_rerank_model(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> RerankModelSetting:
    try:
        return await _current(DefaultModelSettings(db))
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.put("/rerank-model", response_model=RerankModelSetting)
async def set_rerank_model(
    body: RerankModelUpdate, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> RerankModelSetting:
    settings = DefaultModelSettings(db)
    try:
        if body.model is not None and body.model not in await settings.installed_rerankers():
            raise HTTPException(status_code=400, detail=f"{body.model} is not an installed reranker.")
        await settings.set_rerank(body.model)
        return await _current(settings)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/rerank-model-catalog", response_model=RerankModelCatalogResponse)
async def get_rerank_model_catalog(
    db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
) -> RerankModelCatalogResponse:
    """Settings > Models > Reranker models."""
    try:
        return await RerankModelCatalogService(db).build()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
