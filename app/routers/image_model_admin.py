"""Settings > Model's image-model actions (see app/services/image_model_service.py): list the local
stable-diffusion.cpp engine's models (feeds its Model dropdown), and pull/delete one by its catalog tag — the
image-model twins of /api/settings/pull-model and /delete-model, which only know the chat engines."""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import DiffusionModelCatalogResponse, ImageModelList, ImageModelRequest, OkResponse
from app.services import http_proxy_service
from app.services.auth_service import get_current_user, require_admin
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.image_model_service import ImageModelStore

router = APIRouter(prefix="/api/settings/image-models", tags=["image-models"])


@router.get("/catalog", response_model=DiffusionModelCatalogResponse)
async def get_diffusion_model_catalog(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The Diffusion models section — visible to every user (install state isn't secret); only pulling,
    uninstalling and removing are admin-only."""
    store = await ImageModelStore.open(db)
    return DiffusionModelCatalogResponse(
        entries=await store.catalog_entries(is_admin=user.role == "admin"), local=await store.is_local()
    )


@router.get("", response_model=ImageModelList)
async def list_image_models(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    store = await ImageModelStore.open(db)
    return ImageModelList(models=store.list_files(), local=await store.is_local(), folder=str(store.root))


@router.post("/pull")
async def pull_image_model(
    body: ImageModelRequest, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    """Downloads an image model from Hugging Face into the local engine's models folder, streaming progress over
    SSE in the same wire format as /api/settings/pull-model."""
    store = await ImageModelStore.open(db)
    try:
        repo_id, filename, dest = await store.prepare_pull(body.tag)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    proxy_url = (await http_proxy_service.get_http_proxy_config(db)).proxy_url()

    async def event_stream():
        async for event in store.download(repo_id, filename, dest, proxy_url):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/delete", response_model=OkResponse)
async def delete_image_model(
    body: ImageModelRequest, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    store = await ImageModelStore.open(db)
    try:
        await store.delete(body.tag)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    await ExtendedModelCatalog(db).remove(body.tag)  # an uninstalled model must not come back as a suggestion
    return OkResponse(ok=True)
