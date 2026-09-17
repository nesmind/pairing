"""
GET/POST/DELETE for the "browse more models" catalog behind Settings >
Model (see app/services/extended_model_catalog_service.py), plus the
Hugging Face search/file-lookup endpoints that feed its "Add a model"
flow (see app/services/huggingface_client.py) — a separate file from
app/routers/settings.py (already at CLAUDE.md's line cap) for the same
reason app/routers/theme.py is its own file.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import (
    AddExtendedModelRequest,
    ExtendedModelCatalogResponse,
    HfRepoFilesRequest,
    HfRepoFilesResponse,
    RemoveExtendedModelRequest,
    SearchHfModelsRequest,
    SearchHfModelsResponse,
)
from app.services import extended_model_catalog_service, http_proxy_service
from app.services.auth_service import get_current_user, require_admin
from app.services.huggingface_client import HuggingFaceLookupError, get_repo_files, search_models

router = APIRouter(prefix="/api/settings/model-catalog/extended", tags=["settings"])


@router.get("", response_model=ExtendedModelCatalogResponse)
async def get_extended_catalog(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Every user can browse this list (same visibility rule as the default catalog — an admin-hidden tag is
    left out for a regular user, not just marked hidden), but only an admin can add to or remove from it."""
    entries = await extended_model_catalog_service.build_extended_catalog(db, user)
    return ExtendedModelCatalogResponse(entries=entries)


@router.post("/search-hf", response_model=SearchHfModelsResponse)
async def search_hf_models(
    body: SearchHfModelsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Searches Hugging Face for GGUF-tagged repos matching `body.query` — the first step of "Add a model."
    Admin-only (same as the rest of this add flow), routed through the same admin-configured outbound proxy as
    the Ollama/ComfyUI installers (Settings > System)."""
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        results = await search_models(body.query, proxy_url=proxy_config.proxy_url())
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchHfModelsResponse(results=results)


@router.post("/hf-files", response_model=HfRepoFilesResponse)
async def get_hf_repo_files(
    body: HfRepoFilesRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Lists every single-file GGUF variant `body.repo_id` offers, with real sizes — the file-picker step shown
    once an admin picks a repo from search_hf_models' results (or already knows the exact repo path and skips
    search entirely)."""
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        repo = await get_repo_files(body.repo_id, proxy_url=proxy_config.proxy_url())
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return HfRepoFilesResponse(**repo)


@router.post("", response_model=ExtendedModelCatalogResponse)
async def add_extended_model(
    body: AddExtendedModelRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Verifies `body.repo_id`/`body.filename` really exist on Hugging Face (skipped if the resulting tag is
    already installed — see add_model's own docstring) before adding, so the catalog never ends up carrying a
    typo'd or made-up entry. Same proxy-routing as search_hf_models/get_hf_repo_files above — on a machine with
    no internet access and no proxy configured, this fails with a clear "check your connection or configure a
    proxy" message rather than hanging or a raw traceback."""
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        await extended_model_catalog_service.add_model(
            db, body.repo_id, body.filename, proxy_url=proxy_config.proxy_url()
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    entries = await extended_model_catalog_service.build_extended_catalog(db, admin)
    return ExtendedModelCatalogResponse(entries=entries)


@router.delete("", response_model=ExtendedModelCatalogResponse)
async def remove_extended_model(
    body: RemoveExtendedModelRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Removes a tag from the extended catalog only — never uninstalls it from Ollama if it happens to be
    installed, same "curation, not deletion" distinction app/routers/settings.py's hide-model endpoint draws."""
    await extended_model_catalog_service.remove_model(db, body.tag)
    entries = await extended_model_catalog_service.build_extended_catalog(db, admin)
    return ExtendedModelCatalogResponse(entries=entries)
