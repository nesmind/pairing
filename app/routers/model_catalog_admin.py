"""GET/POST/DELETE for the "browse more models" catalog (Settings > Model;
app/services/extended_model_catalog_service.py) and the Hugging Face search/file lookups behind
"Add a model" (huggingface_client.py). Separate from settings.py for the file-size cap.
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
from app.services import http_proxy_service
from app.services.auth_service import get_current_user, require_admin
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError

router = APIRouter(prefix="/api/settings/model-catalog/extended", tags=["settings"])


@router.get("", response_model=ExtendedModelCatalogResponse)
async def get_extended_catalog(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """Visible to all users (admin-hidden tags are left out for regular users); only admins add or remove."""
    entries = await ExtendedModelCatalog(db).build(user)
    return ExtendedModelCatalogResponse(entries=entries)


@router.post("/search-hf", response_model=SearchHfModelsResponse)
async def search_hf_models(
    body: SearchHfModelsRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Searches Hugging Face for GGUF repos matching `body.query` (first step of "Add a model").
    Admin-only; uses the admin-configured outbound proxy.
    """
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        results = await HuggingFaceCatalogSearch.search(body.query, proxy_url=proxy_config.proxy_url())
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchHfModelsResponse(results=results)


@router.post("/hf-files", response_model=HfRepoFilesResponse)
async def get_hf_repo_files(
    body: HfRepoFilesRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Lists a repo's single-file GGUF variants with sizes (the file-picker step)."""
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    try:
        repo = await HuggingFaceCatalogSearch.repo_files(body.repo_id, proxy_url=proxy_config.proxy_url())
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return HfRepoFilesResponse(**repo)


@router.post("", response_model=ExtendedModelCatalogResponse)
async def add_extended_model(
    body: AddExtendedModelRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Verifies `body.repo_id`/`body.filename` exist on Hugging Face before adding. For an already-installed
    tag a lookup failure is tolerated (ExtendedModelCatalog.add). With no internet and no proxy, a
    not-yet-installed add fails with a clear message rather than hanging.
    """
    proxy_config = await http_proxy_service.get_http_proxy_config(db)
    catalog = ExtendedModelCatalog(db)
    try:
        added = await catalog.add(body.repo_id, body.filename, proxy_url=proxy_config.proxy_url())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HuggingFaceLookupError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    entries = await catalog.build(admin)
    # added["note"] is ALREADY_INSTALLED_NOTE only on that branch (see ExtendedModelCatalogResponse.already_installed).
    already_installed = added["note"] == ExtendedModelCatalog.ALREADY_INSTALLED_NOTE
    return ExtendedModelCatalogResponse(entries=entries, already_installed=already_installed)


@router.delete("", response_model=ExtendedModelCatalogResponse)
async def remove_extended_model(
    body: RemoveExtendedModelRequest,
    db: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
):
    """Removes a tag from the extended catalog only; never uninstalls it (curation, not deletion, like settings.py's
    hide-model)."""
    catalog = ExtendedModelCatalog(db)
    await catalog.remove(body.tag)
    entries = await catalog.build(admin)
    return ExtendedModelCatalogResponse(entries=entries)
