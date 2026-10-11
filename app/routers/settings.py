"""Endpoints for the Settings page: installed models, RAG availability, and per-user default generation
params/model for new conversations. Logic: services/model_catalog_service.py, settings_service.py.
Related: routers/theme.py (UI theme), model_catalog_admin.py ("browse more models"), users.py (accounts).

Per-conversation params go through PATCH /api/conversations/{id} (routers/conversations.py); this file
holds the global defaults and environment info.
"""

import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app import hardware, model_catalog
from app.config import APP_VERSION
from app.database import get_db
from app.models import User
from app.schemas import (
    ChangePasswordRequest,
    DefaultModel,
    DeleteModelResponse,
    EmbeddingModelCatalogResponse,
    GenerationParams,
    HideModelRequest,
    HideModelResponse,
    InstalledModelsResponse,
    ModelCatalogResponse,
    OkResponse,
    PullModelRequest,
    RagAvailability,
    RagLimits,
    SystemInfo,
)
from app.services import settings_service
from app.services.auth_service import get_current_user, hash_password, require_admin, verify_password
from app.services.duplicate_install_detector import DuplicateInstallDetector
from app.services.embedding_model_catalog_service import EmbeddingModelCatalogService
from app.services.engines.registry import registry
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.inference_client import InferenceError, delete_model, pull_model_stream
from app.services.model_catalog_service import ChatModelCatalogBuilder, HiddenModelTags, get_default_embedding_model

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/model-catalog", response_model=ModelCatalogResponse)
async def get_model_catalog(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return await ChatModelCatalogBuilder(db, user).build()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/installed-models", response_model=InstalledModelsResponse)
async def get_installed_models(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Installed chat-capable model tags for the chat page's model switcher: a plain name list."""
    try:
        models = await ChatModelCatalogBuilder(db, user).installed_tags()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InstalledModelsResponse(models=models)


@router.post("/hide-model", response_model=HideModelResponse)
async def hide_model(
    body: HideModelRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Disables (hidden=True) or enables a model tag for regular users: left out of their picker and refused in
    their chats (admins unaffected). Admin-only; uninstalls nothing.
    """
    hidden_tags = HiddenModelTags(db)
    tags = await hidden_tags.get()
    if body.hidden:
        tags.add(body.tag)
    else:
        tags.discard(body.tag)
    await hidden_tags.set(tags)
    return HideModelResponse(tag=body.tag, hidden=body.hidden)


@router.post("/pull-model")
async def pull_model(body: PullModelRequest, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Downloads a model into the engine, streaming SSE like the chat stream (routers/chat.py). Admin-only: a
    system-wide, possibly many-GB download. Refuses tags missing from both catalogs (app/model_catalog.py,
    then the extended one) or hardware-incompatible, even if the UI is bypassed.
    """
    entry = (
        model_catalog.CATALOG.find(body.tag)
        or await ExtendedModelCatalog(db).find(body.tag)
        or model_catalog.CATALOG.find_embedding(body.tag)
    )
    if entry is None or not entry.locally_runnable:
        detail = entry.unavailable_reason if entry else "Unknown model tag."
        raise HTTPException(status_code=400, detail=detail)

    if not body.confirm_duplicate:
        duplicate_tag = await DuplicateInstallDetector.find(body.tag, entry.architecture, entry.download_gb)
        if duplicate_tag:
            raise HTTPException(
                status_code=409,
                detail={
                    "duplicate_of": duplicate_tag,
                    "message": (
                        f'This looks like the same model as the already-installed "{duplicate_tag}" — pull anyway?'
                    ),
                },
            )

    capacity_gb = hardware.available_capacity_gb()
    if capacity_gb < entry.min_ram_gb:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This machine has ~{capacity_gb:.1f}GB of RAM/VRAM available; "
                f"{body.tag} needs at least {entry.min_ram_gb}GB."
            ),
        )

    async def event_stream():
        try:
            async for progress in pull_model_stream(body.tag):
                yield f"data: {json.dumps(progress)}\n\n"
        except InferenceError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/delete-model", response_model=DeleteModelResponse)
async def delete_model_endpoint(
    body: PullModelRequest, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)
):
    """Removes a pulled model, freeing disk. Admin-only (system-wide). Deleting a model a conversation uses is
    allowed: its chat gets a normal "model not found" error (chat_service.py) until another is picked.
    """
    try:
        await delete_model(body.tag)
    except InferenceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    await ExtendedModelCatalog(db).remove(body.tag)  # an uninstalled model must not come back as a suggestion
    return DeleteModelResponse(deleted=body.tag)


@router.get("/rag-availability", response_model=RagAvailability)
async def rag_availability(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Whether RAG is usable, checked in order: admin-enabled, the active engine can serve embeddings, the
    embedding model is installed; so the Settings page can explain itself.

    The admin switch (RagLimits.enabled) is a system-wide override, checked first as the most deliberate
    answer. Engine capability comes next: for a connector like RunPod (capabilities.embeddings False) an
    installed model is irrelevant, and that is the more accurate reason.
    """
    limits = await settings_service.get_rag_limits(db)
    if not limits.enabled:
        return RagAvailability(available=False, reason="RAG is not enabled on this system, ask admin")
    engine = registry.active()
    if not engine.capabilities.embeddings:
        return RagAvailability(
            available=False,
            reason=f"{engine.display_name} doesn't support embeddings — the knowledge base needs a different "
            "active engine (see External servers/Connectors).",
        )
    if (await get_default_embedding_model(db)) is None:
        return RagAvailability(available=False, reason="No embedding model found yet — ask admin.")
    return RagAvailability(available=True)


@router.get("/embedding-model-catalog", response_model=EmbeddingModelCatalogResponse)
async def get_embedding_model_catalog(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Embedding models from EMBEDDING_CATALOG with install status (the embedding analogue of GET
    /api/settings/model-catalog)."""
    return await EmbeddingModelCatalogService(db).build()


@router.get("/rag-limits", response_model=RagLimits)
async def get_rag_limits_endpoint(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Current upload caps; open to all users since the Knowledge tab enforces them client-side. Only changing is
    admin-only."""
    return await settings_service.get_rag_limits(db)


@router.put("/rag-limits", response_model=RagLimits)
async def set_rag_limits_endpoint(
    body: RagLimits,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Sets the RAG upload caps (Settings > System). Applies to the next upload; existing documents over a
    lowered limit stay.
    """
    await settings_service.set_rag_limits(db, body)
    return body


@router.get("/defaults", response_model=GenerationParams)
async def get_defaults(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The user's own defaults for new conversations."""
    return await settings_service.get_default_params(db, user)


@router.put("/defaults", response_model=GenerationParams)
async def set_defaults(
    params: GenerationParams,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the user's own defaults; never affects others, admin included."""
    value = params.model_dump()
    await settings_service.set_default_params(db, user.id, value)
    return value


@router.get("/default-model", response_model=DefaultModel)
async def get_default_model_endpoint(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The model new conversations start with, for Settings' "Defaults for new chats"."""
    return DefaultModel(model=await settings_service.get_default_model(db, user))


@router.put("/default-model", response_model=DefaultModel)
async def set_default_model(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the user's own default model; never affects others, admin included."""
    await settings_service.set_default_model(db, user.id, body.model)
    return body


@router.post("/change-password", response_model=OkResponse)
async def change_password(
    body: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Self-service password change, unlike admin-only PATCH /users/{username}. Requires the current password."""
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    user.password_hash = hash_password(body.new_password)
    await db.commit()
    return OkResponse()


@router.get("/system-info", response_model=SystemInfo)
def system_info(_admin: User = Depends(require_admin)):
    """Admin snapshot for the System tab: hardware and app version. (Engine/ComfyUI hosts are on the External
    servers tab, users in routers/users.py.)
    """
    return SystemInfo(
        hardware=hardware.hardware_summary(),
        app_version=APP_VERSION,
        local_install_supported=hardware.local_install_supported(),
    )
