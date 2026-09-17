"""
Endpoints backing the Settings page: which Ollama models are installed,
whether RAG is usable, and the per-user default generation params/model
that new conversations start from. See
app/services/model_catalog_service.py and app/services/settings_service.py
for the actual business logic, app/routers/theme.py for the app-wide UI
theme, app/routers/model_catalog_admin.py for the admin-managed
"browse more models" catalog (both separate files — see each one's own
docstring for why), and app/routers/users.py for account management (a
separate file since it's a big enough topic on its own).

Per-conversation params (what you're actually editing 90% of the time)
are read/written through PATCH /api/conversations/{id} in
app/routers/conversations.py instead — this file is only the "global
defaults" and "environment info" half of the page.
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
from app.services import extended_model_catalog_service, model_catalog_service, settings_service
from app.services.auth_service import get_current_user, hash_password, require_admin, verify_password
from app.services.ollama_admin import delete_model, pull_model_stream
from app.services.ollama_client import OllamaError, has_embedding_model

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/model-catalog", response_model=ModelCatalogResponse)
async def get_model_catalog(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    try:
        return await model_catalog_service.build_model_catalog(db, user)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/installed-models", response_model=InstalledModelsResponse)
async def get_installed_models(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Installed chat-capable model tags, for the chat page's "switch
    model" picker on an existing conversation — a plain name list, none
    of get_model_catalog's hardware/download info, since there's nothing
    to install here."""
    try:
        models = await model_catalog_service.get_installed_models_for_user(db, user)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InstalledModelsResponse(models=models)


@router.post("/hide-model", response_model=HideModelResponse)
async def hide_model(
    body: HideModelRequest,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Hides or unhides one model tag from regular users' picker.
    Admin-only, and purely a visibility control — it doesn't uninstall
    anything and doesn't stop an existing conversation from using a
    hidden model, it just declutters the list regular users choose
    from."""
    tags = await model_catalog_service.get_hidden_tags(db)
    if body.hidden:
        tags.add(body.tag)
    else:
        tags.discard(body.tag)
    await model_catalog_service.set_hidden_tags(db, tags)
    return HideModelResponse(tag=body.tag, hidden=body.hidden)


@router.post("/pull-model")
async def pull_model(body: PullModelRequest, db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Downloads a model into Ollama, streaming progress over SSE (same wire format as the chat stream in
    app/routers/chat.py). Admin-only: pulling is a system-wide, potentially huge (many-GB) download that affects
    every user, not a per-conversation preference. Refuses anything not in either catalog (the default one in
    app/model_catalog.py, or the admin-managed extended one in
    app.services.extended_model_catalog_service — checked in that order) or not hardware-compatible, even if a
    client bypasses the UI's own checks."""
    entry = (
        model_catalog.find_entry(body.tag)
        or await extended_model_catalog_service.find_entry(db, body.tag)
        or model_catalog.find_embedding_entry(body.tag)
    )
    if entry is None or not entry["locally_runnable"]:
        detail = entry["unavailable_reason"] if entry else "Unknown model tag."
        raise HTTPException(status_code=400, detail=detail)

    capacity_gb = hardware.available_capacity_gb()
    if capacity_gb < entry["min_ram_gb"]:
        raise HTTPException(
            status_code=400,
            detail=(
                f"This machine has ~{capacity_gb:.1f}GB of RAM/VRAM available; "
                f"{body.tag} needs at least {entry['min_ram_gb']}GB."
            ),
        )

    async def event_stream():
        try:
            async for progress in pull_model_stream(body.tag):
                yield f"data: {json.dumps(progress)}\n\n"
        except OllamaError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"
            return
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/delete-model", response_model=DeleteModelResponse)
async def delete_model_endpoint(body: PullModelRequest, _admin: User = Depends(require_admin)):
    """Removes a pulled model from Ollama, freeing its disk space. Admin-only for the same reason pulling is: it's
    a system-wide action affecting every user, not a per-conversation preference. Deleting a model someone's
    conversation is still set to isn't blocked here — that conversation just gets a normal "model not found" chat
    error (handled gracefully in app/services/chat_service.py) until its user picks a different one, same as if it
    had never been installed."""
    try:
        await delete_model(body.tag)
    except OllamaError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return DeleteModelResponse(deleted=body.tag)


@router.get("/rag-availability", response_model=RagAvailability)
async def rag_availability(_user: User = Depends(get_current_user)):
    """Whether the embedding model the knowledge base needs is actually installed, so the Settings page can
    explain itself instead of the knowledge base just silently doing nothing."""
    return RagAvailability(available=await has_embedding_model())


@router.get("/embedding-model-catalog", response_model=EmbeddingModelCatalogResponse)
async def get_embedding_model_catalog(_user: User = Depends(get_current_user)):
    """The embedding models app/model_catalog.py's EMBEDDING_CATALOG knows about, each with install status —
    the embedding-model analogue of GET /api/settings/model-catalog above, which never includes these (see
    EMBEDDING_CATALOG's own docstring for why they're kept separate)."""
    return await model_catalog_service.build_embedding_model_catalog()


@router.get("/rag-limits", response_model=RagLimits)
async def get_rag_limits_endpoint(db: AsyncSession = Depends(get_db), _user: User = Depends(get_current_user)):
    """Current upload caps — open to any logged-in user (not admin-only) since the Knowledge tab's upload box
    needs these to show/enforce them client-side; only *changing* them is admin-only, below."""
    return await settings_service.get_rag_limits(db)


@router.put("/rag-limits", response_model=RagLimits)
async def set_rag_limits_endpoint(
    body: RagLimits,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Sets the admin-configured RAG upload caps from Settings > System. Takes effect immediately for the next
    upload — existing documents already over a newly-lowered limit are left alone rather than retroactively
    deleted."""
    await settings_service.set_rag_limits(db, body)
    return body


@router.get("/defaults", response_model=GenerationParams)
async def get_defaults(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The current user's own defaults new conversations are created with."""
    return await settings_service.get_default_params(db, user)


@router.put("/defaults", response_model=GenerationParams)
async def set_defaults(
    params: GenerationParams,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the current user's own defaults — never affects any other user's defaults, including admin's."""
    value = params.model_dump()
    await settings_service.set_default_params(db, user.id, value)
    return value


@router.get("/default-model", response_model=DefaultModel)
async def get_default_model_endpoint(db: AsyncSession = Depends(get_db), user: User = Depends(get_current_user)):
    """The model a brand-new conversation of the current user's starts with — shown/settable from Settings'
    "Defaults for new chats" view."""
    return DefaultModel(model=await settings_service.get_default_model(db, user))


@router.put("/default-model", response_model=DefaultModel)
async def set_default_model(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Saves the current user's own default model — never affects any other user's default, including admin's."""
    await settings_service.set_default_model(db, user.id, body.model)
    return body


@router.post("/change-password", response_model=OkResponse)
async def change_password(
    body: ChangePasswordRequest,
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Lets any logged-in user change their own password — self-service, unlike PATCH /users/{username}
    (admin-only, no current-password check, can touch anyone's account). Requires the current password as proof
    of ownership, same as changing a password on virtually any other site, rather than letting a bare active
    session silently take over the account's credentials."""
    if not verify_password(body.current_password, user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    user.password_hash = hash_password(body.new_password)
    await db.commit()
    return OkResponse()


@router.get("/system-info", response_model=SystemInfo)
def system_info(_admin: User = Depends(require_admin)):
    """Admin-only snapshot for the Settings page's System tab: this
    machine's hardware and the app's own version. (Which Ollama/ComfyUI
    host the app is talking to now lives on the External servers tab
    instead — see app/routers/ollama_admin.py/comfyui_admin.py — since
    that's admin-configurable now, not a fixed environment fact. The
    user list is its own endpoint — see app/routers/users.py.)"""
    return SystemInfo(
        hardware=hardware.hardware_summary(),
        app_version=APP_VERSION,
        local_install_supported=hardware.local_install_supported(),
    )
