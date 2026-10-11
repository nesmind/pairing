"""Account management for Settings > Users (admin-only), plus small system-wide admin settings: default model
for new accounts, reply settings from chat_settings_service.py (title mode, channel delivery, timeouts),
and Telemetry/System retention (retention_settings_service.py). Logic: user_service.py, model_catalog_service.py.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import User
from app.schemas import (
    ChannelDeliveryMode,
    DefaultModel,
    InstalledModelsResponse,
    OkResponse,
    ReplyTimeout,
    RetentionSettings,
    TitleMode,
    UserCreate,
    UserOut,
    UserUpdate,
)
from app.services import chat_settings_service, model_catalog_service, retention_settings_service, user_service
from app.services.auth_service import get_current_user, require_admin
from app.services.inference_client import InferenceError

router = APIRouter(prefix="/api/settings", tags=["users"])


@router.get("/default-model-for-new-users", response_model=DefaultModel)
async def get_default_model_for_new_users_endpoint(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """The model new accounts start with. Admin-only."""
    try:
        installed = await model_catalog_service.installed_chat_models()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    model = await model_catalog_service.get_default_model_for_new_users(db, installed)
    return DefaultModel(model=model or "")


@router.put("/default-model-for-new-users", response_model=DefaultModel)
async def set_default_model_for_new_users_endpoint(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await model_catalog_service.set_default_model_for_new_users(db, body.model)
    return body


@router.get("/installed-vision-models", response_model=InstalledModelsResponse)
async def get_installed_vision_models_endpoint(_admin: User = Depends(require_admin)):
    """Installed vision-capable model tags for the default-vision-model picker. Admin-only."""
    try:
        models = await model_catalog_service.installed_vision_models()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InstalledModelsResponse(models=models)


@router.get("/default-vision-model", response_model=DefaultModel)
async def get_default_vision_model_endpoint(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """The model that answers any message with an image attachment, for that reply only
    (chat_service.build_reply_stream); never the conversation's own model.
    """
    return DefaultModel(model=await model_catalog_service.get_default_vision_model(db) or "")


@router.put("/default-vision-model", response_model=DefaultModel)
async def set_default_vision_model_endpoint(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await model_catalog_service.set_default_vision_model(db, body.model)
    return body


@router.get("/installed-embedding-models", response_model=InstalledModelsResponse)
async def get_installed_embedding_models_endpoint(_admin: User = Depends(require_admin)):
    """Installed embedding-capable model tags for the default-embedding-model picker. Admin-only."""
    try:
        models = await model_catalog_service.installed_embedding_models()
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InstalledModelsResponse(models=models)


@router.get("/default-embedding-model", response_model=DefaultModel)
async def get_default_embedding_model_endpoint(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """The model inference_client.embed uses for the Knowledge base
    (model_catalog_service.get_default_embedding_model)."""
    return DefaultModel(model=await model_catalog_service.get_default_embedding_model(db) or "")


@router.put("/default-embedding-model", response_model=DefaultModel)
async def set_default_embedding_model_endpoint(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await model_catalog_service.set_default_embedding_model(db, body.model)
    return body


@router.get("/title-mode", response_model=TitleMode)
async def get_title_mode_endpoint(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Whether a new chat's title comes from quick text processing ("simple") or the model ("smart")
    (chat_settings_service.get_title_mode).
    """
    return TitleMode(mode=await chat_settings_service.get_title_mode(db))


@router.put("/title-mode", response_model=TitleMode)
async def set_title_mode_endpoint(
    body: TitleMode,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await chat_settings_service.set_title_mode(db, body.mode)
    return body


@router.get("/channel-delivery-mode", response_model=ChannelDeliveryMode)
async def get_channel_delivery_mode_endpoint(
    db: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Open to all users: chat.js needs it to choose between polling and a live /subscribe connection for a
    channel (chat_settings_service.get_channel_delivery_mode).
    """
    return ChannelDeliveryMode(mode=await chat_settings_service.get_channel_delivery_mode(db))


@router.put("/channel-delivery-mode", response_model=ChannelDeliveryMode)
async def set_channel_delivery_mode_endpoint(
    body: ChannelDeliveryMode,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await chat_settings_service.set_channel_delivery_mode(db, body.mode)
    return body


@router.get("/reply-timeout", response_model=ReplyTimeout)
async def get_reply_timeout_endpoint(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """How long to wait for a model to finish a reply (chat_settings_service.get_reply_timeout_seconds).
    Admin-only: a backend cap the frontend never branches on.
    """
    return ReplyTimeout(timeout_seconds=await chat_settings_service.get_reply_timeout_seconds(db))


@router.put("/reply-timeout", response_model=ReplyTimeout)
async def set_reply_timeout_endpoint(
    body: ReplyTimeout,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await chat_settings_service.set_reply_timeout_seconds(db, body.timeout_seconds)
    return body


@router.get("/vision-reply-timeout", response_model=ReplyTimeout)
async def get_vision_reply_timeout_endpoint(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Like reply-timeout, for replies with an image; it needs a more generous limit
    (chat_settings_service.get_vision_reply_timeout_seconds).
    """
    return ReplyTimeout(timeout_seconds=await chat_settings_service.get_vision_reply_timeout_seconds(db))


@router.put("/vision-reply-timeout", response_model=ReplyTimeout)
async def set_vision_reply_timeout_endpoint(
    body: ReplyTimeout,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await chat_settings_service.set_vision_reply_timeout_seconds(db, body.timeout_seconds)
    return body


@router.get("/retention", response_model=RetentionSettings)
async def get_retention_endpoint(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """How long Telemetry/System history is kept (retention_poller). Admin-only, like reply-timeout."""
    return await retention_settings_service.get_retention_settings(db)


@router.put("/retention", response_model=RetentionSettings)
async def set_retention_endpoint(
    body: RetentionSettings,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await retention_settings_service.set_retention_settings(db, body)
    return body


@router.get("/users", response_model=list[UserOut])
async def list_users(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Every account, for the Users section. Admin-only."""
    return (await db.execute(select(User).order_by(User.created_at))).scalars().all()


@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Creates an account. Admin-only: no self-registration, by design (private, invite-only)."""
    try:
        return await user_service.create_user(db, body)
    except InferenceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/users/{username}", response_model=UserOut)
async def update_user(
    username: str,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Edits an account: password, admin/user role, disabled or enabled."""
    return await user_service.update_user(db, username, body)


@router.delete("/users/{username}", response_model=OkResponse)
async def delete_user(
    username: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Permanently deletes an account and everything it owns; disabling (PATCH) is the reversible alternative."""
    await user_service.delete_user(db, username)
    return OkResponse()
