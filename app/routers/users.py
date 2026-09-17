"""
Account management for the Settings > Users tab (admin-only), plus small
admin-configured system-wide settings that don't cleanly belong to any
other file: the default model brand-new accounts are created with, the
reply-generation settings from app/services/chat_settings_service.py
(chat-title mode, channel delivery mode, reply timeout), and the
Telemetry/System retention windows from
app/services/retention_settings_service.py. See
app/services/user_service.py and app/services/model_catalog_service.py
for the actual business logic — this file only handles HTTP routing,
status codes, and Depends() injection.
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
from app.services.ollama_client import OllamaError

router = APIRouter(prefix="/api/settings", tags=["users"])


@router.get("/default-model-for-new-users", response_model=DefaultModel)
async def get_default_model_for_new_users_endpoint(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """The model new accounts are created with. Admin-only — regular
    users have no reason to see or change this."""
    try:
        installed = await model_catalog_service.installed_chat_models()
    except OllamaError as exc:
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
    """Installed vision-capable model tags, for the "default vision
    model" picker below — admin-only, unlike GET /installed-models,
    since only this settings section ever needs it."""
    try:
        models = await model_catalog_service.installed_vision_models()
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return InstalledModelsResponse(models=models)


@router.get("/default-vision-model", response_model=DefaultModel)
async def get_default_vision_model_endpoint(
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """The model any message with an image attachment is answered by,
    for that reply only (see app.services.chat_service.build_reply_stream)
    — never the conversation's own model, so switching this setting never
    changes what an existing chat "normally" replies with."""
    return DefaultModel(model=await model_catalog_service.get_default_vision_model(db) or "")


@router.put("/default-vision-model", response_model=DefaultModel)
async def set_default_vision_model_endpoint(
    body: DefaultModel,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    await model_catalog_service.set_default_vision_model(db, body.model)
    return body


@router.get("/title-mode", response_model=TitleMode)
async def get_title_mode_endpoint(db: AsyncSession = Depends(get_db), _admin: User = Depends(require_admin)):
    """Whether a new chat's title comes from quick local text processing
    ("simple") or by asking the model itself ("smart") — see
    app.services.chat_settings_service.get_title_mode."""
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
    """Open to any logged-in user, not admin-only — unlike title-mode,
    chat.js needs this value client-side to decide whether to poll or
    open a live /subscribe connection for a channel chat it has open
    (see app.services.chat_settings_service.get_channel_delivery_mode)."""
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
    """How long the app waits for a model to finish a reply before
    giving up on it — see app.services.chat_settings_service.get_reply_timeout_seconds.
    Admin-only to both read and change, unlike channel-delivery-mode
    above: this is a pure backend operational cap, nothing in the
    frontend needs to branch on it."""
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
    """Same as reply-timeout above, but for a reply with an image attached — see
    app.services.chat_settings_service.get_vision_reply_timeout_seconds' own docstring for why that needs its own,
    more generous setting rather than sharing reply-timeout's."""
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
    """How long the Stats page's Telemetry/System views keep their poller-written history — see
    app.services.retention_poller. Admin-only to both read and change, same reasoning as reply-timeout above."""
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
    """Every account, for the System tab's Users section. Admin-only —
    usernames and roles aren't sensitive exactly, but there's no reason
    for a regular user to see the full account list either."""
    return (await db.execute(select(User).order_by(User.created_at))).scalars().all()


@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(
    body: UserCreate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Creates a new account. Admin-only: there's no public
    self-registration in this app by design, since it's meant for a
    private, invite-only deployment."""
    try:
        return await user_service.create_user(db, body)
    except OllamaError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.patch("/users/{username}", response_model=UserOut)
async def update_user(
    username: str,
    body: UserUpdate,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Edits an existing account: change its password, promote/demote
    between admin and user, and/or disable or re-enable it."""
    return await user_service.update_user(db, username, body)


@router.delete("/users/{username}", response_model=OkResponse)
async def delete_user(
    username: str,
    db: AsyncSession = Depends(get_db),
    _admin: User = Depends(require_admin),
):
    """Permanently deletes an account and everything it owns. There's no
    undo; disabling (via PATCH above) is the reversible alternative."""
    await user_service.delete_user(db, username)
    return OkResponse()
