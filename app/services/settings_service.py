"""
Per-user/system-wide settings: default model, default generation params,
admin-configured RAG upload limits, the local-instances/proxy-mode
knobs, and how to launch ComfyUI (see app/services/comfyui_service.py
for the process supervisor that actually reads this). See
app/services/chat_settings_service.py instead for settings that
specifically govern how a reply is generated (title mode, channel
delivery mode, reply timeout) — split out purely to keep this file
under CLAUDE.md's file-size rule. See app/services/theme_service.py for
the app-wide UI theme (a separate file — see that file's own docstring
for why), app/services/model_catalog_service.py for the model
catalog itself, and app/services/user_service.py for account management
— this file is only the "simple key/value setting" half of the Settings
page.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DEFAULT_GENERATION_PARAMS, DEFAULT_MODEL
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import ComfyUIProcessConfig, MatricxonServerConfig, OllamaServerConfig, RagLimits
from app.services import engine_service

DEFAULTS_KEY = "default_generation_params"
DEFAULT_MODEL_KEY = "default_model"
RAG_LIMITS_KEY = "rag_limits"
INSTANCE_COUNT_KEY = "instance_count"
PROXY_MODE_KEY = "proxy_mode"
COMFYUI_CONFIG_KEY = "comfyui_config"
OLLAMA_SERVER_CONFIG_KEY = "ollama_server_config"
MATRICXON_SERVER_CONFIG_KEY = "matricxon_server_config"

# Defaults before an admin has ever saved anything on the System tab —
# matches what was asked for: 5MB per file, 1GB (1024MB) per user.
_DEFAULT_MAX_FILE_MB = 5.0
_DEFAULT_MAX_USER_SPACE_MB = 1024.0

# How many local app-process instances should be running (see
# app.services.instance_service) — 1 (just the primary, today's only
# behavior) until an admin raises it from Settings > System.
DEFAULT_INSTANCE_COUNT = 1

# Whether the app itself load-balances across local instances ("local",
# the default — see app.services.instance_pool/instance_proxy) or
# leaves that entirely to an external reverse proxy, if any ("proxy").
# Inert either way with the default single instance — only starts doing
# anything once an admin raises the instance count above 1.
_VALID_PROXY_MODES = {"proxy", "local"}
DEFAULT_PROXY_MODE = "local"


def default_model_key(engine: str) -> str:
    """The AppSetting key for one engine's own remembered default model — scoped per engine, not a single
    shared value, so switching between Ollama and Matricxon (see app.services.engine_service) never wipes or
    cross-contaminates the other's pick. Confirmed live, 2026-09-22: with a single shared key, an admin's
    default model vanished after every engine switch — the old app.services.engine_switch_service explicitly
    wiped it so it could never point at a tag the *other* engine doesn't have. Per-engine keys remove that
    failure mode entirely: each engine's own stored default only ever needs to stay valid within its own
    installed set, so there's nothing left to go stale from a switch."""
    return f"{DEFAULT_MODEL_KEY}:{engine}"


async def get_default_model(db: AsyncSession, user: User) -> str:
    """Returns `user`'s own default model for whichever engine is currently active (see
    app.services.engine_service.current_engine) — what a brand-new conversation of theirs uses if they didn't
    specify one (see app.services.conversation_service.create_conversation). Same fallback chain as
    get_default_params: the user's own choice for this engine, then a system-wide row for it, then
    app.config.DEFAULT_MODEL."""
    key = default_model_key(engine_service.current_engine())
    row = await db.get(AppSetting, (user.id, key))
    if row:
        return row.value["model"]
    system_row = await db.get(AppSetting, (SYSTEM_OWNER_ID, key))
    if system_row:
        return system_row.value["model"]
    return DEFAULT_MODEL


async def set_default_model(db: AsyncSession, owner_id: str, model: str) -> None:
    key = default_model_key(engine_service.current_engine())
    row = await db.get(AppSetting, (owner_id, key))
    value = {"model": model}
    if row is None:
        db.add(AppSetting(owner_id=owner_id, key=key, value=value))
    else:
        row.value = value
    await db.commit()


async def get_default_params(db: AsyncSession, user: User) -> dict:
    """Returns `user`'s own default generation params — what a brand-new
    conversation of theirs starts with (see
    app/services/conversation_service.py). Falls back to a system-wide
    row (SYSTEM_OWNER_ID) if they haven't saved their own yet, and
    finally to the hardcoded app.config.DEFAULT_GENERATION_PARAMS if
    neither exists — each user's own save only ever affects their own
    future chats, which is what makes settings genuinely per-user."""
    row = await db.get(AppSetting, (user.id, DEFAULTS_KEY))
    if row:
        return dict(row.value)
    system_row = await db.get(AppSetting, (SYSTEM_OWNER_ID, DEFAULTS_KEY))
    if system_row:
        return dict(system_row.value)
    return dict(DEFAULT_GENERATION_PARAMS)


async def set_default_params(db: AsyncSession, owner_id: str, value: dict) -> None:
    row = await db.get(AppSetting, (owner_id, DEFAULTS_KEY))
    if row is None:
        db.add(AppSetting(owner_id=owner_id, key=DEFAULTS_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_rag_limits(db: AsyncSession) -> RagLimits:
    """The current admin-configured RAG upload caps (see
    app/services/document_upload.py's upload handler, which enforces
    these). System-wide — not a per-user preference."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, RAG_LIMITS_KEY))
    if row:
        return RagLimits(**row.value)
    return RagLimits(max_file_mb=_DEFAULT_MAX_FILE_MB, max_user_space_mb=_DEFAULT_MAX_USER_SPACE_MB)


async def set_rag_limits(db: AsyncSession, limits: RagLimits) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, RAG_LIMITS_KEY))
    value = limits.model_dump()
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=RAG_LIMITS_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_instance_count(db: AsyncSession) -> int:
    """How many local app-process instances should be running (see
    app.services.instance_service.reconcile_on_startup, the only reader
    of this besides the Settings page itself). System-wide, admin-
    configured from Settings > System — not a per-user preference, and
    not read from .env: unlike DATABASE_URL/AUTO_MIGRATE, nothing needs
    this value before the database is already up and queryable."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, INSTANCE_COUNT_KEY))
    if row and isinstance(row.value.get("count"), int):
        return row.value["count"]
    return DEFAULT_INSTANCE_COUNT


async def set_instance_count(db: AsyncSession, count: int) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, INSTANCE_COUNT_KEY))
    value = {"count": count}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=INSTANCE_COUNT_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_proxy_mode(db: AsyncSession) -> str:
    """Whether the app load-balances across local instances itself
    ("local" — the default) or leaves that to an external reverse
    proxy, if any ("proxy"). System-wide, admin-configured from
    Settings > System — read once at primary startup and cached in
    app.services.instance_pool, not re-read from the DB per request."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, PROXY_MODE_KEY))
    if row and row.value.get("mode") in _VALID_PROXY_MODES:
        return row.value["mode"]
    return DEFAULT_PROXY_MODE


async def set_proxy_mode(db: AsyncSession, mode: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, PROXY_MODE_KEY))
    value = {"mode": mode}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=PROXY_MODE_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_comfyui_config(db: AsyncSession) -> ComfyUIProcessConfig:
    """How to launch ComfyUI on this machine (see
    app.services.comfyui_service.start, the only reader besides the
    Settings page itself) — system-wide, admin-configured, unset (all
    None) until an admin has saved it at least once."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, COMFYUI_CONFIG_KEY))
    if row:
        return ComfyUIProcessConfig(**row.value)
    return ComfyUIProcessConfig()


async def set_comfyui_config(db: AsyncSession, config: ComfyUIProcessConfig) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, COMFYUI_CONFIG_KEY))
    value = config.model_dump()
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=COMFYUI_CONFIG_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_ollama_server_config(db: AsyncSession) -> OllamaServerConfig:
    """How to run Ollama (see app.services.ollama_process/ollama_pool,
    the only readers besides the Settings page itself) — system-wide,
    admin-configured, with no .env/environment-variable fallback: this is
    the one and only source of truth from a fresh install onward."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, OLLAMA_SERVER_CONFIG_KEY))
    if row:
        return OllamaServerConfig(**row.value)
    return OllamaServerConfig()


async def set_ollama_server_config(db: AsyncSession, config: OllamaServerConfig) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, OLLAMA_SERVER_CONFIG_KEY))
    value = config.model_dump()
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=OLLAMA_SERVER_CONFIG_KEY, value=value))
    else:
        row.value = value
    await db.commit()


async def get_matricxon_server_config(db: AsyncSession) -> MatricxonServerConfig:
    """How to run Matricxon (see app.services.matricxon_process/matricxon_pool) — same
    system-wide, admin-configured, no-.env-fallback shape as get_ollama_server_config above."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MATRICXON_SERVER_CONFIG_KEY))
    if row:
        return MatricxonServerConfig(**row.value)
    return MatricxonServerConfig()


async def set_matricxon_server_config(db: AsyncSession, config: MatricxonServerConfig) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MATRICXON_SERVER_CONFIG_KEY))
    value = config.model_dump()
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=MATRICXON_SERVER_CONFIG_KEY, value=value))
    else:
        row.value = value
    await db.commit()
