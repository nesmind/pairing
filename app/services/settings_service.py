"""Per-user and system-wide settings: default model, default generation params, admin RAG upload limits,
local-instances/proxy-mode knobs and the Ollama and Matricxon server configs. Reply-generation
settings (title mode, channel delivery, reply timeout) are in chat_settings_service.py (file-size rule); the
UI theme in theme_service.py; the model catalog in model_catalog_service.py; accounts in user_service.py.
This file is the plain key/value half of the Settings page.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DEFAULT_GENERATION_PARAMS, DEFAULT_MODEL
from app.models import SYSTEM_OWNER_ID, AppSetting, User
from app.schemas import MatricxonServerConfig, OllamaServerConfig, RagLimits
from app.services import engine_service, secret_crypto

DEFAULTS_KEY = "default_generation_params"
DEFAULT_MODEL_KEY = "default_model"
RAG_LIMITS_KEY = "rag_limits"
INSTANCE_COUNT_KEY = "instance_count"
PROXY_MODE_KEY = "proxy_mode"
OLLAMA_SERVER_CONFIG_KEY = "ollama_server_config"
MATRICXON_SERVER_CONFIG_KEY = "matricxon_server_config"

# Defaults before an admin saves anything on the System tab: 5MB per file, 1GB (1024MB) per user.
_DEFAULT_MAX_FILE_MB = 5.0
_DEFAULT_MAX_USER_SPACE_MB = 1024.0

# Local app instances to run (instance_service): 1 (just the primary) until raised in Settings > System.
DEFAULT_INSTANCE_COUNT = 1

# Whether the app load-balances across local instances itself ("local", default; instance_pool/instance_proxy)
# or leaves it to an external reverse proxy ("proxy"). Inert with one instance.
_VALID_PROXY_MODES = {"proxy", "local"}
DEFAULT_PROXY_MODE = "local"


def default_model_key(engine: str) -> str:
    """The AppSetting key for one engine's remembered default model. Per engine, not shared, so switching between
    Ollama and Matricxon (engine_service) never wipes or contaminates the other's pick. With one shared key an
    admin's default vanished on every switch: the old engine_switch_service wiped it so it could
    never name a tag the *other* engine lacks. Per-engine keys avoid that: each stays valid within its own installed
    set.
    """
    return f"{DEFAULT_MODEL_KEY}:{engine}"


async def get_default_model(db: AsyncSession, user: User) -> str:
    """`user`'s default model for the active engine (engine_service.current_engine): what a new conversation uses if
    none is specified (conversation_service.create_conversation). Fallback like get_default_params: the user's
    choice for this engine, then the system row, then app.config.DEFAULT_MODEL.
    """
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
    """`user`'s default generation params for new conversations (conversation_service.py). Falls back to the
    system row (SYSTEM_OWNER_ID), then app.config.DEFAULT_GENERATION_PARAMS; a user's save affects only their own future
    chats.
    """
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
    """The admin RAG upload caps (enforced in document_upload.py). System-wide."""
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
    """Local app instances to run (read by instance_service.reconcile_on_startup and the Settings page). System-wide,
    admin-set in Settings > System; not in .env, as nothing needs it before the DB is up (unlike
    DATABASE_URL/AUTO_MIGRATE).
    """
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
    """ "local" (app balances across local instances) or "proxy" (external reverse proxy). System-wide; read once
    at primary startup and cached in instance_pool, not per request.
    """
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


async def get_ollama_server_config(db: AsyncSession) -> OllamaServerConfig:
    """How to run Ollama (read by ollama_process/ollama_pool and the Settings page). System-wide, admin-set, with
    no .env fallback: the only source of truth from a fresh install on.
    """
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
    """How to run Matricxon (matricxon_process/matricxon_pool); same shape as get_ollama_server_config."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MATRICXON_SERVER_CONFIG_KEY))
    if row:
        value = dict(row.value)
        if value.get("admin_secret"):
            value["admin_secret"] = secret_crypto.decrypt(value["admin_secret"])
        return MatricxonServerConfig(**value)
    return MatricxonServerConfig()


async def set_matricxon_server_config(db: AsyncSession, config: MatricxonServerConfig) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, MATRICXON_SERVER_CONFIG_KEY))
    value = config.model_dump()
    if value.get("admin_secret"):
        value["admin_secret"] = secret_crypto.encrypt(value["admin_secret"])
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=MATRICXON_SERVER_CONFIG_KEY, value=value))
    else:
        row.value = value
    await db.commit()
