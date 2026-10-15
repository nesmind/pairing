"""Settings > System > Outbound HTTP proxy: get/set the proxy for internet-facing downloads (engine
installers, Ollama pulls in local mode). Split from settings_service for the file-size rule; same
AppSetting pattern as settings_service's get/set_ollama_server_config.

The password is encrypted at rest (secret_crypto) under "password_encrypted", never plaintext. Unlike the
MySQL password (db_config_url), which sits in .env beside the encryption key and is left unencrypted (see
secret_crypto).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import HttpProxyConfig
from app.services import secret_crypto

HTTP_PROXY_CONFIG_KEY = "http_proxy_config"


async def get_http_proxy_config(db: AsyncSession) -> HttpProxyConfig:
    """The real, unmasked config for internal callers needing a working proxy_url (installers, ollama_process
    wiring). Never send to a browser (use get_http_proxy_config_for_display). System-wide; disabled until saved once.
    """
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, HTTP_PROXY_CONFIG_KEY))
    if not row:
        return HttpProxyConfig()
    value = dict(row.value)
    # Legacy rows may hold a plaintext "password"; left as is (reads fine, re-saved encrypted on the next save).
    encrypted_password = value.pop("password_encrypted", None)
    if encrypted_password:
        value["password"] = secret_crypto.decrypt(encrypted_password)
    return HttpProxyConfig(**value)


async def get_http_proxy_config_for_display(db: AsyncSession) -> HttpProxyConfig:
    """Same config with `password` redacted and `has_password` set: what GET/PUT return to the browser (like
    db_config.DbConfig)."""
    config = await get_http_proxy_config(db)
    return config.model_copy(update={"password": None, "has_password": bool(config.password)})


async def set_http_proxy_config(db: AsyncSession, config: HttpProxyConfig) -> None:
    """A blank/missing `password` keeps the saved one: the browser never has the real value (see
    get_http_proxy_config_for_display), so blank must not clear it.
    """
    if not config.password:
        existing = await get_http_proxy_config(db)
        config = config.model_copy(update={"password": existing.password})

    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, HTTP_PROXY_CONFIG_KEY))
    value = config.model_dump(exclude={"has_password", "password"})
    if config.password:
        value["password_encrypted"] = secret_crypto.encrypt(config.password)
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=HTTP_PROXY_CONFIG_KEY, value=value))
    else:
        row.value = value
    await db.commit()
