"""
Settings > System > Outbound HTTP proxy: get/set the admin-configured
HTTP proxy this app's own internet-facing downloads should route
through (Ollama/ComfyUI installers, and Ollama's own model pulls in
local mode). Split out from app.services.settings_service purely to
keep that file under CLAUDE.md's file-size rule — same AppSetting-
backed pattern as get_comfyui_config/set_comfyui_config there.

The password is encrypted at rest (see app.services.secret_crypto) —
stored under the "password_encrypted" key in the saved JSON, never as
plaintext "password". This is meaningfully different from the MySQL
database password (app.services.db_config_url), which lives in .env
right alongside the encryption key itself and so is deliberately left
unencrypted — see secret_crypto's own module docstring for why.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.schemas import HttpProxyConfig
from app.services import secret_crypto

HTTP_PROXY_CONFIG_KEY = "http_proxy_config"


async def get_http_proxy_config(db: AsyncSession) -> HttpProxyConfig:
    """The real, unmasked config (a real saved password included) —
    for internal callers that need a working proxy_url() (the installer/
    ollama_process router wiring). Never hand this straight to a browser
    as JSON; see get_http_proxy_config_for_display for that. System-wide,
    admin-configured — disabled (all fields empty) until an admin has
    saved it at least once."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, HTTP_PROXY_CONFIG_KEY))
    if not row:
        return HttpProxyConfig()
    value = dict(row.value)
    # Legacy rows saved before encryption was added may still carry a
    # plaintext "password" key — left as-is here (it reads back fine,
    # and gets re-saved as ciphertext on the next successful save).
    encrypted_password = value.pop("password_encrypted", None)
    if encrypted_password:
        value["password"] = secret_crypto.decrypt(encrypted_password)
    return HttpProxyConfig(**value)


async def get_http_proxy_config_for_display(db: AsyncSession) -> HttpProxyConfig:
    """Same config, with `password` redacted and `has_password` set
    instead — what GET/PUT actually return to the browser. Same
    never-re-display-a-saved-secret precedent as
    app.schemas.db_config.DbConfig."""
    config = await get_http_proxy_config(db)
    return config.model_copy(update={"password": None, "has_password": bool(config.password)})


async def set_http_proxy_config(db: AsyncSession, config: HttpProxyConfig) -> None:
    """A blank/missing `password` on `config` means "keep whatever
    password is already saved" — the browser never has the real value to
    resubmit (see get_http_proxy_config_for_display), so a blank field
    must never be treated as "clear the password" here."""
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
