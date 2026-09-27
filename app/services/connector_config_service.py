"""
Admin-configured settings for a Connector (see app.services.connectors) — one
AppSetting row per connector, system-wide. Secret fields (see
ConnectorConfigField.secret) are encrypted at rest via app.services.secret_crypto,
stored under "<field>_encrypted" inside the saved JSON `value`; never re-displayed
once saved (see get_config_for_display) — same precedent as
app.services.http_proxy_service.

This module is the DB-backed source of truth; app.services.connector_config_cache
holds the multi-instance-safe in-process copy every Connector-backed InferenceEngine
adapter actually reads from at call time (no `db` session is ever threaded through an
InferenceEngine method).
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services import secret_crypto
from app.services.connectors import get_connector

_CONFIG_KEY_TEMPLATE = "connector_{connector_id}_config"


def _key(connector_id: str) -> str:
    # connector_id is always a fixed, developer-chosen slug from app.services.connectors.registry.CONNECTORS,
    # never user input — safe to interpolate directly into an AppSetting key.
    return _CONFIG_KEY_TEMPLATE.format(connector_id=connector_id)


async def get_config(db: AsyncSession, connector_id: str) -> dict[str, str]:
    """The real, unmasked field values (secrets decrypted) — for internal callers
    (app.services.connector_config_cache) only. Never hand this straight to a
    browser as JSON; see get_config_for_display for that. A field that was never
    saved (or failed to decrypt — see secret_crypto.decrypt) is simply absent."""
    connector = get_connector(connector_id)
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    stored = dict(row.value.get("fields", {})) if row else {}
    values: dict[str, str] = {}
    for field in connector.config_fields:
        if field.secret:
            encrypted = stored.get(f"{field.name}_encrypted")
            decrypted = secret_crypto.decrypt(encrypted) if encrypted else None
            if decrypted:
                values[field.name] = decrypted
        elif stored.get(field.name):
            values[field.name] = stored[field.name]
    return values


async def get_config_for_display(db: AsyncSession, connector_id: str) -> dict[str, str | bool]:
    """Same values, with every secret field replaced by `has_<field>: bool` instead
    of the real value — same never-re-display-a-saved-secret precedent as
    app.services.http_proxy_service.get_http_proxy_config_for_display."""
    connector = get_connector(connector_id)
    values = await get_config(db, connector_id)
    display: dict[str, str | bool] = {}
    for field in connector.config_fields:
        if field.secret:
            display[f"has_{field.name}"] = bool(values.get(field.name))
        else:
            display[field.name] = values.get(field.name, "")
    return display


async def is_configured(db: AsyncSession, connector_id: str) -> bool:
    """Every required field has a saved (non-blank) value."""
    connector = get_connector(connector_id)
    values = await get_config(db, connector_id)
    return all(values.get(f.name) for f in connector.config_fields if f.required)


async def set_config(db: AsyncSession, connector_id: str, values: dict[str, str]) -> None:
    """A blank/missing value for a `secret` field means "keep whatever is already
    saved" — the browser never has the real secret to resubmit (see
    get_config_for_display), so a blank field must never be treated as "clear it".
    Non-secret fields have no such special case: their current value is always shown
    back to the admin (get_config_for_display), so a blank submission really does
    mean "clear this field"."""
    connector = get_connector(connector_id)
    existing = await get_config(db, connector_id)
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    enabled = bool(row.value.get("enabled", False)) if row else False

    stored: dict[str, str] = {}
    for field in connector.config_fields:
        if field.secret:
            new_value = values.get(field.name) or existing.get(field.name)
            if new_value:
                stored[f"{field.name}_encrypted"] = secret_crypto.encrypt(new_value)
        else:
            new_value = values.get(field.name, "")
            if new_value:
                stored[field.name] = new_value

    # Any config edit invalidates a previous "Test connection" result — new credentials/endpoint haven't been
    # proven to actually work yet (see get_test_result/set_test_result), so a connector must always be re-tested
    # before it can become ready again (app.services.connector_config_cache.is_ready).
    value = {"enabled": enabled, "fields": stored, "last_test_passed": False, "last_test_message": None}
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=_key(connector_id), value=value))
    else:
        row.value = value
    await db.commit()


async def get_enabled(db: AsyncSession, connector_id: str) -> bool:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    return bool(row.value.get("enabled", False)) if row else False


async def set_enabled(db: AsyncSession, connector_id: str, enabled: bool) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    if row is None:
        db.add(AppSetting(owner_id=SYSTEM_OWNER_ID, key=_key(connector_id), value={"enabled": enabled, "fields": {}}))
    else:
        row.value = {**row.value, "enabled": enabled}
    await db.commit()


async def get_test_result(db: AsyncSession, connector_id: str) -> tuple[bool, str | None]:
    """Whether the last "Test connection" action (see app/routers/connectors.py) against this connector's
    currently-saved config actually succeeded, and its message — (False, None) if never tested, or if the config
    has changed since the last test (see set_config, which resets this on every save)."""
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    if not row:
        return False, None
    return bool(row.value.get("last_test_passed", False)), row.value.get("last_test_message")


async def set_test_result(db: AsyncSession, connector_id: str, passed: bool, message: str) -> None:
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    if row is None:
        db.add(
            AppSetting(
                owner_id=SYSTEM_OWNER_ID,
                key=_key(connector_id),
                value={"enabled": False, "fields": {}, "last_test_passed": passed, "last_test_message": message},
            )
        )
    else:
        row.value = {**row.value, "last_test_passed": passed, "last_test_message": message}
    await db.commit()
