"""Admin settings for a Connector (services.connectors): one system-wide AppSetting row per connector.
Secret fields (ConnectorConfigField.secret) are encrypted at rest (secret_crypto) as "<field>_encrypted"
in the JSON `value` and never re-displayed (get_config_for_display), like http_proxy_service.

This is the DB source of truth; connector_config_cache holds the in-process copy the Connector-backed
InferenceEngine adapters read at call time.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services import secret_crypto
from app.services.connectors import get_connector

_CONFIG_KEY_TEMPLATE = "connector_{connector_id}_config"


def _key(connector_id: str) -> str:
    # connector_id is a developer-chosen slug from connectors.registry.CONNECTORS, never user input: safe in an
    # AppSetting key.
    return _CONFIG_KEY_TEMPLATE.format(connector_id=connector_id)


async def get_config(db: AsyncSession, connector_id: str) -> dict[str, str]:
    """Real, unmasked values (secrets decrypted) for internal callers (connector_config_cache) only; never send
    to a browser (see get_config_for_display). Never-saved or undecryptable fields are absent.
    """
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
    """Same values with secrets replaced by `has_<field>: bool` (like
    http_proxy_service.get_http_proxy_config_for_display)."""
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
    """Every required field has a saved non-blank value."""
    connector = get_connector(connector_id)
    values = await get_config(db, connector_id)
    return all(values.get(f.name) for f in connector.config_fields if f.required)


def merge_values(connector_id: str, existing: dict[str, str], values: dict[str, str]) -> dict[str, str]:
    """Form values over the saved ones: a blank secret keeps the saved one, a blank text field clears it."""
    merged: dict[str, str] = {}
    for field in get_connector(connector_id).config_fields:
        if field.secret:
            value = values.get(field.name) or existing.get(field.name, "")
        else:
            value = values.get(field.name, "")
        if value:
            merged[field.name] = value
    return merged


async def set_config(db: AsyncSession, connector_id: str, values: dict[str, str]) -> None:
    """A blank/missing `secret` value keeps the saved one (the browser never has it, see get_config_for_display).
    Non-secret fields are shown back, so blank really clears them.
    """
    connector = get_connector(connector_id)
    existing = await get_config(db, connector_id)
    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, _key(connector_id)))
    enabled = bool(row.value.get("enabled", False)) if row else False

    merged = merge_values(connector_id, existing, values)
    stored: dict[str, str] = {}
    for field in connector.config_fields:
        if field.name in merged:
            key = f"{field.name}_encrypted" if field.secret else field.name
            stored[key] = secret_crypto.encrypt(merged[field.name]) if field.secret else merged[field.name]

    # Any edit invalidates the last "Test connection" result (new credentials are unproven;
    # get_test_result/set_test_result), so the connector must be re-tested to become ready
    # (connector_config_cache.is_ready).
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
    """Whether the last "Test connection" (routers/connectors.py) on the saved config succeeded, plus its message;
    (False, None) if never tested or the config changed since (set_config resets on save).
    """
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
