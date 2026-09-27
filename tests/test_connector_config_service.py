"""Unit tests for app/services/connector_config_service.py — the DB-backed connector
config store, same round-trip/masking/blank-keeps-existing conventions as
tests/test_http_proxy_service.py, generalized to a connector's own declared
config_fields instead of one hand-written schema."""

import pytest

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.services import connector_config_service
from app.services.connectors.registry import get_connector

_ID = "runpod"


@pytest.mark.asyncio
async def test_get_config_is_empty_when_never_saved(db):
    assert await connector_config_service.get_config(db, _ID) == {}


@pytest.mark.asyncio
async def test_set_config_round_trips_non_secret_fields(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "model": "llama3"})

    config = await connector_config_service.get_config(db, _ID)

    assert config["endpoint_id"] == "ep-1"
    assert config["model"] == "llama3"


@pytest.mark.asyncio
async def test_set_config_round_trips_a_secret_field(db):
    await connector_config_service.set_config(db, _ID, {"api_key": "s3cret"})

    assert (await connector_config_service.get_config(db, _ID))["api_key"] == "s3cret"


@pytest.mark.asyncio
async def test_get_config_for_display_masks_the_secret_field(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "s3cret"})

    display = await connector_config_service.get_config_for_display(db, _ID)

    assert display["endpoint_id"] == "ep-1"  # not a secret — shown as-is
    assert "api_key" not in display
    assert display["has_api_key"] is True


@pytest.mark.asyncio
async def test_get_config_for_display_reports_no_secret_when_none_saved(db):
    display = await connector_config_service.get_config_for_display(db, _ID)
    assert display["has_api_key"] is False


@pytest.mark.asyncio
async def test_set_config_with_a_blank_secret_field_keeps_the_existing_one(db):
    await connector_config_service.set_config(db, _ID, {"api_key": "original"})
    # Simulates the admin saving again (e.g. just changing endpoint_id) without
    # retyping a secret they were never shown back.
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-2", "api_key": ""})

    config = await connector_config_service.get_config(db, _ID)

    assert config["api_key"] == "original"
    assert config["endpoint_id"] == "ep-2"


@pytest.mark.asyncio
async def test_set_config_with_a_new_secret_value_replaces_the_existing_one(db):
    await connector_config_service.set_config(db, _ID, {"api_key": "original"})
    await connector_config_service.set_config(db, _ID, {"api_key": "replaced"})

    assert (await connector_config_service.get_config(db, _ID))["api_key"] == "replaced"


@pytest.mark.asyncio
async def test_set_config_with_a_blank_non_secret_field_clears_it(db):
    """Unlike a secret field, a non-secret field's current value is always shown
    back to the admin (get_config_for_display) — a blank submission really does
    mean "clear this field", not "keep the old one"."""
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})
    await connector_config_service.set_config(db, _ID, {"endpoint_id": ""})

    assert await connector_config_service.get_config(db, _ID) == {}


@pytest.mark.asyncio
async def test_set_config_stores_the_secret_field_encrypted_not_plaintext(db):
    await connector_config_service.set_config(db, _ID, {"api_key": "s3cret"})

    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, f"connector_{_ID}_config"))

    assert "api_key" not in row.value["fields"]
    assert "s3cret" not in row.value["fields"]["api_key_encrypted"]


@pytest.mark.asyncio
async def test_is_configured_false_until_every_required_field_is_saved(db):
    """A real save always submits every field together (the admin form is prefilled with the connector's
    current values — see get_config_for_display) — set_config isn't additive across separate calls for
    non-secret fields (see test_set_config_with_a_blank_non_secret_field_clears_it), so this saves all
    required fields in one call, the same as a real form submission would."""
    connector = get_connector(_ID)
    required = [f.name for f in connector.config_fields if f.required]
    assert required, "expected the runpod connector to declare required fields"

    assert await connector_config_service.is_configured(db, _ID) is False

    await connector_config_service.set_config(db, _ID, dict.fromkeys(required, "value"))
    assert await connector_config_service.is_configured(db, _ID) is True


@pytest.mark.asyncio
async def test_enabled_defaults_to_false_and_round_trips(db):
    assert await connector_config_service.get_enabled(db, _ID) is False

    await connector_config_service.set_enabled(db, _ID, True)
    assert await connector_config_service.get_enabled(db, _ID) is True

    await connector_config_service.set_enabled(db, _ID, False)
    assert await connector_config_service.get_enabled(db, _ID) is False


@pytest.mark.asyncio
async def test_set_enabled_does_not_disturb_saved_config(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})
    await connector_config_service.set_enabled(db, _ID, True)

    assert (await connector_config_service.get_config(db, _ID))["endpoint_id"] == "ep-1"


@pytest.mark.asyncio
async def test_set_config_does_not_disturb_the_enabled_flag(db):
    await connector_config_service.set_enabled(db, _ID, True)
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})

    assert await connector_config_service.get_enabled(db, _ID) is True


@pytest.mark.asyncio
async def test_get_test_result_defaults_to_never_tested(db):
    assert await connector_config_service.get_test_result(db, _ID) == (False, None)


@pytest.mark.asyncio
async def test_set_test_result_round_trips(db):
    await connector_config_service.set_test_result(db, _ID, True, "Connected successfully.")
    assert await connector_config_service.get_test_result(db, _ID) == (True, "Connected successfully.")

    await connector_config_service.set_test_result(db, _ID, False, "RunPod rejected the API key.")
    assert await connector_config_service.get_test_result(db, _ID) == (False, "RunPod rejected the API key.")


@pytest.mark.asyncio
async def test_set_test_result_does_not_disturb_saved_config_or_enabled(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})
    await connector_config_service.set_enabled(db, _ID, True)

    await connector_config_service.set_test_result(db, _ID, True, "ok")

    assert (await connector_config_service.get_config(db, _ID))["endpoint_id"] == "ep-1"
    assert await connector_config_service.get_enabled(db, _ID) is True


@pytest.mark.asyncio
async def test_set_config_resets_a_previously_passed_test(db):
    """Any config edit invalidates a prior "Test connection" pass — new credentials/endpoint haven't been
    proven to work yet, so the connector must be re-tested before it can be ready again."""
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})
    await connector_config_service.set_test_result(db, _ID, True, "Connected successfully.")

    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-2"})

    assert await connector_config_service.get_test_result(db, _ID) == (False, None)
