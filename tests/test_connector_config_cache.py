"""Tests for services/connector_config_cache.py, the multi-instance-safe in-process cache runpod_client reads,
mirroring tests/test_engine_service.py's write-through-cache conventions.
"""

import pytest

from app.services import connector_config_cache, connector_config_service

pytestmark = pytest.mark.usefixtures("runpod_listed")  # RunPod is the example connector here

_ID = "runpod"


@pytest.fixture(autouse=True)
def reset_cache():
    connector_config_cache._cache = {}
    yield
    connector_config_cache._cache = {}


@pytest.mark.asyncio
async def test_get_config_is_empty_before_any_load(db):
    assert connector_config_cache.get_config(_ID) == {}
    assert connector_config_cache.is_enabled(_ID) is False
    assert connector_config_cache.is_ready(_ID) is False


@pytest.mark.asyncio
async def test_load_cache_from_db_populates_config_and_enabled(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "k", "model": "m"})
    await connector_config_service.set_enabled(db, _ID, True)

    await connector_config_cache.load_cache_from_db(db)

    assert connector_config_cache.get_config(_ID) == {"endpoint_id": "ep-1", "api_key": "k", "model": "m"}
    assert connector_config_cache.is_enabled(_ID) is True


@pytest.mark.asyncio
async def test_is_ready_requires_enabled_configured_and_a_passed_test(db):
    await connector_config_service.set_enabled(db, _ID, True)
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.is_ready(_ID) is False

    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "k", "model": "m"})
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.is_ready(_ID) is False

    await connector_config_service.set_test_result(db, _ID, True, "Connected successfully.")
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.is_ready(_ID) is True


@pytest.mark.asyncio
async def test_is_ready_false_when_configured_and_tested_but_not_enabled(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "k", "model": "m"})
    await connector_config_service.set_test_result(db, _ID, True, "Connected successfully.")
    await connector_config_cache.load_cache_from_db(db)

    assert connector_config_cache.is_ready(_ID) is False


@pytest.mark.asyncio
async def test_is_ready_false_when_last_test_failed(db):
    await connector_config_service.set_enabled(db, _ID, True)
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "k", "model": "m"})
    await connector_config_service.set_test_result(db, _ID, False, "RunPod rejected the API key.")
    await connector_config_cache.load_cache_from_db(db)

    assert connector_config_cache.is_ready(_ID) is False


@pytest.mark.asyncio
async def test_is_ready_false_again_after_a_config_edit_resets_the_test(db):
    await connector_config_service.set_enabled(db, _ID, True)
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1", "api_key": "k", "model": "m"})
    await connector_config_service.set_test_result(db, _ID, True, "Connected successfully.")
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.is_ready(_ID) is True

    # Editing the config (even re-saving the same required fields) invalidates the prior test result.
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-2", "api_key": "k", "model": "m"})
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.is_ready(_ID) is False


@pytest.mark.asyncio
async def test_load_cache_from_db_reflects_a_config_change(db):
    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-1"})
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.get_config(_ID)["endpoint_id"] == "ep-1"

    await connector_config_service.set_config(db, _ID, {"endpoint_id": "ep-2"})
    await connector_config_cache.load_cache_from_db(db)
    assert connector_config_cache.get_config(_ID)["endpoint_id"] == "ep-2"
