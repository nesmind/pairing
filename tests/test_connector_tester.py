"""Tests for services/connector_tester.py and the test-values route: form values are tested before saving, and only a
test of the saved values is recorded."""

import pytest
from fastapi import HTTPException

from app.routers import connectors
from app.schemas import ConnectorConfigUpdate
from app.services import connector_config_cache, connector_config_service, groq_client, server_pool_broadcast
from app.services.connector_tester import ConnectorTester


@pytest.fixture(autouse=True)
def fake_groq(monkeypatch):
    """Groq's test passes only for the key "good"; the tested configs are kept."""
    connector_config_cache._cache = {}
    tested: list[dict] = []

    async def fake_test(config):
        tested.append(config)
        return config.get("api_key") == "good", "checked"

    async def no_broadcast(_path):
        return []

    monkeypatch.setattr(groq_client, "test_connection", fake_test)
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", no_broadcast)
    yield tested
    connector_config_cache._cache = {}


@pytest.mark.asyncio
async def test_unsaved_values_are_tested_but_not_recorded(db, fake_groq):
    result = await ConnectorTester(db, "groq").run({"api_key": "good"})

    assert (result.passed, result.saved) == (True, False)
    assert fake_groq[-1] == {"api_key": "good"}
    assert await connector_config_service.get_test_result(db, "groq") == (False, None)


@pytest.mark.asyncio
async def test_a_blank_secret_tests_the_saved_one_and_records_it(db, fake_groq):
    await connector_config_service.set_config(db, "groq", {"api_key": "good"})

    result = await ConnectorTester(db, "groq").run({"api_key": ""})

    assert (result.passed, result.saved) == (True, True)
    assert await connector_config_service.get_test_result(db, "groq") == (True, "checked")


@pytest.mark.asyncio
async def test_a_new_secret_over_a_saved_one_is_not_recorded(db, fake_groq):
    await connector_config_service.set_config(db, "groq", {"api_key": "good"})

    result = await ConnectorTester(db, "groq").run({"api_key": "bad"})

    assert (result.passed, result.saved) == (False, False)
    assert await connector_config_service.get_test_result(db, "groq") == (False, None)


def test_merge_keeps_blank_secrets_and_clears_blank_text():
    merged = connector_config_service.merge_values(
        "openrouter", {"api_key": "saved", "models": "all"}, {"api_key": "", "models": ""}
    )
    assert merged == {"api_key": "saved"}


@pytest.mark.asyncio
async def test_route_rejects_an_unknown_connector(db):
    with pytest.raises(HTTPException) as exc:
        await connectors.test_connector_values("nonexistent", ConnectorConfigUpdate(values={}), db=db, _admin=None)
    assert exc.value.status_code == 404
