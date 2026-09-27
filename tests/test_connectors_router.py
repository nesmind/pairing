"""Unit tests for app/routers/connectors.py — called directly, not through a
TestClient (see tests/test_engine_admin_router.py's own docstring for why this
project's router tests don't use that pattern)."""

import pytest
from fastapi import HTTPException

from app.routers import connectors
from app.schemas import ConnectorConfigUpdate, ConnectorEnabledUpdate
from app.services import connector_config_cache, runpod_client, server_pool_broadcast


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host):
        self.client = _FakeClient(host) if host else None


@pytest.fixture(autouse=True)
def reset_cache_and_broadcast(monkeypatch):
    connector_config_cache._cache = {}

    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    yield
    connector_config_cache._cache = {}


@pytest.mark.asyncio
async def test_list_connectors_includes_the_runpod_connector(db):
    result = await connectors.list_connectors(db=db, _admin=None)

    ids = [c.id for c in result.connectors]
    assert "runpod" in ids


@pytest.mark.asyncio
async def test_list_connectors_reports_not_configured_or_enabled_by_default(db):
    result = await connectors.list_connectors(db=db, _admin=None)

    runpod = next(c for c in result.connectors if c.id == "runpod")
    assert runpod.configured is False
    assert runpod.enabled is False
    assert runpod.ready is False


@pytest.mark.asyncio
async def test_update_connector_config_saves_and_refreshes_the_cache(db):
    result = await connectors.update_connector_config(
        "runpod",
        ConnectorConfigUpdate(values={"endpoint_id": "ep-1", "api_key": "k", "model": "m"}),
        db=db,
        _admin=None,
    )

    assert result.configured is True
    assert connector_config_cache.get_config("runpod")["endpoint_id"] == "ep-1"


@pytest.mark.asyncio
async def test_update_connector_config_never_echoes_the_saved_secret_back(db):
    result = await connectors.update_connector_config(
        "runpod",
        ConnectorConfigUpdate(values={"endpoint_id": "ep-1", "api_key": "s3cret", "model": "m"}),
        db=db,
        _admin=None,
    )

    assert "api_key" not in result.values
    assert result.values["has_api_key"] is True


@pytest.mark.asyncio
async def test_update_connector_enabled_alone_is_not_enough_to_be_ready(db):
    """Enabling a connector that's configured but never passed a "Test connection" must not make it ready — see
    test_test_connector_config_marks_it_ready_once_passed for the full happy path."""
    await connectors.update_connector_config(
        "runpod",
        ConnectorConfigUpdate(values={"endpoint_id": "ep-1", "api_key": "k", "model": "m"}),
        db=db,
        _admin=None,
    )

    result = await connectors.update_connector_enabled(
        "runpod", ConnectorEnabledUpdate(enabled=True), db=db, _admin=None
    )

    assert result.enabled is True
    assert result.ready is False
    assert connector_config_cache.is_ready("runpod") is False


@pytest.mark.asyncio
async def test_test_connector_config_marks_it_ready_once_passed(db, monkeypatch):
    async def fake_test_connection(config):
        assert config == {"endpoint_id": "ep-1", "api_key": "k", "model": "m"}
        return True, "Connected successfully."

    monkeypatch.setattr(runpod_client, "test_connection", fake_test_connection)
    await connectors.update_connector_config(
        "runpod",
        ConnectorConfigUpdate(values={"endpoint_id": "ep-1", "api_key": "k", "model": "m"}),
        db=db,
        _admin=None,
    )
    await connectors.update_connector_enabled("runpod", ConnectorEnabledUpdate(enabled=True), db=db, _admin=None)

    result = await connectors.test_connector_config("runpod", db=db, _admin=None)

    assert result.last_test_passed is True
    assert result.last_test_message == "Connected successfully."
    assert result.ready is True
    assert connector_config_cache.is_ready("runpod") is True


@pytest.mark.asyncio
async def test_test_connector_config_reports_a_failed_test(db, monkeypatch):
    async def fake_test_connection(config):
        return False, "RunPod rejected the API key (401 Unauthorized)."

    monkeypatch.setattr(runpod_client, "test_connection", fake_test_connection)
    await connectors.update_connector_config(
        "runpod",
        ConnectorConfigUpdate(values={"endpoint_id": "ep-1", "api_key": "bad", "model": "m"}),
        db=db,
        _admin=None,
    )
    await connectors.update_connector_enabled("runpod", ConnectorEnabledUpdate(enabled=True), db=db, _admin=None)

    result = await connectors.test_connector_config("runpod", db=db, _admin=None)

    assert result.last_test_passed is False
    assert result.ready is False


@pytest.mark.asyncio
async def test_test_connector_config_rejects_an_unknown_connector_id(db):
    with pytest.raises(HTTPException) as exc_info:
        await connectors.test_connector_config("nonexistent", db=db, _admin=None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_connector_config_rejects_an_unknown_connector_id(db):
    with pytest.raises(HTTPException) as exc_info:
        await connectors.update_connector_config("nonexistent", ConnectorConfigUpdate(values={}), db=db, _admin=None)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_connector_config_broadcasts_to_other_instances(monkeypatch, db):
    calls = []

    async def fake_broadcast(path):
        calls.append(path)
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    await connectors.update_connector_config(
        "runpod", ConnectorConfigUpdate(values={"endpoint_id": "ep-1"}), db=db, _admin=None
    )

    assert calls == ["/api/connectors/internal-refresh"]


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db):
    with pytest.raises(HTTPException) as exc_info:
        await connectors.internal_refresh(_FakeRequest("203.0.113.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_reloads_the_cache_from_the_shared_db(db):
    from app.services import connector_config_service

    await connector_config_service.set_config(db, "runpod", {"endpoint_id": "ep-1"})

    await connectors.internal_refresh(_FakeRequest("127.0.0.1"), db=db)

    assert connector_config_cache.get_config("runpod")["endpoint_id"] == "ep-1"
