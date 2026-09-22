"""Unit tests for app/routers/engine_admin.py — the "Active engine" GET/PUT + internal-refresh, called directly
(see tests/test_ollama_admin_router.py's own docstring for why this project's router tests don't use a
TestClient)."""

import pytest
from fastapi import HTTPException

from app.models import SYSTEM_OWNER_ID, AppSetting
from app.routers import engine_admin
from app.schemas import ActiveEngineConfig
from app.services import engine_service, engine_switch_service, server_pool_broadcast, settings_service


async def _async_return(value):
    return value


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host):
        self.client = _FakeClient(host) if host else None


@pytest.fixture(autouse=True)
def reset_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_get_active_engine_defaults_when_never_configured(db):
    result = await engine_admin.get_active_engine(db=db, _admin=None)
    assert result.active_engine == engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_set_active_engine_persists_and_updates_the_cache(db, monkeypatch):
    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return([]))

    result = await engine_admin.set_active_engine(ActiveEngineConfig(active_engine="ollama"), db=db, _admin=None)

    assert result.active_engine == "ollama"
    assert engine_service.current_engine() == "ollama"
    assert await engine_service.get_active_engine(db) == "ollama"


@pytest.mark.asyncio
async def test_set_active_engine_clears_a_stale_default_when_the_engine_actually_changes(db, monkeypatch):
    """The real bug this covers: switching engines used to leave the previous engine's default model tag
    behind, pointed at something the newly active engine may never have heard of (see
    engine_switch_service.clear_stale_default_models's own docstring)."""

    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    monkeypatch.setattr(engine_switch_service, "list_models", lambda: _async_return([]))
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "stale-tag")

    await engine_admin.set_active_engine(ActiveEngineConfig(active_engine="ollama"), db=db, _admin=None)

    assert await db.get(AppSetting, (SYSTEM_OWNER_ID, settings_service.DEFAULT_MODEL_KEY)) is None


@pytest.mark.asyncio
async def test_set_active_engine_leaves_defaults_alone_on_a_no_op_save(db, monkeypatch):
    """Re-picking the already-active engine (e.g. just pressing Save again) must never touch an already-valid
    default — only an actual change in which engine is active should trigger the stale-default sweep."""

    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    def fail_if_called():
        raise AssertionError("clear_stale_default_models must not run on a no-op engine save")

    monkeypatch.setattr(engine_switch_service, "list_models", fail_if_called)
    await settings_service.set_default_model(db, SYSTEM_OWNER_ID, "still-there-tag")

    await engine_admin.set_active_engine(
        ActiveEngineConfig(active_engine=engine_service.DEFAULT_ENGINE), db=db, _admin=None
    )

    row = await db.get(AppSetting, (SYSTEM_OWNER_ID, settings_service.DEFAULT_MODEL_KEY))
    assert row is not None
    assert row.value["model"] == "still-there-tag"


@pytest.mark.asyncio
async def test_set_active_engine_broadcasts_to_other_instances(db, monkeypatch):
    calls = []

    async def fake_broadcast(path):
        calls.append(path)
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    await engine_admin.set_active_engine(ActiveEngineConfig(active_engine="matricxon"), db=db, _admin=None)

    assert calls == ["/api/settings/engine/internal-refresh"]


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db):
    with pytest.raises(HTTPException) as exc_info:
        await engine_admin.internal_refresh(_FakeRequest("203.0.113.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_reloads_the_cache_from_the_shared_db(db):
    await engine_service.set_active_engine(db, "matricxon")
    engine_service._cached_engine = "ollama"  # simulate this sibling not having refreshed yet

    await engine_admin.internal_refresh(_FakeRequest("127.0.0.1"), db=db)

    assert engine_service.current_engine() == "matricxon"
