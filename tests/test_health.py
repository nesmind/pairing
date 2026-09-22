"""Unit tests for GET /health (app/routers/health.py) — how it combines the active engine's own
pool probe and a database check into one overall status/status-code. Per-host probing itself is
tested separately in tests/test_ollama_pool.py/test_matricxon_pool.py's check_hosts tests; this
file stubs that out and only covers the route's own combining logic, for both engines."""

import json

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.routers.health import health
from app.services import engine_service, matricxon_pool, ollama_pool


def _body(response) -> dict:
    return json.loads(response.body)


@pytest.fixture(autouse=True)
def reset_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_health_ok_when_the_active_engine_and_database_are_both_reachable(db, monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_check_hosts():
        return {"http://h1:11434": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 200
    body = _body(response)
    assert body == {
        "status": "ok",
        "database": True,
        "active_engine": "ollama",
        "engine_hosts": {"http://h1:11434": True},
    }


@pytest.mark.asyncio
async def test_health_checks_matricxon_when_it_is_the_active_engine(db, monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def fake_ollama_check_hosts():
        pytest.fail("must not probe Ollama while Matricxon is the active engine")

    async def fake_matricxon_check_hosts():
        return {"http://m1:8420": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_ollama_check_hosts)
    monkeypatch.setattr(matricxon_pool, "check_hosts", fake_matricxon_check_hosts)

    response = await health(db=db)

    assert response.status_code == 200
    body = _body(response)
    assert body["active_engine"] == "matricxon"
    assert body["engine_hosts"] == {"http://m1:8420": True}


@pytest.mark.asyncio
async def test_health_ok_when_at_least_one_engine_host_is_up(db, monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_check_hosts():
        return {"http://h1:11434": False, "http://h2:11434": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 200
    assert _body(response)["status"] == "ok"


@pytest.mark.asyncio
async def test_health_unhealthy_when_every_engine_host_is_down(db, monkeypatch):
    """The scenario this endpoint exists for: a machine's only local engine died, so every instance
    running there should report unhealthy and the proxy should stop routing to that machine."""
    engine_service._cached_engine = "ollama"

    async def fake_check_hosts():
        return {"http://h1:11434": False, "http://h2:11434": False}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 503
    body = _body(response)
    assert body["status"] == "unhealthy"
    assert body["database"] is True
    assert all(up is False for up in body["engine_hosts"].values())


@pytest.mark.asyncio
async def test_health_unhealthy_when_database_is_unreachable(db, monkeypatch):
    engine_service._cached_engine = "ollama"

    async def fake_check_hosts():
        return {"http://h1:11434": True}

    async def fake_execute(*_args, **_kwargs):
        raise SQLAlchemyError("database is down")

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)
    monkeypatch.setattr(db, "execute", fake_execute)

    response = await health(db=db)

    assert response.status_code == 503
    body = _body(response)
    assert body["status"] == "unhealthy"
    assert body["database"] is False
