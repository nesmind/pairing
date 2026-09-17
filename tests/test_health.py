"""Unit tests for GET /health (app/routers/health.py) — how it combines
the Ollama-pool probe and a database check into one overall
status/status-code. Per-host Ollama probing itself is tested separately
in tests/test_ollama_pool.py's check_hosts tests; this file stubs that
out and only covers the route's own combining logic."""

import json

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.routers.health import health
from app.services import ollama_pool


def _body(response) -> dict:
    return json.loads(response.body)


@pytest.mark.asyncio
async def test_health_ok_when_ollama_and_database_are_both_reachable(db, monkeypatch):
    async def fake_check_hosts():
        return {"http://h1:11434": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 200
    body = _body(response)
    assert body == {"status": "ok", "database": True, "ollama_hosts": {"http://h1:11434": True}}


@pytest.mark.asyncio
async def test_health_ok_when_at_least_one_ollama_host_is_up(db, monkeypatch):
    async def fake_check_hosts():
        return {"http://h1:11434": False, "http://h2:11434": True}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 200
    assert _body(response)["status"] == "ok"


@pytest.mark.asyncio
async def test_health_unhealthy_when_every_ollama_host_is_down(db, monkeypatch):
    """The scenario this endpoint exists for: a machine's only local
    Ollama died, so every instance running there should report
    unhealthy and the proxy should stop routing to that machine."""

    async def fake_check_hosts():
        return {"http://h1:11434": False, "http://h2:11434": False}

    monkeypatch.setattr(ollama_pool, "check_hosts", fake_check_hosts)

    response = await health(db=db)

    assert response.status_code == 503
    body = _body(response)
    assert body["status"] == "unhealthy"
    assert body["database"] is True
    assert all(up is False for up in body["ollama_hosts"].values())


@pytest.mark.asyncio
async def test_health_unhealthy_when_database_is_unreachable(db, monkeypatch):
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
