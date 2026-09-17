"""Unit tests for app/routers/db_admin.py's internal_switch_database —
the loopback-only endpoint app.services.instance_db_broadcast calls on
every other local instance after a live database switch. Never reached
through a real ASGI app in this project's test suite (see
tests/test_instance_proxy_http.py's own docstring on why: no
TestClient/ASGI-integration pattern here) — called directly with a
lightweight stand-in for the one attribute (`request.client.host`) the
endpoint actually reads."""

import pytest
from fastapi import HTTPException

from app import database
from app.routers import db_admin
from app.schemas import InternalSwitchDatabaseRequest


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host):
        self.client = _FakeClient(host) if host else None


@pytest.mark.asyncio
async def test_internal_switch_database_rejects_a_non_loopback_caller(monkeypatch):
    async def fail_if_called(_url):
        raise AssertionError("switch_database must not run for a non-loopback caller")

    monkeypatch.setattr(database, "switch_database", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await db_admin.internal_switch_database(
            InternalSwitchDatabaseRequest(database_url="sqlite:////evil.db"), _FakeRequest("203.0.113.9")
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_switch_database_rejects_a_missing_client(monkeypatch):
    """request.client can be None for some ASGI transports — must fail
    closed, not treat "unknown" as trusted."""

    async def fail_if_called(_url):
        raise AssertionError("switch_database must not run when request.client is missing")

    monkeypatch.setattr(database, "switch_database", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await db_admin.internal_switch_database(
            InternalSwitchDatabaseRequest(database_url="sqlite:////evil.db"), _FakeRequest(None)
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_switch_database_accepts_a_loopback_caller(monkeypatch):
    switched_to = []

    async def fake_switch_database(url):
        switched_to.append(url)

    monkeypatch.setattr(database, "switch_database", fake_switch_database)

    result = await db_admin.internal_switch_database(
        InternalSwitchDatabaseRequest(database_url="sqlite:////new.db"), _FakeRequest("127.0.0.1")
    )

    assert result.ok is True
    assert switched_to == ["sqlite:////new.db"]
