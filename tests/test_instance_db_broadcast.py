"""Unit tests for app/services/instance_db_broadcast.py — the fix for a
real production incident: app.database.switch_database only ever rebinds
the *current process's* own live engine, so with instance_count > 1,
every other local instance silently kept running against the old
database until this existed. Every test fakes instance_process.
read_tracking/is_alive (same pattern tests/test_instance_pool.py already
uses) rather than spawning real processes."""

import httpx
import pytest

from app.services import instance_db_broadcast, instance_process


def test_other_live_ports_excludes_self_and_dead_or_untracked_indices(monkeypatch):
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {1: {"pid": 101}, 2: {"pid": 102}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: pid == 101)  # only index 1 is actually alive
    monkeypatch.setattr(instance_db_broadcast, "INSTANCE_INDEX", 0)  # this call originated on the primary

    ports = instance_db_broadcast._other_live_ports()

    assert sorted(ports) == [instance_process.port_for_index(1)]  # not self (0), not the dead index 2


def test_other_live_ports_excludes_self_when_self_is_a_sibling(monkeypatch):
    """The primary (index 0) is always a candidate even though it's
    never in the tracking file — confirmed here from a *sibling's* own
    point of view, where self-exclusion must drop 0, not a tracked
    index."""
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {1: {"pid": 101}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(instance_db_broadcast, "INSTANCE_INDEX", 1)  # this call originated on sibling 1

    ports = instance_db_broadcast._other_live_ports()

    assert sorted(ports) == [instance_process.port_for_index(0)]


@pytest.mark.asyncio
async def test_broadcast_database_switch_posts_to_every_other_live_instance(monkeypatch):
    calls = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, json):
            calls.append((url, json))
            return _FakeResponse()

    monkeypatch.setattr(instance_db_broadcast, "_other_live_ports", lambda: [8001, 8002])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient())

    failures = await instance_db_broadcast.broadcast_database_switch("sqlite:////new.db")

    assert failures == []
    assert calls == [
        ("http://127.0.0.1:8001/api/settings/database/internal-switch", {"database_url": "sqlite:////new.db"}),
        ("http://127.0.0.1:8002/api/settings/database/internal-switch", {"database_url": "sqlite:////new.db"}),
    ]


@pytest.mark.asyncio
async def test_broadcast_database_switch_is_best_effort_and_reports_failures(monkeypatch):
    """One unreachable/crashed sibling must not raise, and must not stop
    the *other* siblings from still being called — this is the whole
    point of the fix being best-effort rather than all-or-nothing."""

    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url, json):
            if "8001" in url:
                raise httpx.ConnectError("Connection refused", request=httpx.Request("POST", url))
            return _FakeResponse()

    monkeypatch.setattr(instance_db_broadcast, "_other_live_ports", lambda: [8001, 8002])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient())

    failures = await instance_db_broadcast.broadcast_database_switch("sqlite:////new.db")

    assert len(failures) == 1
    assert "8001" in failures[0]
