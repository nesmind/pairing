"""Unit tests for app/services/server_pool_broadcast.py — mirrors
tests/test_instance_db_broadcast.py's own conventions exactly (same
fix shape: propagate a live config change to every other local
instance's own in-memory state). Every test fakes instance_process.
read_tracking/is_alive rather than spawning real processes."""

import httpx
import pytest

from app.services import instance_process, server_pool_broadcast


def test_other_live_ports_excludes_self_and_dead_or_untracked_indices(monkeypatch):
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {1: {"pid": 101}, 2: {"pid": 102}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: pid == 101)
    monkeypatch.setattr(server_pool_broadcast, "INSTANCE_INDEX", 0)

    ports = server_pool_broadcast._other_live_ports()

    assert sorted(ports) == [instance_process.port_for_index(1)]


def test_other_live_ports_excludes_self_when_self_is_a_sibling(monkeypatch):
    monkeypatch.setattr(instance_process, "read_tracking", lambda: {1: {"pid": 101}})
    monkeypatch.setattr(instance_process, "is_alive", lambda pid: True)
    monkeypatch.setattr(server_pool_broadcast, "INSTANCE_INDEX", 1)

    ports = server_pool_broadcast._other_live_ports()

    assert sorted(ports) == [instance_process.port_for_index(0)]


@pytest.mark.asyncio
async def test_broadcast_refresh_posts_to_every_other_live_instance(monkeypatch):
    calls = []

    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url):
            calls.append(url)
            return _FakeResponse()

    monkeypatch.setattr(server_pool_broadcast, "_other_live_ports", lambda: [8001, 8002])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient())

    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/ollama/internal-refresh")

    assert failures == []
    assert calls == [
        "http://127.0.0.1:8001/api/settings/ollama/internal-refresh",
        "http://127.0.0.1:8002/api/settings/ollama/internal-refresh",
    ]


@pytest.mark.asyncio
async def test_broadcast_refresh_is_best_effort_and_reports_failures(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, url):
            if "8001" in url:
                raise httpx.ConnectError("Connection refused", request=httpx.Request("POST", url))
            return _FakeResponse()

    monkeypatch.setattr(server_pool_broadcast, "_other_live_ports", lambda: [8001, 8002])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient())

    failures = await server_pool_broadcast.broadcast_refresh("/api/settings/comfyui/internal-refresh")

    assert len(failures) == 1
    assert "8001" in failures[0]
