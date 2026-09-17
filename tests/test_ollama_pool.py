"""Unit tests for app/services/ollama_pool.py — the thin Ollama-flavored
wrapper around app/services/server_pool.py's generic HostPool (see
tests/test_server_pool.py for the shared selection/cooldown/failover
logic itself, tested once there rather than duplicated here). This file
only covers what's actually Ollama-specific: local/remote resolution via
refresh_from_config, and that every wrapper function delegates to the
live pool instance."""

import pytest

from app.schemas import OllamaServerConfig
from app.services import ollama_pool


@pytest.fixture(autouse=True)
def isolated_pool(monkeypatch):
    """Every test gets its own fresh pool state, isolated from
    ollama_pool.LOCAL_OLLAMA_HOST's real value and from any other
    test's leftover host list/cooldowns."""
    monkeypatch.setattr(ollama_pool, "LOCAL_OLLAMA_HOST", "http://local-ollama:11434")
    ollama_pool._pool.set_hosts(["http://local-ollama:11434"])
    yield


def test_refresh_from_config_local_mode_resolves_to_the_local_host():
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="local", remote_hosts=["http://stale:1"]))
    assert ollama_pool.get_effective_hosts() == ["http://local-ollama:11434"]


def test_refresh_from_config_remote_mode_resolves_to_the_configured_hosts():
    remote = ["http://r1:11434", "http://r2:11434"]
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="remote", remote_hosts=remote))
    assert ollama_pool.get_effective_hosts() == remote


def test_refresh_from_config_remote_mode_with_no_hosts_falls_back_to_local():
    # OllamaServerConfig's own validator normalizes "remote" with an
    # empty host list back to "local" (see app.schemas.ollama_server_config
    # — a real state the settings UI could otherwise leave an admin
    # stuck in), so this never reaches the pool as remote-with-no-hosts.
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="remote", remote_hosts=[]))
    assert ollama_pool.get_effective_hosts() == ["http://local-ollama:11434"]


def test_pick_host_delegates_to_the_live_pool():
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="local"))
    assert ollama_pool.pick_host() == "http://local-ollama:11434"


def test_mark_failure_and_recovered_delegate_to_the_live_pool():
    host = "http://local-ollama:11434"
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="remote", remote_hosts=[host, "http://r2:1"]))
    ollama_pool.mark_failure(host)
    assert ollama_pool.pick_host() != host
    ollama_pool.mark_recovered(host)
    assert ollama_pool._pool._states[host].cooldown_until == 0.0


@pytest.mark.asyncio
async def test_track_request_delegates_to_the_live_pool():
    host = "http://local-ollama:11434"
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="local"))
    async with ollama_pool.track_request(host):
        assert ollama_pool._pool._states[host].in_flight == 1
    assert ollama_pool._pool._states[host].in_flight == 0


@pytest.mark.asyncio
async def test_stream_with_failover_delegates_to_the_live_pool():
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="local"))

    async def make_stream(_host):
        yield "ok"

    result = [c async for c in ollama_pool.stream_with_failover(make_stream)]
    assert result == ["ok"]


@pytest.mark.asyncio
async def test_check_hosts_reports_reachability(monkeypatch):
    ollama_pool.refresh_from_config(OllamaServerConfig(mode="local"))

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url):
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())
    result = await ollama_pool.check_hosts()
    assert result == {"http://local-ollama:11434": True}


@pytest.mark.asyncio
async def test_check_host_pings_an_arbitrary_host_not_necessarily_saved(monkeypatch):
    # Deliberately never refresh_from_config'd — a check-host call has to
    # work for a candidate URL an admin just typed, before it's ever
    # part of the live pool.
    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url):
            return _FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())
    assert await ollama_pool.check_host("http://10.9.9.9:11434") is True
