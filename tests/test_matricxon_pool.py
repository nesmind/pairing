"""Unit tests for app/services/matricxon_pool.py — mirrors tests/test_ollama_pool.py's own coverage exactly
(local/remote resolution, wrapper delegation to the live HostPool), just against the independent Matricxon
pool instance — see that file's own docstring for the shared server_pool.HostPool logic tested once there."""

import httpx
import pytest

from app.schemas import MatricxonServerConfig
from app.services import matricxon_pool


@pytest.fixture(autouse=True)
def isolated_pool(monkeypatch):
    # matricxon_pool._pool is a module-level singleton — set_hosts() calls made during a test (directly, or via
    # refresh_from_config) aren't undone by monkeypatch's own auto-revert, so without restoring it explicitly on
    # teardown a fake host from this file would otherwise leak into every later test in the suite (confirmed
    # live: it did, breaking an unrelated test elsewhere that happened to run afterward and hit this pool for
    # real).
    real_hosts = matricxon_pool.get_effective_hosts()
    monkeypatch.setattr(matricxon_pool, "LOCAL_MATRICXON_HOST", "http://local-matricxon:8420")
    matricxon_pool._pool.set_hosts(["http://local-matricxon:8420"])
    yield
    matricxon_pool._pool.set_hosts(real_hosts)


def test_refresh_from_config_local_mode_resolves_to_the_local_host():
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="local", remote_hosts=["http://stale:1"]))
    assert matricxon_pool.get_effective_hosts() == ["http://local-matricxon:8420"]


def test_refresh_from_config_remote_mode_resolves_to_the_configured_hosts():
    remote = ["http://r1:8420", "http://r2:8420"]
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="remote", remote_hosts=remote))
    assert matricxon_pool.get_effective_hosts() == remote


def test_refresh_from_config_remote_mode_with_no_hosts_falls_back_to_local():
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="remote", remote_hosts=[]))
    assert matricxon_pool.get_effective_hosts() == ["http://local-matricxon:8420"]


def test_pick_host_delegates_to_the_live_pool():
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="local"))
    assert matricxon_pool.pick_host() == "http://local-matricxon:8420"


def test_mark_failure_and_recovered_delegate_to_the_live_pool():
    host = "http://local-matricxon:8420"
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="remote", remote_hosts=[host, "http://r2:1"]))
    matricxon_pool.mark_failure(host)
    assert matricxon_pool.pick_host() != host
    matricxon_pool.mark_recovered(host)
    assert matricxon_pool._pool._states[host].cooldown_until == 0.0


@pytest.mark.asyncio
async def test_track_request_delegates_to_the_live_pool():
    host = "http://local-matricxon:8420"
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="local"))
    async with matricxon_pool.track_request(host):
        assert matricxon_pool._pool._states[host].in_flight == 1
    assert matricxon_pool._pool._states[host].in_flight == 0


@pytest.mark.asyncio
async def test_stream_with_failover_delegates_to_the_live_pool():
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="local"))

    async def make_stream(_host):
        yield "ok"

    result = [c async for c in matricxon_pool.stream_with_failover(make_stream)]
    assert result == ["ok"]


@pytest.mark.asyncio
async def test_check_hosts_reports_reachability(monkeypatch):
    matricxon_pool.refresh_from_config(MatricxonServerConfig(mode="local"))

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())
    result = await matricxon_pool.check_hosts()
    assert result == {"http://local-matricxon:8420": True}


@pytest.mark.asyncio
async def test_check_host_pings_an_arbitrary_host_not_necessarily_saved(monkeypatch):
    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, _url):
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())
    assert await matricxon_pool.check_host("http://10.9.9.9:8420") is True
