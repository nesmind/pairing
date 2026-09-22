"""Unit tests for app/services/comfyui_pool.py — mirrors
tests/test_ollama_pool.py's shape exactly (same thin-wrapper contract
around app/services/server_pool.py's HostPool), just for ComfyUI's own
config schema and the call_with_failover shape it uses instead of
stream_with_failover."""

import pytest

from app.schemas import ComfyUIProcessConfig
from app.services import comfyui_pool


@pytest.fixture(autouse=True)
def isolated_pool(monkeypatch):
    monkeypatch.setattr(comfyui_pool, "COMFYUI_HOST", "http://local-comfyui:8188")
    comfyui_pool._pool.set_hosts(["http://local-comfyui:8188"])
    yield


def test_refresh_from_config_local_mode_resolves_to_the_local_host():
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="local", remote_hosts=["http://stale:1"]))
    assert comfyui_pool.get_effective_hosts() == ["http://local-comfyui:8188"]


def test_refresh_from_config_remote_mode_resolves_to_the_configured_hosts():
    remote = ["http://r1:8188", "http://r2:8188"]
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="remote", remote_hosts=remote))
    assert comfyui_pool.get_effective_hosts() == remote


def test_pick_host_delegates_to_the_live_pool():
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="local"))
    assert comfyui_pool.pick_host() == "http://local-comfyui:8188"


def test_mark_failure_and_recovered_delegate_to_the_live_pool():
    host = "http://local-comfyui:8188"
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="remote", remote_hosts=[host, "http://r2:1"]))
    comfyui_pool.mark_failure(host)
    assert comfyui_pool.pick_host() != host
    comfyui_pool.mark_recovered(host)
    assert comfyui_pool._pool._states[host].cooldown_until == 0.0


@pytest.mark.asyncio
async def test_call_with_failover_delegates_to_the_live_pool():
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="local"))

    async def make_call(host):
        return host

    result = await comfyui_pool.call_with_failover(make_call)
    assert result == "http://local-comfyui:8188"


@pytest.mark.asyncio
async def test_check_hosts_reports_reachability(monkeypatch):
    comfyui_pool.refresh_from_config(ComfyUIProcessConfig(mode="local"))

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
    result = await comfyui_pool.check_hosts()
    assert result == {"http://local-comfyui:8188": True}


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

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())
    assert await comfyui_pool.check_host("http://10.9.9.9:8188") is True
