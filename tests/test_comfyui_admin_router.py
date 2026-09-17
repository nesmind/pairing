"""Unit tests for app/routers/comfyui_admin.py — called directly, not
through a TestClient/ASGI app (see tests/test_instance_proxy_http.py's
own docstring on why this project doesn't use that pattern). The
underlying start/stop logic is already covered by
tests/test_comfyui_service.py; this file only covers the router's own
job — wiring config get/set through settings_service, and mapping
comfyui_service's ValueError onto a 400."""

import pytest
from fastapi import HTTPException

from app.config import COMFYUI_GITHUB_REPO, COMFYUI_PINNED_VERSION
from app.routers import comfyui_admin
from app.schemas import ComfyUIProcessConfig, HttpProxyConfig
from app.services import (
    comfyui_installer,
    comfyui_pool,
    comfyui_service,
    http_proxy_service,
    server_pool_broadcast,
    settings_service,
)


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host):
        self.client = _FakeClient(host) if host else None


@pytest.mark.asyncio
async def test_get_config_reads_through_settings_service(db):
    config = await comfyui_admin.get_config(db=db, _admin=None)
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_check_host_reports_the_pinged_hosts_reachability(monkeypatch):
    async def fake_check_host(_host):
        return True

    monkeypatch.setattr(comfyui_pool, "check_host", fake_check_host)
    result = await comfyui_admin.check_host(host="http://10.0.0.5:8188", _admin=None)
    assert result.host == "http://10.0.0.5:8188"
    assert result.healthy is True


@pytest.mark.asyncio
async def test_check_host_reports_an_unreachable_host(monkeypatch):
    async def fake_check_host(_host):
        return False

    monkeypatch.setattr(comfyui_pool, "check_host", fake_check_host)
    result = await comfyui_admin.check_host(host="http://10.0.0.9:8188", _admin=None)
    assert result.healthy is False


@pytest.mark.asyncio
async def test_update_config_persists_and_round_trips(db):
    body = ComfyUIProcessConfig(mode="remote", remote_hosts=["http://10.0.0.1:8188"])
    result = await comfyui_admin.update_config(body, db=db, _admin=None)

    assert result.remote_hosts == ["http://10.0.0.1:8188"]
    assert (await comfyui_admin.get_config(db=db, _admin=None)).mode == "remote"


@pytest.mark.asyncio
async def test_start_maps_value_error_to_400(db, monkeypatch):
    async def fake_start(_db):
        raise ValueError("ComfyUI can only be managed from the primary instance.")

    monkeypatch.setattr(comfyui_service, "start", fake_start)

    with pytest.raises(HTTPException) as exc_info:
        await comfyui_admin.start(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_maps_value_error_to_400(db, monkeypatch):
    async def fake_stop(_db):
        raise ValueError("ComfyUI can only be managed from the primary instance.")

    monkeypatch.setattr(comfyui_service, "stop", fake_stop)

    with pytest.raises(HTTPException) as exc_info:
        await comfyui_admin.stop(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_status_reads_through_comfyui_service(db):
    status = await comfyui_admin.get_status(db=db, _admin=None)
    assert status.running is False


@pytest.mark.asyncio
async def test_update_config_refreshes_this_instances_own_pool_immediately(db, monkeypatch):
    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    refreshed = {}
    monkeypatch.setattr(comfyui_pool, "refresh_from_config", lambda config: refreshed.setdefault("config", config))

    body = ComfyUIProcessConfig(mode="remote", remote_hosts=["http://x:1"])
    await comfyui_admin.update_config(body, db=db, _admin=None)

    assert refreshed["config"].remote_hosts == ["http://x:1"]


@pytest.mark.asyncio
async def test_update_config_broadcasts_to_other_instances(db, monkeypatch):
    monkeypatch.setattr(comfyui_pool, "refresh_from_config", lambda _config: None)
    calls = []

    async def fake_broadcast(path):
        calls.append(path)
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    await comfyui_admin.update_config(ComfyUIProcessConfig(), db=db, _admin=None)

    assert calls == ["/api/settings/comfyui/internal-refresh"]


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db, monkeypatch):
    def fail_if_called(_config):
        raise AssertionError("must not refresh for a non-loopback caller")

    monkeypatch.setattr(comfyui_pool, "refresh_from_config", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await comfyui_admin.internal_refresh(_FakeRequest("203.0.113.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_accepts_a_loopback_caller_and_reloads_the_pool(db, monkeypatch):
    await settings_service.set_comfyui_config(db, ComfyUIProcessConfig(mode="remote", remote_hosts=["http://r1:1"]))
    refreshed = {}
    monkeypatch.setattr(comfyui_pool, "refresh_from_config", lambda config: refreshed.setdefault("config", config))

    result = await comfyui_admin.internal_refresh(_FakeRequest("127.0.0.1"), db=db)

    assert result.ok is True
    assert refreshed["config"].remote_hosts == ["http://r1:1"]


@pytest.mark.asyncio
async def test_install_defaults_returns_the_pinned_repo_and_version():
    result = await comfyui_admin.install_defaults(_admin=None)
    assert result.repo == COMFYUI_GITHUB_REPO
    assert result.version == COMFYUI_PINNED_VERSION


@pytest.mark.asyncio
async def test_install_passes_the_saved_override_through_to_install_stream(db, monkeypatch):
    await comfyui_admin.update_config(
        ComfyUIProcessConfig(install_repo="me/comfy-fork", install_version="v1.0"), db=db, _admin=None
    )
    captured = {}

    async def fake_install_stream(repo, version, proxy_url=None):
        captured["repo"] = repo
        captured["version"] = version
        captured["proxy_url"] = proxy_url
        yield {"done": True}

    monkeypatch.setattr(comfyui_installer, "install_stream", fake_install_stream)

    response = await comfyui_admin.install(db=db, _admin=None)
    events = [chunk async for chunk in response.body_iterator]

    assert captured == {"repo": "me/comfy-fork", "version": "v1.0", "proxy_url": None}
    assert events == ['data: {"done": true}\n\n']


@pytest.mark.asyncio
async def test_install_passes_the_configured_proxy_through_to_install_stream(db, monkeypatch):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080))
    captured = {}

    async def fake_install_stream(repo, version, proxy_url=None):
        captured["proxy_url"] = proxy_url
        yield {"done": True}

    monkeypatch.setattr(comfyui_installer, "install_stream", fake_install_stream)

    response = await comfyui_admin.install(db=db, _admin=None)
    [chunk async for chunk in response.body_iterator]

    assert captured["proxy_url"] == "http://10.0.0.5:8080"
