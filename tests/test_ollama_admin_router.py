"""Unit tests for app/routers/ollama_admin.py — called directly, not
through a TestClient/ASGI app (see tests/test_instance_proxy_http.py's
own docstring on why this project doesn't use that pattern). The
underlying start/stop/apply_local_config logic is already covered by
tests/test_ollama_process.py; this file only covers the router's own
job — wiring config get/set through settings_service, and mapping
ollama_process's ValueError/RuntimeError onto the right HTTP status."""

import pytest
from fastapi import HTTPException

from app.config import OLLAMA_GITHUB_REPO, OLLAMA_PINNED_VERSION
from app.routers import ollama_admin
from app.schemas import HttpProxyConfig, OllamaServerConfig, OllamaServerStatus
from app.services import (
    github_releases,
    http_proxy_service,
    ollama_installer,
    ollama_pool,
    ollama_process,
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
    config = await ollama_admin.get_config(db=db, _admin=None)
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_check_host_reports_the_pinged_hosts_reachability(monkeypatch):
    async def fake_check_host(_host):
        return True

    monkeypatch.setattr(ollama_pool, "check_host", fake_check_host)
    result = await ollama_admin.check_host(host="http://10.0.0.5:11434", _admin=None)
    assert result.host == "http://10.0.0.5:11434"
    assert result.healthy is True


@pytest.mark.asyncio
async def test_check_host_reports_an_unreachable_host(monkeypatch):
    async def fake_check_host(_host):
        return False

    monkeypatch.setattr(ollama_pool, "check_host", fake_check_host)
    result = await ollama_admin.check_host(host="http://10.0.0.9:11434", _admin=None)
    assert result.healthy is False


@pytest.mark.asyncio
async def test_update_config_persists_without_touching_the_process(db, monkeypatch):
    monkeypatch.setattr(ollama_process, "start", lambda *_a, **_kw: pytest.fail("must not start/stop"))
    monkeypatch.setattr(ollama_process, "stop", lambda *_a, **_kw: pytest.fail("must not start/stop"))

    body = OllamaServerConfig(num_parallel=4)
    result = await ollama_admin.update_config(body, db=db, _admin=None)

    assert result.num_parallel == 4
    assert (await ollama_admin.get_config(db=db, _admin=None)).num_parallel == 4


@pytest.mark.asyncio
async def test_apply_config_maps_value_error_to_400(db, monkeypatch):
    async def fake_apply(_config, proxy_url=None):
        raise ValueError("Ollama can only be managed from the primary instance.")

    monkeypatch.setattr(ollama_process, "apply_local_config", fake_apply)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.apply_config(OllamaServerConfig(), db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_config_maps_runtime_error_to_502(db, monkeypatch):
    async def fake_apply(_config, proxy_url=None):
        raise RuntimeError("Ollama did not come back up in time.")

    monkeypatch.setattr(ollama_process, "apply_local_config", fake_apply)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.apply_config(OllamaServerConfig(), db=db, _admin=None)
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_start_reads_the_saved_config_and_passes_it_through(db, monkeypatch):
    await ollama_admin.update_config(OllamaServerConfig(num_parallel=8), db=db, _admin=None)
    captured = {}

    async def fake_start(config, proxy_url=None):
        captured["config"] = config
        return await ollama_process.get_status()

    monkeypatch.setattr(ollama_process, "start", fake_start)
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)

    await ollama_admin.start(db=db, _admin=None)

    assert captured["config"].num_parallel == 8


@pytest.mark.asyncio
async def test_start_maps_value_error_to_400(db, monkeypatch):
    async def fake_start(_config, proxy_url=None):
        raise ValueError("Ollama can only be managed from the primary instance.")

    monkeypatch.setattr(ollama_process, "start", fake_start)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.start(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_maps_value_error_to_400(db, monkeypatch):
    async def fake_stop(_binary_path=None):
        raise ValueError("Ollama can only be managed from the primary instance.")

    monkeypatch.setattr(ollama_process, "stop", fake_stop)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.stop(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_status_reads_through_ollama_process(db, monkeypatch):
    monkeypatch.setattr(ollama_process, "_is_running", lambda _binary_path=None: False)
    status = await ollama_admin.get_status(db=db, _admin=None)
    assert status.running is False


@pytest.mark.asyncio
async def test_get_status_passes_the_saved_binary_path_override(db, monkeypatch):
    await ollama_admin.update_config(OllamaServerConfig(binary_path="/opt/ollama/ollama"), db=db, _admin=None)
    captured = {}

    async def fake_get_status(binary_path=None):
        captured["binary_path"] = binary_path
        return OllamaServerStatus(running=False, installed=False)

    monkeypatch.setattr(ollama_process, "get_status", fake_get_status)

    await ollama_admin.get_status(db=db, _admin=None)

    assert captured["binary_path"] == "/opt/ollama/ollama"


@pytest.mark.asyncio
async def test_update_config_refreshes_this_instances_own_pool_immediately(db, monkeypatch):
    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    refreshed = {}
    monkeypatch.setattr(ollama_pool, "refresh_from_config", lambda config: refreshed.setdefault("config", config))

    await ollama_admin.update_config(OllamaServerConfig(num_parallel=6), db=db, _admin=None)

    assert refreshed["config"].num_parallel == 6


@pytest.mark.asyncio
async def test_update_config_broadcasts_to_other_instances(db, monkeypatch):
    monkeypatch.setattr(ollama_pool, "refresh_from_config", lambda _config: None)
    calls = []

    async def fake_broadcast(path):
        calls.append(path)
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    await ollama_admin.update_config(OllamaServerConfig(), db=db, _admin=None)

    assert calls == ["/api/settings/ollama/internal-refresh"]


@pytest.mark.asyncio
async def test_update_config_does_not_raise_when_a_sibling_fails_to_refresh(db, monkeypatch):
    monkeypatch.setattr(ollama_pool, "refresh_from_config", lambda _config: None)

    async def fake_broadcast(_path):
        return ["instance on port 8001: down"]

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    result = await ollama_admin.update_config(OllamaServerConfig(num_parallel=2), db=db, _admin=None)
    assert result.num_parallel == 2


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db, monkeypatch):
    def fail_if_called(_config):
        raise AssertionError("must not refresh for a non-loopback caller")

    monkeypatch.setattr(ollama_pool, "refresh_from_config", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.internal_refresh(_FakeRequest("203.0.113.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_missing_client(db, monkeypatch):
    def fail_if_called(_config):
        raise AssertionError("must not refresh for a missing client")

    monkeypatch.setattr(ollama_pool, "refresh_from_config", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await ollama_admin.internal_refresh(_FakeRequest(None), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_accepts_a_loopback_caller_and_reloads_the_pool(db, monkeypatch):
    await settings_service.set_ollama_server_config(db, OllamaServerConfig(mode="remote", remote_hosts=["http://r1:1"]))
    refreshed = {}
    monkeypatch.setattr(ollama_pool, "refresh_from_config", lambda config: refreshed.setdefault("config", config))

    result = await ollama_admin.internal_refresh(_FakeRequest("127.0.0.1"), db=db)

    assert result.ok is True
    assert refreshed["config"].remote_hosts == ["http://r1:1"]


@pytest.mark.asyncio
async def test_install_defaults_returns_the_pinned_repo_and_version():
    result = await ollama_admin.install_defaults(_admin=None)
    assert result.repo == OLLAMA_GITHUB_REPO
    assert result.version == OLLAMA_PINNED_VERSION


@pytest.mark.asyncio
async def test_available_versions_falls_back_to_the_pinned_repo_when_none_is_given(monkeypatch):
    seen = {}

    async def fake_list_tags(repo):
        seen["repo"] = repo
        return ["v0.33.3", "v0.33.2"]

    monkeypatch.setattr(github_releases, "list_tags", fake_list_tags)

    result = await ollama_admin.available_versions(repo=None, _admin=None)

    assert seen["repo"] == OLLAMA_GITHUB_REPO
    assert result.versions == ["v0.33.3", "v0.33.2"]


@pytest.mark.asyncio
async def test_available_versions_uses_a_given_repo_override(monkeypatch):
    seen = {}

    async def fake_list_tags(repo):
        seen["repo"] = repo
        return []

    monkeypatch.setattr(github_releases, "list_tags", fake_list_tags)

    await ollama_admin.available_versions(repo="me/ollama-fork", _admin=None)

    assert seen["repo"] == "me/ollama-fork"


@pytest.mark.asyncio
async def test_install_passes_the_saved_override_through_to_install_stream(db, monkeypatch):
    await ollama_admin.update_config(
        OllamaServerConfig(install_repo="me/ollama-fork", install_version="v1.0"), db=db, _admin=None
    )
    captured = {}

    async def fake_install_stream(repo, version, proxy_url=None):
        captured["repo"] = repo
        captured["version"] = version
        captured["proxy_url"] = proxy_url
        yield {"done": True}

    monkeypatch.setattr(ollama_installer, "install_stream", fake_install_stream)

    response = await ollama_admin.install(db=db, _admin=None)
    events = [chunk async for chunk in response.body_iterator]

    assert captured == {"repo": "me/ollama-fork", "version": "v1.0", "proxy_url": None}
    assert events == ['data: {"done": true}\n\n']


@pytest.mark.asyncio
async def test_install_passes_none_through_when_no_override_saved(db, monkeypatch):
    captured = {}

    async def fake_install_stream(repo, version, proxy_url=None):
        captured["repo"] = repo
        captured["version"] = version
        captured["proxy_url"] = proxy_url
        yield {"done": True}

    monkeypatch.setattr(ollama_installer, "install_stream", fake_install_stream)

    response = await ollama_admin.install(db=db, _admin=None)
    [chunk async for chunk in response.body_iterator]

    assert captured == {"repo": None, "version": None, "proxy_url": None}


@pytest.mark.asyncio
async def test_install_passes_the_configured_proxy_through_to_install_stream(db, monkeypatch):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="10.0.0.5", port=8080))
    captured = {}

    async def fake_install_stream(repo, version, proxy_url=None):
        captured["proxy_url"] = proxy_url
        yield {"done": True}

    monkeypatch.setattr(ollama_installer, "install_stream", fake_install_stream)

    response = await ollama_admin.install(db=db, _admin=None)
    [chunk async for chunk in response.body_iterator]

    assert captured["proxy_url"] == "http://10.0.0.5:8080"


@pytest.mark.asyncio
async def test_start_passes_the_configured_proxy_through_to_ollama_process(db, monkeypatch):
    await http_proxy_service.set_http_proxy_config(db, HttpProxyConfig(enabled=True, host="1.2.3.4", port=3128))
    captured = {}

    async def fake_start(_config, proxy_url=None):
        captured["proxy_url"] = proxy_url
        return await ollama_process.get_status()

    monkeypatch.setattr(ollama_process, "start", fake_start)

    await ollama_admin.start(db=db, _admin=None)

    assert captured["proxy_url"] == "http://1.2.3.4:3128"


@pytest.mark.asyncio
async def test_auto_detected_path_reads_through_ollama_process(monkeypatch):
    monkeypatch.setattr(ollama_process, "auto_detect_binary", lambda: "/usr/bin/ollama")
    monkeypatch.setattr(ollama_process, "auto_detect_models_path", lambda: "/home/user/pAIring/models")
    result = await ollama_admin.auto_detected_path(_admin=None)
    assert result.path == "/usr/bin/ollama"
    assert result.models_path == "/home/user/pAIring/models"


@pytest.mark.asyncio
async def test_auto_detected_path_reports_none_when_nothing_found(monkeypatch):
    monkeypatch.setattr(ollama_process, "auto_detect_binary", lambda: None)
    result = await ollama_admin.auto_detected_path(_admin=None)
    assert result.path is None
