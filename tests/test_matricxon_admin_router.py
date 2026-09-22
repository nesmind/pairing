"""Unit tests for app/routers/matricxon_admin.py — mirrors tests/test_ollama_admin_router.py's own coverage and
calling convention (see that file's docstring for why these call the router functions directly rather than
through a TestClient). The one real difference from Ollama's version: _save_and_broadcast only refreshes/
broadcasts the live pool when Matricxon is the *active* engine (see app.services.engine_service) — Ollama's
router has no sibling engine to defer to, so it always refreshes unconditionally."""

import pytest
from fastapi import HTTPException

from app.routers import matricxon_admin
from app.schemas import MatricxonServerConfig, MatricxonServerStatus
from app.services import engine_service, github_releases, matricxon_pool, matricxon_process, server_pool_broadcast


class _FakeClient:
    def __init__(self, host):
        self.host = host


class _FakeRequest:
    def __init__(self, host):
        self.client = _FakeClient(host) if host else None


@pytest.fixture(autouse=True)
def reset_cache():
    """Some tests below flip the active engine to "matricxon" (see
    test_update_config_refreshes_the_pool_when_matricxon_is_the_active_engine) — engine_service._cached_engine is
    module-level global state that would otherwise leak into every other test in the suite, same reasoning as
    tests/test_engine_service.py's own identical fixture."""
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.mark.asyncio
async def test_get_config_reads_through_settings_service(db):
    config = await matricxon_admin.get_config(db=db, _admin=None)
    assert config.mode == "local"


@pytest.mark.asyncio
async def test_check_host_reports_the_pinged_hosts_reachability(monkeypatch):
    async def fake_check_host(_host):
        return True

    monkeypatch.setattr(matricxon_pool, "check_host", fake_check_host)
    result = await matricxon_admin.check_host(host="http://10.0.0.5:8420", _admin=None)
    assert result.host == "http://10.0.0.5:8420"
    assert result.healthy is True


@pytest.mark.asyncio
async def test_update_config_persists_without_touching_the_process(db, monkeypatch):
    monkeypatch.setattr(matricxon_process, "start", lambda *_a, **_kw: pytest.fail("must not start/stop"))
    monkeypatch.setattr(matricxon_process, "stop", lambda *_a, **_kw: pytest.fail("must not start/stop"))

    body = MatricxonServerConfig(max_loaded_models=4)
    result = await matricxon_admin.update_config(body, db=db, _admin=None)

    assert result.max_loaded_models == 4
    assert (await matricxon_admin.get_config(db=db, _admin=None)).max_loaded_models == 4


@pytest.mark.asyncio
async def test_update_config_does_not_refresh_the_pool_when_matricxon_is_not_the_active_engine(db, monkeypatch):
    # Matricxon is the default active engine (see app.services.engine_service.DEFAULT_ENGINE) — switch to Ollama
    # first so this actually exercises "saving Matricxon's own config while it isn't the live engine must not
    # touch matricxon_pool or broadcast to siblings."
    await engine_service.set_active_engine(db, "ollama")
    monkeypatch.setattr(matricxon_pool, "refresh_from_config", lambda _c: pytest.fail("must not refresh"))
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", lambda _p: pytest.fail("must not broadcast"))

    result = await matricxon_admin.update_config(MatricxonServerConfig(max_loaded_models=2), db=db, _admin=None)
    assert result.max_loaded_models == 2


@pytest.mark.asyncio
async def test_update_config_refreshes_the_pool_when_matricxon_is_the_active_engine(db, monkeypatch):
    await engine_service.set_active_engine(db, "matricxon")
    refreshed = {}
    monkeypatch.setattr(matricxon_pool, "refresh_from_config", lambda config: refreshed.setdefault("config", config))

    async def fake_broadcast(_path):
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)

    await matricxon_admin.update_config(MatricxonServerConfig(max_loaded_models=6), db=db, _admin=None)

    assert refreshed["config"].max_loaded_models == 6


@pytest.mark.asyncio
async def test_apply_config_maps_value_error_to_400(db, monkeypatch):
    async def fake_apply(_config, proxy_url=None):
        raise ValueError("Matricxon can only be managed from the primary instance.")

    monkeypatch.setattr(matricxon_process, "apply_local_config", fake_apply)

    with pytest.raises(HTTPException) as exc_info:
        await matricxon_admin.apply_config(MatricxonServerConfig(), db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_apply_config_maps_runtime_error_to_502(db, monkeypatch):
    async def fake_apply(_config, proxy_url=None):
        raise RuntimeError("Matricxon did not come back up in time.")

    monkeypatch.setattr(matricxon_process, "apply_local_config", fake_apply)

    with pytest.raises(HTTPException) as exc_info:
        await matricxon_admin.apply_config(MatricxonServerConfig(), db=db, _admin=None)
    assert exc_info.value.status_code == 502


@pytest.mark.asyncio
async def test_start_reads_the_saved_config_and_passes_it_through(db, monkeypatch):
    await matricxon_admin.update_config(MatricxonServerConfig(max_loaded_models=8), db=db, _admin=None)
    captured = {}

    async def fake_start(config, proxy_url=None):
        captured["config"] = config
        return MatricxonServerStatus(running=True, installed=True)

    monkeypatch.setattr(matricxon_process, "start", fake_start)

    await matricxon_admin.start(db=db, _admin=None)

    assert captured["config"].max_loaded_models == 8


@pytest.mark.asyncio
async def test_start_maps_value_error_to_400(db, monkeypatch):
    async def fake_start(_config, proxy_url=None):
        raise ValueError("Matricxon can only be managed from the primary instance.")

    monkeypatch.setattr(matricxon_process, "start", fake_start)

    with pytest.raises(HTTPException) as exc_info:
        await matricxon_admin.start(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_stop_maps_value_error_to_400(db, monkeypatch):
    async def fake_stop(_project_dir=None):
        raise ValueError("Matricxon can only be managed from the primary instance.")

    monkeypatch.setattr(matricxon_process, "stop", fake_stop)

    with pytest.raises(HTTPException) as exc_info:
        await matricxon_admin.stop(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_status_reads_through_matricxon_process(db, monkeypatch):
    monkeypatch.setattr(matricxon_process, "_find_project_dir", lambda _project_dir=None: None)
    status = await matricxon_admin.get_status(db=db, _admin=None)
    assert status.running is False


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db, monkeypatch):
    def fail_if_called(_config):
        raise AssertionError("must not refresh for a non-loopback caller")

    monkeypatch.setattr(matricxon_pool, "refresh_from_config", fail_if_called)

    with pytest.raises(HTTPException) as exc_info:
        await matricxon_admin.internal_refresh(_FakeRequest("203.0.113.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_install_defaults_reflects_the_pinned_config(admin_user):
    from app.config import MATRICXON_GITHUB_REPO, MATRICXON_PINNED_VERSION

    result = await matricxon_admin.install_defaults(_admin=admin_user)
    assert result.repo == MATRICXON_GITHUB_REPO
    assert result.version == MATRICXON_PINNED_VERSION


@pytest.mark.asyncio
async def test_available_versions_falls_back_to_the_pinned_repo_when_none_is_given(admin_user, monkeypatch):
    from app.config import MATRICXON_GITHUB_REPO

    seen = {}

    async def fake_list_tags(repo):
        seen["repo"] = repo
        return ["v0.2", "v0.1"]

    monkeypatch.setattr(github_releases, "list_tags", fake_list_tags)

    result = await matricxon_admin.available_versions(repo=None, _admin=admin_user)

    assert seen["repo"] == MATRICXON_GITHUB_REPO
    assert result.versions == ["v0.2", "v0.1"]


@pytest.mark.asyncio
async def test_available_versions_uses_a_given_repo_override(admin_user, monkeypatch):
    seen = {}

    async def fake_list_tags(repo):
        seen["repo"] = repo
        return []

    monkeypatch.setattr(github_releases, "list_tags", fake_list_tags)

    await matricxon_admin.available_versions(repo="me/fork", _admin=admin_user)

    assert seen["repo"] == "me/fork"


@pytest.mark.asyncio
async def test_auto_detected_path_reads_through_matricxon_process(monkeypatch):
    monkeypatch.setattr(matricxon_process, "auto_detect_project_dir", lambda: "/home/user/matricxon")
    monkeypatch.setattr(matricxon_process, "auto_detect_models_dir", lambda: "/home/user/matricxon/data/models")
    result = await matricxon_admin.auto_detected_path(_admin=None)
    assert result.path == "/home/user/matricxon"
    assert result.models_path == "/home/user/matricxon/data/models"


@pytest.mark.asyncio
async def test_auto_detected_path_reports_none_when_nothing_found(monkeypatch):
    monkeypatch.setattr(matricxon_process, "auto_detect_project_dir", lambda: None)
    monkeypatch.setattr(matricxon_process, "auto_detect_models_dir", lambda: None)
    result = await matricxon_admin.auto_detected_path(_admin=None)
    assert result.path is None
    assert result.models_path is None
