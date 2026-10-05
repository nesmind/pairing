"""Unit tests for app/routers/sdcpp_admin.py and image_engine_admin.py — called directly (see
tests/test_comfyui_admin_router.py's docstring on why); start/stop logic is covered by test_sdcpp_service.py."""

import pytest
from fastapi import HTTPException

from app.config import SDCPP_GITHUB_REPO, SDCPP_PINNED_VERSION
from app.routers import image_engine_admin, sdcpp_admin
from app.schemas import ImageEngineConfig, SdCppConfig
from app.services import github_releases, image_engine_service, sdcpp_pool, sdcpp_service, server_pool_broadcast


class _FakeRequest:
    def __init__(self, host):
        self.client = type("C", (), {"host": host})() if host else None


@pytest.mark.asyncio
async def test_config_round_trips_and_refreshes_pool_and_peers(db, monkeypatch):
    refreshed, broadcasts = {}, []

    async def fake_broadcast(path):
        broadcasts.append(path)
        return []

    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", fake_broadcast)
    monkeypatch.setattr(sdcpp_pool, "refresh_from_config", lambda config: refreshed.setdefault("c", config))

    body = SdCppConfig(mode="remote", remote_hosts=["http://10.0.0.1:8189"], model_path="/m.gguf")
    await sdcpp_admin.update_config(body, db=db, _admin=None)

    assert (await sdcpp_admin.get_config(db=db, _admin=None)).remote_hosts == ["http://10.0.0.1:8189"]
    assert refreshed["c"].mode == "remote"
    assert broadcasts == ["/api/settings/sdcpp/internal-refresh"]


def test_remote_mode_without_hosts_falls_back_to_local():
    assert SdCppConfig(mode="remote").mode == "local"


@pytest.mark.asyncio
async def test_internal_refresh_rejects_a_non_loopback_caller(db):
    with pytest.raises(HTTPException) as exc_info:
        await sdcpp_admin.internal_refresh(_FakeRequest("10.0.0.9"), db=db)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_internal_refresh_accepts_loopback(db, monkeypatch):
    called = []
    monkeypatch.setattr(sdcpp_pool, "refresh_from_config", called.append)
    assert (await sdcpp_admin.internal_refresh(_FakeRequest("127.0.0.1"), db=db)).ok is True
    assert len(called) == 1


@pytest.mark.asyncio
async def test_check_host_reports_reachability(monkeypatch):
    async def fake_check_host(_host):
        return True

    monkeypatch.setattr(sdcpp_pool, "check_host", fake_check_host)
    result = await sdcpp_admin.check_host(host="http://h:8189", _admin=None)
    assert result.host == "http://h:8189" and result.healthy is True


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["start", "stop"])
async def test_start_stop_map_value_error_to_400(db, monkeypatch, action):
    async def fake(_db):
        raise ValueError("nope")

    monkeypatch.setattr(sdcpp_service, action, fake)
    with pytest.raises(HTTPException) as exc_info:
        await getattr(sdcpp_admin, action)(db=db, _admin=None)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_status_and_install_defaults(db, monkeypatch):
    assert (await sdcpp_admin.get_status(db=db, _admin=None)).running is False
    defaults = await sdcpp_admin.install_defaults(_admin=None)
    assert (defaults.repo, defaults.version) == (SDCPP_GITHUB_REPO, SDCPP_PINNED_VERSION)

    seen = []

    async def fake_list_tags(repo):
        seen.append(repo)
        return ["master-1"]

    monkeypatch.setattr(github_releases, "list_tags", fake_list_tags)
    assert (await sdcpp_admin.available_versions(repo=None, _admin=None)).versions == ["master-1"]
    assert seen == [SDCPP_GITHUB_REPO]


@pytest.mark.asyncio
async def test_image_engine_defaults_to_sdcpp_and_refuses_the_disabled_comfyui(db):
    assert (await image_engine_admin.get_image_engine(db=db, _admin=None)).active_image_engine == "sdcpp"
    with pytest.raises(HTTPException) as exc_info:
        await image_engine_admin.set_image_engine(ImageEngineConfig(active_image_engine="comfyui"), db=db, _admin=None)
    assert exc_info.value.status_code == 400
    assert (await image_engine_admin.get_image_engine(db=db, _admin=None)).active_image_engine == "sdcpp"


@pytest.mark.asyncio
async def test_an_install_that_saved_comfyui_falls_back_to_sdcpp(db, monkeypatch):
    from app.services import image_engine_service

    monkeypatch.setattr(image_engine_service, "ENABLED_IMAGE_ENGINES", ("sdcpp", "comfyui"))
    await image_engine_service.set_active_image_engine(db, "comfyui")
    assert await image_engine_service.get_active_image_engine(db) == "comfyui"
    monkeypatch.setattr(image_engine_service, "ENABLED_IMAGE_ENGINES", ("sdcpp",))  # as in production
    assert await image_engine_service.get_active_image_engine(db) == "sdcpp"


@pytest.mark.asyncio
async def test_install_query_build_overrides_the_saved_one(db, monkeypatch):
    from app.services import http_proxy_service, image_engine_service, sdcpp_installer

    await image_engine_service.set_sdcpp_config(db, SdCppConfig(build="cpu"))
    seen = {}

    async def fake_stream(_repo, _version, proxy_url=None, build="auto"):
        seen["build"] = build
        yield {"done": True}

    monkeypatch.setattr(sdcpp_installer, "install_stream", fake_stream)
    monkeypatch.setattr(http_proxy_service, "get_http_proxy_config", _fake_proxy)

    for requested, expected in (("vulkan", "vulkan"), (None, "cpu")):
        response = await sdcpp_admin.install(build=requested, db=db, _admin=None)
        [_ async for _ in response.body_iterator]
        assert seen["build"] == expected


async def _fake_proxy(_db):
    from app.schemas import HttpProxyConfig

    return HttpProxyConfig()


async def _no_peers(_path):
    return []


@pytest.mark.asyncio
async def test_apply_failure_is_a_400_and_saves_nothing(db, monkeypatch):
    async def fake_apply(_db, _config):
        raise ValueError("This model is not supported")

    monkeypatch.setattr(sdcpp_service, "apply", fake_apply)
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", _no_peers)
    before = await image_engine_service.get_sdcpp_config(db)
    with pytest.raises(HTTPException) as exc_info:
        await sdcpp_admin.apply_config(body=SdCppConfig(model_path="/m/tiny.gguf"), db=db, _admin=None)
    assert exc_info.value.status_code == 400
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == before.model_path


@pytest.mark.asyncio
async def test_apply_success_saves(db, monkeypatch):
    async def fake_apply(_db, _config):
        return await sdcpp_service.get_status(_db)

    monkeypatch.setattr(sdcpp_service, "apply", fake_apply)
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", _no_peers)
    await sdcpp_admin.apply_config(body=SdCppConfig(model_path="/m/ok.gguf"), db=db, _admin=None)
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == "/m/ok.gguf"


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["config", "apply"])
async def test_a_missing_images_folder_is_a_400_and_nothing_is_saved_or_restarted(db, monkeypatch, action):
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", _no_peers)

    async def must_not_run(*_a):
        raise AssertionError("the server must not be touched")

    monkeypatch.setattr(sdcpp_service, "apply", must_not_run)
    before = await image_engine_service.get_sdcpp_config(db)
    body = SdCppConfig(images_path="/definitely/not/here")
    call = sdcpp_admin.update_config if action == "config" else sdcpp_admin.apply_config
    with pytest.raises(HTTPException) as exc_info:
        await call(body=body, db=db, _admin=None)
    assert exc_info.value.status_code == 400 and "does not exist" in exc_info.value.detail
    assert await image_engine_service.get_sdcpp_config(db) == before


@pytest.mark.asyncio
async def test_an_existing_images_folder_is_saved(db, monkeypatch, tmp_path):
    monkeypatch.setattr(server_pool_broadcast, "broadcast_refresh", _no_peers)
    await sdcpp_admin.update_config(body=SdCppConfig(images_path=str(tmp_path)), db=db, _admin=None)
    assert (await image_engine_service.get_sdcpp_config(db)).images_path == str(tmp_path)
