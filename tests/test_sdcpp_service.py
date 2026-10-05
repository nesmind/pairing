"""Unit tests for app/services/sdcpp_service.py + sdcpp_process.py — process primitives are monkeypatched."""

import pytest

from app.schemas import SdCppConfig
from app.services import image_engine_service, sdcpp_process, sdcpp_service


@pytest.fixture(autouse=True)
def isolated_tracking_file(tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_process, "_TRACKING_FILE", tmp_path / "sdcpp.json")


async def _configure(db, **overrides):
    config = {"binary_path": "/opt/sd/sd-server", "model_path": "/opt/sd/models/m.gguf", **overrides}
    await image_engine_service.set_sdcpp_config(db, SdCppConfig(**config))


def _fake_ping(result):
    async def ping():
        return result

    return ping


@pytest.mark.asyncio
async def test_start_rejects_when_binary_unset(db):
    with pytest.raises(ValueError, match="sd-server"):
        await sdcpp_service.start(db)


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    from app.services import image_model_service

    async def fake_resolve(_db):
        return tmp_path

    monkeypatch.setattr(image_model_service, "resolve_models_dir", fake_resolve)
    return tmp_path


@pytest.mark.asyncio
async def test_start_rejects_when_no_model_is_chosen_and_the_folder_is_empty(db, models_dir):
    await _configure(db, model_path=None)
    with pytest.raises(ValueError, match="Model dropdown"):
        await sdcpp_service.start(db)


@pytest.mark.asyncio
async def test_start_rejects_when_several_models_exist_and_none_is_chosen(db, models_dir):
    await _configure(db, model_path=None)
    (models_dir / "a.gguf").write_bytes(b"x")
    (models_dir / "b.gguf").write_bytes(b"x")
    with pytest.raises(ValueError, match="Model dropdown"):
        await sdcpp_service.start(db)


@pytest.mark.asyncio
async def test_start_uses_and_saves_the_only_installed_model(db, models_dir, monkeypatch):
    await _configure(db, model_path=None)
    (models_dir / "only.gguf").write_bytes(b"x")
    spawned = {}
    monkeypatch.setattr(sdcpp_process, "spawn", lambda binary, model, extra: spawned.setdefault("model", model) and 5)
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda *_a, **_kw: True)
    monkeypatch.setattr(sdcpp_service, "_ping_health", _fake_ping(True))

    await sdcpp_service.start(db)

    assert spawned["model"] == str(models_dir / "only.gguf")
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == str(models_dir / "only.gguf")


@pytest.mark.asyncio
async def test_start_rejects_on_a_non_primary_instance(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(sdcpp_service, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await sdcpp_service.start(db)


@pytest.mark.asyncio
async def test_start_spawns_and_writes_tracking(db, monkeypatch):
    await _configure(db, extra_args="--threads 2")
    spawned = {}

    def fake_spawn(binary, model, extra):
        spawned.update(binary=binary, model=model, extra=extra)
        return 4242

    monkeypatch.setattr(sdcpp_process, "spawn", fake_spawn)
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda *_a, **_kw: True)
    monkeypatch.setattr(sdcpp_service, "_ping_health", _fake_ping(True))

    status = await sdcpp_service.start(db)

    assert status.running is True and status.pid == 4242 and status.healthy is True
    assert spawned == {"binary": "/opt/sd/sd-server", "model": "/opt/sd/models/m.gguf", "extra": "--threads 2"}
    assert sdcpp_process.read_tracking() == {"pid": 4242}


@pytest.mark.asyncio
async def test_start_is_a_noop_when_already_running(db, monkeypatch):
    await _configure(db)
    sdcpp_process.write_tracking({"pid": 7})
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda *_a, **_kw: True)
    monkeypatch.setattr(sdcpp_service, "_ping_health", _fake_ping(True))

    def fail(*_a, **_kw):
        raise AssertionError("must not spawn a second server")

    monkeypatch.setattr(sdcpp_process, "spawn", fail)
    assert (await sdcpp_service.start(db)).pid == 7


@pytest.mark.asyncio
async def test_start_that_exits_immediately_leaves_it_not_running(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(sdcpp_process, "spawn", lambda *_a, **_kw: None)
    status = await sdcpp_service.start(db)
    assert status.running is False
    assert sdcpp_process.read_tracking() is None


@pytest.mark.asyncio
async def test_stop_terminates_and_clears_tracking(db, monkeypatch):
    await _configure(db)
    sdcpp_process.write_tracking({"pid": 99})
    terminated = []
    monkeypatch.setattr(sdcpp_process, "terminate", lambda pid, path: terminated.append((pid, path)))
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda *_a, **_kw: False)

    status = await sdcpp_service.stop(db)

    assert terminated == [(99, "/opt/sd/sd-server")]
    assert status.running is False
    assert sdcpp_process.read_tracking() is None


@pytest.mark.asyncio
async def test_stop_rejects_on_a_non_primary_instance(db, monkeypatch):
    monkeypatch.setattr(sdcpp_service, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await sdcpp_service.stop(db)


@pytest.mark.asyncio
async def test_status_reports_installed_when_a_binary_path_is_configured(db):
    await _configure(db)
    assert (await sdcpp_service.get_status(db)).installed is True


def test_build_argv_uses_the_configured_host_port_and_extra_args(monkeypatch):
    monkeypatch.setattr(sdcpp_process, "SDCPP_HOST", "http://localhost:9999")
    argv = sdcpp_process.build_argv("/b/sd-server", "/m/x.gguf", "--threads 2 --fa")
    assert argv == [
        "/b/sd-server",
        "-l",
        "127.0.0.1",
        "--listen-port",
        "9999",
        "-m",
        "/m/x.gguf",
        "--threads",
        "2",
        "--fa",
    ]


def test_tracking_round_trips_and_tolerates_corruption(tmp_path):
    assert sdcpp_process.read_tracking() is None
    sdcpp_process.write_tracking({"pid": 5})
    assert sdcpp_process.read_tracking() == {"pid": 5}
    (tmp_path / "sdcpp.json").write_text("{not json")
    assert sdcpp_process.read_tracking() is None
    sdcpp_process.write_tracking(None)
    assert not (tmp_path / "sdcpp.json").exists()


@pytest.mark.asyncio
async def test_start_surfaces_an_unsupported_model_and_clears_tracking(db, monkeypatch):
    await _configure(db)

    def fake_spawn(binary, model, extra):
        raise sdcpp_process.StartupError("This model is not supported by the installed stable-diffusion.cpp")

    monkeypatch.setattr(sdcpp_process, "spawn", fake_spawn)
    sdcpp_process.write_tracking({"pid": 1})
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda pid, path: False)

    with pytest.raises(ValueError, match="not supported"):
        await sdcpp_service.start(db)
    assert sdcpp_process.read_tracking() is None


@pytest.mark.asyncio
async def test_apply_leaves_a_stopped_server_stopped(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(sdcpp_process, "spawn", lambda *_a: pytest.fail("must not start"))
    assert (
        await sdcpp_service.apply(db, SdCppConfig(binary_path="/opt/sd/sd-server", model_path="/n.gguf"))
    ).running is False


@pytest.mark.asyncio
async def test_apply_restores_the_previous_model_when_the_new_one_fails(db, monkeypatch):
    await _configure(db)
    state = {"alive": True, "spawned": []}
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda pid, path: state["alive"])
    monkeypatch.setattr(sdcpp_process, "terminate", lambda pid, path: state.update(alive=False))
    monkeypatch.setattr(sdcpp_service, "_ping_health", _fake_ping(True))
    sdcpp_process.write_tracking({"pid": 7})

    def fake_spawn(binary, model, extra):
        state["spawned"].append(model)
        if model == "/bad.gguf":
            raise sdcpp_process.StartupError("This model is not supported.")
        state["alive"] = True
        return 8

    monkeypatch.setattr(sdcpp_process, "spawn", fake_spawn)
    with pytest.raises(ValueError, match="not supported.*Nothing was saved.*running again"):
        await sdcpp_service.apply(db, SdCppConfig(binary_path="/opt/sd/sd-server", model_path="/bad.gguf"))
    assert state["spawned"] == ["/bad.gguf", "/opt/sd/models/m.gguf"]
    assert (await image_engine_service.get_sdcpp_config(db)).model_path == "/opt/sd/models/m.gguf"


@pytest.mark.asyncio
async def test_apply_switches_model_when_it_loads(db, monkeypatch):
    await _configure(db)
    state = {"alive": True}
    monkeypatch.setattr(sdcpp_process, "is_alive", lambda pid, path: state["alive"])
    monkeypatch.setattr(sdcpp_process, "terminate", lambda pid, path: state.update(alive=False))
    monkeypatch.setattr(sdcpp_service, "_ping_health", _fake_ping(True))
    monkeypatch.setattr(sdcpp_process, "spawn", lambda *_a: state.update(alive=True) or 9)
    sdcpp_process.write_tracking({"pid": 7})
    status = await sdcpp_service.apply(db, SdCppConfig(binary_path="/opt/sd/sd-server", model_path="/ok.gguf"))
    assert status.running and sdcpp_process.read_tracking() == {"pid": 9}
