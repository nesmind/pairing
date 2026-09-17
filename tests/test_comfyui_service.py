"""Unit tests for app/services/comfyui_service.py — async orchestration
+ locking around app/services/comfyui_process.py's primitives, all
monkeypatched (same convention as tests/test_instance_service.py for the
equivalent "local instances" feature)."""

import pytest

from app.schemas import ComfyUIProcessConfig
from app.services import comfyui_process, comfyui_service, settings_service


@pytest.fixture(autouse=True)
def isolated_tracking_file(tmp_path, monkeypatch):
    monkeypatch.setattr(comfyui_process, "_TRACKING_FILE", tmp_path / "comfyui.json")


async def _configure(db):
    await settings_service.set_comfyui_config(
        db, ComfyUIProcessConfig(python_path="/opt/venv/bin/python", main_py_path="/opt/ComfyUI/main.py")
    )


@pytest.mark.asyncio
async def test_start_rejects_when_unconfigured(db):
    with pytest.raises(ValueError):
        await comfyui_service.start(db)


@pytest.mark.asyncio
async def test_start_rejects_on_a_non_primary_instance(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(comfyui_service, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await comfyui_service.start(db)


@pytest.mark.asyncio
async def test_start_spawns_and_writes_tracking(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(comfyui_process, "spawn", lambda *_a, **_kw: 4242)
    monkeypatch.setattr(comfyui_process, "is_alive", lambda *_a, **_kw: True)
    monkeypatch.setattr(comfyui_service, "_ping_health", _fake_ping(True))

    status = await comfyui_service.start(db)

    assert status.running is True
    assert status.pid == 4242
    assert comfyui_process.read_tracking() == {"pid": 4242}


@pytest.mark.asyncio
async def test_start_is_a_noop_when_already_running(db, monkeypatch):
    await _configure(db)
    comfyui_process.write_tracking({"pid": 111})
    monkeypatch.setattr(comfyui_process, "is_alive", lambda *_a, **_kw: True)
    monkeypatch.setattr(comfyui_process, "spawn", lambda *_a, **_kw: pytest.fail("should not spawn again"))
    monkeypatch.setattr(comfyui_service, "_ping_health", _fake_ping(True))

    status = await comfyui_service.start(db)

    assert status.pid == 111


@pytest.mark.asyncio
async def test_start_clears_tracking_when_spawn_fails_immediately(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(comfyui_process, "spawn", lambda *_a, **_kw: None)

    status = await comfyui_service.start(db)

    assert status.running is False
    assert comfyui_process.read_tracking() is None


@pytest.mark.asyncio
async def test_stop_rejects_on_a_non_primary_instance(db, monkeypatch):
    monkeypatch.setattr(comfyui_service, "IS_PRIMARY", False)
    with pytest.raises(ValueError):
        await comfyui_service.stop(db)


@pytest.mark.asyncio
async def test_stop_terminates_and_clears_tracking(db, monkeypatch):
    await _configure(db)
    comfyui_process.write_tracking({"pid": 555})
    terminated = []
    monkeypatch.setattr(comfyui_process, "terminate", lambda pid, _path: terminated.append(pid))
    monkeypatch.setattr(comfyui_process, "is_alive", lambda *_a, **_kw: False)

    status = await comfyui_service.stop(db)

    assert terminated == [555]
    assert status.running is False
    assert comfyui_process.read_tracking() is None


@pytest.mark.asyncio
async def test_stop_is_a_noop_when_not_running(db, monkeypatch):
    await _configure(db)
    monkeypatch.setattr(comfyui_process, "terminate", lambda *_a: pytest.fail("should not terminate"))

    status = await comfyui_service.stop(db)

    assert status.running is False


@pytest.mark.asyncio
async def test_get_status_reports_not_running_when_never_started(db):
    status = await comfyui_service.get_status(db)
    assert status.running is False
    assert status.pid is None


def _fake_ping(healthy: bool):
    async def _fake():
        return healthy

    return _fake
