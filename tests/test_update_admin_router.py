"""Tests for routers/update_admin.py and services/update_targets.py (router functions called directly)."""

import json

import pytest
from fastapi import HTTPException

from app.routers import update_admin
from app.schemas import MatricxonServerConfig, MatricxonServerStatus
from app.schemas.update import SourceUpdateStatus
from app.services import matricxon_process
from app.services.source_updater import SourceUpdater
from app.services.update_targets import MatricxonUpdate, PairingUpdate


async def _fake_status(self):
    return SourceUpdateStatus(name=self.target.name, available=True, method="git", update_available=False)


async def _body(response) -> list[dict]:
    chunks = [chunk async for chunk in response.body_iterator]
    return [json.loads(chunk.removeprefix("data: ")) for chunk in chunks]


@pytest.fixture(autouse=True)
def _not_in_a_container(monkeypatch):
    monkeypatch.setattr(PairingUpdate, "in_container", lambda _self: False)


@pytest.mark.asyncio
async def test_status_has_no_matricxon_when_not_installed(db, monkeypatch):
    monkeypatch.setattr(SourceUpdater, "status", _fake_status)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _dir=None: None)

    overview = await update_admin.status(db=db, _admin=None)

    assert overview.pairing.name == "pAIring"
    assert overview.matricxon is None


@pytest.mark.asyncio
async def test_status_includes_a_local_matricxon(db, monkeypatch, tmp_path):
    monkeypatch.setattr(SourceUpdater, "status", _fake_status)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _dir=None: tmp_path)

    overview = await update_admin.status(db=db, _admin=None)

    assert overview.matricxon.name == "Matricxon"


@pytest.mark.asyncio
async def test_update_matricxon_rejects_when_not_installed(db, monkeypatch):
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _dir=None: None)
    with pytest.raises(HTTPException) as exc:
        await update_admin.update_matricxon(db=db, _admin=None)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_update_pairing_streams_the_updater_events(db, monkeypatch):
    async def fake_stream(_self):
        yield {"step": 1, "total_steps": 3, "status": "Fetching"}
        yield {"done": True, "changed": True}

    monkeypatch.setattr(SourceUpdater, "update_stream", fake_stream)

    events = await _body(await update_admin.update_pairing(db=db, _admin=None))

    assert events[-1] == {"done": True, "changed": True}


@pytest.mark.asyncio
async def test_matricxon_update_restarts_a_running_server(monkeypatch, tmp_path):
    calls = []

    async def fake_stream(_self):
        yield {"done": True, "changed": True}

    async def fake_get_status(_dir=None):
        return MatricxonServerStatus(running=True, installed=True)

    async def fake_stop(_dir=None):
        calls.append("stop")

    async def fake_start(_config, proxy_url=None):
        calls.append("start")

    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _dir=None: tmp_path)
    monkeypatch.setattr(matricxon_process, "get_status", fake_get_status)
    monkeypatch.setattr(matricxon_process, "stop", fake_stop)
    monkeypatch.setattr(matricxon_process, "start", fake_start)
    monkeypatch.setattr(SourceUpdater, "update_stream", fake_stream)

    events = [e async for e in MatricxonUpdate(MatricxonServerConfig()).update_stream()]

    assert calls == ["stop", "start"]
    assert events[-1]["done"] is True and events[-1]["restarted"] is True


def test_restart_info_prefers_the_script_outside_systemd(monkeypatch, tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "restart.sh").write_text("")
    monkeypatch.setattr(PairingUpdate, "systemd_service", staticmethod(lambda: None))

    info = PairingUpdate(base_dir=tmp_path).restart_info()

    assert info.mode == "script" and info.can_restart is True


def test_restart_info_under_systemd_gives_the_command(monkeypatch, tmp_path):
    monkeypatch.setattr(PairingUpdate, "systemd_service", staticmethod(lambda: "pairing"))

    info = PairingUpdate(base_dir=tmp_path).restart_info()

    assert info.can_restart is False
    assert info.hint == "sudo systemctl restart pairing"


@pytest.mark.asyncio
async def test_restart_endpoint_refuses_without_a_script(monkeypatch, tmp_path):
    monkeypatch.setattr(PairingUpdate, "systemd_service", staticmethod(lambda: None))
    monkeypatch.setattr(update_admin, "PairingUpdate", lambda: PairingUpdate(base_dir=tmp_path))
    with pytest.raises(HTTPException) as exc:
        await update_admin.restart_pairing(_admin=None)
    assert exc.value.status_code == 400
