"""Tests for services/matricxon_live_settings.py and the /plan and /apply-live routes: a Matricxon settings Save
restarts only for settings fixed at process start; the rest go to the running process."""

import httpx
import pytest
from fastapi import HTTPException

from app.routers import matricxon_admin
from app.schemas import MatricxonServerConfig
from app.services import matricxon_http, settings_service
from app.services.matricxon_live_settings import LiveApplyUnsupportedError, MatricxonSettingsPlan, push_live


def _plan(**changes) -> MatricxonSettingsPlan:
    old = MatricxonServerConfig()
    return MatricxonSettingsPlan(old, old.model_copy(update=changes))


class TestPlan:
    def test_live_changes_need_no_restart(self):
        plan = _plan(max_decode_batch=4, log_level=1, gemv_backend="numba")
        assert plan.restart_required is False
        assert plan.live_payload() == {"max_decode_batch": 4, "log_level": 1, "gemv_backend": "numba"}

    def test_keep_alive_is_sent_in_seconds(self):
        assert _plan(keep_alive_minutes=10).live_payload() == {"default_keep_alive_seconds": 600}

    def test_a_cleared_value_asks_for_the_default(self):
        old = MatricxonServerConfig(prompt_cache_slots=8)
        plan = MatricxonSettingsPlan(old, old.model_copy(update={"prompt_cache_slots": None}))
        assert plan.live_payload() == {"prompt_cache_slots": None}

    def test_start_time_settings_need_a_restart(self):
        plan = _plan(device="cuda", torch_threads=2, max_loaded_models=3)
        assert plan.restart_required is True
        assert plan.restart_labels == ["Compute device", "CPU threads"]

    def test_nothing_changed_needs_nothing(self):
        plan = _plan(remote_hosts=["http://10.0.0.9:8420"])
        assert plan.restart_required is False and plan.live_payload() == {}

    def test_cluster_mode_keeps_the_restart_path(self):
        old = MatricxonServerConfig(cluster_role="seed")
        plan = MatricxonSettingsPlan(old, old.model_copy(update={"max_decode_batch": 4}))
        assert plan.restart_required is True and plan.restart_labels == ["Cluster mode"]


def _fake_matricxon(monkeypatch, handler):
    """Routes matricxon_http.client calls to `handler` instead of the network."""
    real_client = matricxon_http.client
    monkeypatch.setattr(
        matricxon_http, "client", lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw)
    )


class TestPushLive:
    @pytest.mark.asyncio
    async def test_returns_what_matricxon_applied(self, monkeypatch):
        seen = {}

        def handler(request):
            seen["path"], seen["body"] = request.url.path, request.content
            return httpx.Response(200, json={"applied_now": ["max_decode_batch"], "applied_next_load": []})

        _fake_matricxon(monkeypatch, handler)
        result = await push_live({"max_decode_batch": 4})
        assert seen["path"] == "/api/runtime-settings" and b'"max_decode_batch":4' in seen["body"]
        assert result == {"applied_now": ["max_decode_batch"], "applied_next_load": []}

    @pytest.mark.asyncio
    async def test_an_older_matricxon_is_reported_as_unsupported(self, monkeypatch):
        _fake_matricxon(monkeypatch, lambda _r: httpx.Response(404))
        with pytest.raises(LiveApplyUnsupportedError):
            await push_live({"log_level": 1})

    @pytest.mark.asyncio
    async def test_an_unreachable_matricxon_is_a_runtime_error(self, monkeypatch):
        def handler(_request):
            raise httpx.ConnectError("refused")

        _fake_matricxon(monkeypatch, handler)
        with pytest.raises(RuntimeError, match="Could not reach Matricxon"):
            await push_live({"log_level": 1})


class TestRoutes:
    @pytest.mark.asyncio
    async def test_plan_reports_a_restart_with_labels(self, db):
        result = await matricxon_admin.save_plan(MatricxonServerConfig(device="cuda"), db=db, _admin=None)
        assert result.restart_required is True and result.restart_labels == ["Compute device"]

    @pytest.mark.asyncio
    async def test_apply_live_pushes_then_saves(self, db, monkeypatch):
        pushed = {}

        async def fake_push(payload):
            pushed.update(payload)
            return {"applied_now": ["max_decode_batch"], "applied_next_load": []}

        monkeypatch.setattr(matricxon_admin, "push_live", fake_push)
        result = await matricxon_admin.apply_live(MatricxonServerConfig(max_decode_batch=5), db=db, _admin=None)
        assert pushed == {"max_decode_batch": 5}
        assert result.applied_now == ["max_decode_batch"]
        assert (await settings_service.get_matricxon_server_config(db)).max_decode_batch == 5

    @pytest.mark.asyncio
    async def test_apply_live_refuses_changes_that_need_a_restart(self, db, monkeypatch):
        monkeypatch.setattr(matricxon_admin, "push_live", lambda _p: pytest.fail("must not push"))
        with pytest.raises(HTTPException) as exc:
            await matricxon_admin.apply_live(MatricxonServerConfig(torch_threads=2), db=db, _admin=None)
        assert exc.value.status_code == 409

    @pytest.mark.asyncio
    async def test_apply_live_does_not_save_when_matricxon_refuses(self, db, monkeypatch):
        async def failing_push(_payload):
            raise RuntimeError("Matricxon refused the new settings (422)")

        monkeypatch.setattr(matricxon_admin, "push_live", failing_push)
        with pytest.raises(HTTPException) as exc:
            await matricxon_admin.apply_live(MatricxonServerConfig(max_decode_batch=6), db=db, _admin=None)
        assert exc.value.status_code == 502
        assert (await settings_service.get_matricxon_server_config(db)).max_decode_batch != 6

    @pytest.mark.asyncio
    async def test_apply_live_turns_an_older_matricxon_into_409(self, db, monkeypatch):
        async def unsupported(_payload):
            raise LiveApplyUnsupportedError("no route")

        monkeypatch.setattr(matricxon_admin, "push_live", unsupported)
        with pytest.raises(HTTPException) as exc:
            await matricxon_admin.apply_live(MatricxonServerConfig(log_level=2), db=db, _admin=None)
        assert exc.value.status_code == 409
