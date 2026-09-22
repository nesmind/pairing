"""Unit tests for app/services/matricxon_ps_poller.py — the Matricxon-flavored twin of
tests/test_ollama_ps_poller.py (see that module's own docstring for the shared background-loop precedent this
mirrors). Uses the shared `db` fixture, which already patches matricxon_ps_poller.AsyncSessionLocal (see
conftest.py)."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select

from app.models import OllamaModelSnapshot
from app.services import matricxon_pool, matricxon_ps_poller


class _FakeResponse:
    def __init__(self, json_body: dict):
        self._json_body = json_body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_body


class _FakePsClient:
    def __init__(self, responses: dict[str, object]):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url: str):
        for host, outcome in self._responses.items():
            if url.startswith(host):
                if isinstance(outcome, Exception):
                    raise outcome
                return _FakeResponse(outcome)
        raise AssertionError(f"unexpected host in {url}")


@pytest.mark.asyncio
async def test_poll_once_writes_one_snapshot_row_per_loaded_model_stamped_with_the_matricxon_engine(monkeypatch, db):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://host-a:8420"])
    responses = {
        "http://host-a:8420": {
            "models": [
                {"name": "ministral-3:3b", "size": 100, "size_vram": 0, "expires_at": "2026-01-01T00:00:00Z"},
                {"name": "gemma4:2b", "size": 200, "size_vram": 0},
            ]
        }
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient(responses))

    await matricxon_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert {r.model_name for r in rows} == {"ministral-3:3b", "gemma4:2b"}
    assert all(r.host == "http://host-a:8420" for r in rows)
    assert all(r.engine == "matricxon" for r in rows)


@pytest.mark.asyncio
async def test_poll_once_skips_an_unreachable_host_but_still_polls_the_rest(monkeypatch, db):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://down:8420", "http://up:8420"])
    responses = {
        "http://down:8420": httpx.ConnectError("boom", request=httpx.Request("GET", "http://down:8420/api/ps")),
        "http://up:8420": {"models": [{"name": "ministral-3:3b", "size": 1, "size_vram": 0}]},
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient(responses))

    await matricxon_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert len(rows) == 1
    assert rows[0].host == "http://up:8420"


@pytest.mark.asyncio
async def test_poll_once_writes_nothing_when_no_models_are_loaded_anywhere(monkeypatch, db):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://host-a:8420"])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient({"http://host-a:8420": {"models": []}}))

    await matricxon_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_run_matricxon_ps_poller_survives_a_poll_failure(monkeypatch):
    """A single bad iteration (e.g. a transient DB error) must not kill the loop permanently — verified here by
    making the *first* _poll_once call raise and confirming a second call still happens."""
    calls = 0

    async def fake_poll_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")
        raise SystemExit  # stop the infinite loop cleanly once the second attempt has happened

    monkeypatch.setattr(matricxon_ps_poller, "_poll_once", fake_poll_once)
    monkeypatch.setattr(matricxon_ps_poller, "POLL_INTERVAL_SECONDS", 0)

    with pytest.raises(SystemExit):
        await matricxon_ps_poller._run_matricxon_ps_poller()

    assert calls == 2


@pytest.mark.asyncio
async def test_start_matricxon_ps_poller_tracks_its_own_task_with_a_strong_reference(monkeypatch):
    """asyncio only holds a *weak* reference to a bare create_task() result — without the strong-ref set this
    module keeps, the poller task could be garbage-collected mid-run (same idiom as ollama_ps_poller's own)."""

    async def fake_run_forever():
        await asyncio.sleep(999)

    monkeypatch.setattr(matricxon_ps_poller, "_run_matricxon_ps_poller", fake_run_forever)

    matricxon_ps_poller.start_matricxon_ps_poller()

    assert len(matricxon_ps_poller._background_poller_tasks) == 1
    tasks = list(matricxon_ps_poller._background_poller_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_parse_expires_at_handles_missing_and_malformed_values():
    assert matricxon_ps_poller._parse_expires_at(None) is None
    assert matricxon_ps_poller._parse_expires_at("not-a-date") is None
    parsed = matricxon_ps_poller._parse_expires_at("2026-01-01T00:00:00Z")
    assert parsed == datetime(2026, 1, 1, tzinfo=UTC)
