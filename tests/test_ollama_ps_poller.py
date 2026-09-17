"""Unit tests for app/services/ollama_ps_poller.py — the first periodic
background-loop precedent in this codebase (see its own module
docstring). Uses the shared `db` fixture, which already patches
ollama_ps_poller.AsyncSessionLocal at this module (see conftest.py)."""

import asyncio
from datetime import UTC, datetime

import httpx
import pytest
from sqlalchemy import select

from app.models import OllamaModelSnapshot
from app.services import ollama_pool, ollama_ps_poller


class _FakeResponse:
    def __init__(self, json_body: dict):
        self._json_body = json_body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_body


class _FakePsClient:
    """Stands in for httpx.AsyncClient's GET /api/ps — one canned
    response (or a raised error) per host, keyed by URL prefix."""

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
async def test_poll_once_writes_one_snapshot_row_per_loaded_model(monkeypatch, db):
    monkeypatch.setattr(ollama_pool, "get_effective_hosts", lambda: ["http://host-a:11434"])
    responses = {
        "http://host-a:11434": {
            "models": [
                {"name": "llama3:latest", "size": 100, "size_vram": 50, "expires_at": "2026-01-01T00:00:00Z"},
                {"name": "moondream:1.8b", "size": 200, "size_vram": 0},
            ]
        }
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient(responses))

    await ollama_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert {r.model_name for r in rows} == {"llama3:latest", "moondream:1.8b"}
    assert all(r.host == "http://host-a:11434" for r in rows)


@pytest.mark.asyncio
async def test_poll_once_skips_an_unreachable_host_but_still_polls_the_rest(monkeypatch, db):
    monkeypatch.setattr(ollama_pool, "get_effective_hosts", lambda: ["http://down:11434", "http://up:11434"])
    responses = {
        "http://down:11434": httpx.ConnectError("boom", request=httpx.Request("GET", "http://down:11434/api/ps")),
        "http://up:11434": {"models": [{"name": "llama3:latest", "size": 1, "size_vram": 1}]},
    }
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient(responses))

    await ollama_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert len(rows) == 1
    assert rows[0].host == "http://up:11434"


@pytest.mark.asyncio
async def test_poll_once_writes_nothing_when_no_models_are_loaded_anywhere(monkeypatch, db):
    monkeypatch.setattr(ollama_pool, "get_effective_hosts", lambda: ["http://host-a:11434"])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakePsClient({"http://host-a:11434": {"models": []}}))

    await ollama_ps_poller._poll_once()

    rows = (await db.execute(select(OllamaModelSnapshot))).scalars().all()
    assert rows == []


@pytest.mark.asyncio
async def test_run_ollama_ps_poller_survives_a_poll_failure(monkeypatch):
    """A single bad iteration (e.g. a transient DB error) must not kill the loop permanently — verified here by
    making the *first* _poll_once call raise and confirming a second call still happens."""
    calls = 0

    async def fake_poll_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")
        raise SystemExit  # stop the infinite loop cleanly once the second attempt has happened

    monkeypatch.setattr(ollama_ps_poller, "_poll_once", fake_poll_once)
    monkeypatch.setattr(ollama_ps_poller, "POLL_INTERVAL_SECONDS", 0)

    with pytest.raises(SystemExit):
        await ollama_ps_poller._run_ollama_ps_poller()

    assert calls == 2


@pytest.mark.asyncio
async def test_start_ollama_ps_poller_tracks_its_own_task_with_a_strong_reference(monkeypatch):
    """asyncio only holds a *weak* reference to a bare create_task() result — without the strong-ref set this
    module keeps, the poller task could be garbage-collected mid-run (same idiom as title_service's own
    background tasks)."""

    async def fake_run_forever():
        await asyncio.sleep(999)

    monkeypatch.setattr(ollama_ps_poller, "_run_ollama_ps_poller", fake_run_forever)

    ollama_ps_poller.start_ollama_ps_poller()

    assert len(ollama_ps_poller._background_poller_tasks) == 1
    # Cancel *and* await, not just fire-and-forget cancel() — an unawaited cancellation can keep resolving on
    # its own schedule after this test returns, bleeding stray event-loop activity into whatever test runs next
    # in the same process (see title_service's own background-task tests for this same convention).
    tasks = list(ollama_ps_poller._background_poller_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_parse_expires_at_handles_missing_and_malformed_values():
    assert ollama_ps_poller._parse_expires_at(None) is None
    assert ollama_ps_poller._parse_expires_at("not-a-date") is None
    parsed = ollama_ps_poller._parse_expires_at("2026-01-01T00:00:00Z")
    assert parsed == datetime(2026, 1, 1, tzinfo=UTC)
