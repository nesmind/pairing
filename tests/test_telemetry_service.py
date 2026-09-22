"""Unit tests for app/services/telemetry_service.py — the by-host/by-model group_by() counts are plain SQL (same
convention as tests/test_stats_service.py), but the latency time-series bucketing and p95 figure are hand-rolled
Python (see that module's own docstring on why SQL isn't portable enough here), so those get direct, exact-value
coverage rather than just a smoke test. Every _event() below defaults to engine="ollama" (TelemetryEvent's own
Python-side column default — see app/models/telemetry.py) and every test asks get_engine_telemetry_summary for
"ollama" specifically, matching that default; test_a_matricxon_events_never_leak_into_ollamas_summary at the
bottom is the one test that actually exercises the engine filter itself."""

from datetime import timedelta

import pytest

from app.models import TelemetryEvent
from app.models._base import utcnow
from app.services import telemetry_service


def _event(**kwargs) -> TelemetryEvent:
    defaults = {
        "kind": "chat",
        "host": "http://x:11434",
        "success": True,
        "started_at": utcnow(),
        "duration_ms": 100.0,
        "attempt_count": 1,
    }
    defaults.update(kwargs)
    return TelemetryEvent(**defaults)


@pytest.mark.asyncio
async def test_summary_on_an_empty_database(db):
    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert summary.total_requests == 0
    assert summary.error_count == 0
    assert summary.by_model == []
    assert summary.by_host == []
    assert summary.latency_by_model == []
    assert summary.latency_time_series == []
    assert summary.loaded_models == []


@pytest.mark.asyncio
async def test_total_and_error_counts_include_every_event_regardless_of_age(db):
    old = utcnow() - timedelta(days=30)
    db.add_all(
        [
            _event(success=True, started_at=old),
            _event(success=False, error_message="boom", started_at=old),
            _event(success=True),
        ]
    )
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert summary.total_requests == 3
    assert summary.error_count == 1


@pytest.mark.asyncio
async def test_by_model_and_by_host_group_correctly(db):
    db.add_all(
        [
            _event(model="llama3:latest", host="http://a:11434"),
            _event(model="llama3:latest", host="http://a:11434"),
            _event(model="moondream:1.8b", host="http://b:11434", success=False, error_message="x"),
        ]
    )
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    by_model = {c.key: c.count for c in summary.by_model}
    assert by_model == {"llama3:latest": 2, "moondream:1.8b": 1}

    by_host = {h.host: (h.request_count, h.error_count) for h in summary.by_host}
    assert by_host == {"http://a:11434": (2, 0), "http://b:11434": (1, 1)}


@pytest.mark.asyncio
async def test_events_without_a_model_are_excluded_from_by_model(db):
    db.add(_event(model=None))
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert summary.by_model == []
    assert summary.total_requests == 1


@pytest.mark.asyncio
async def test_latency_by_model_computes_avg_and_bounded_p95(db):
    db.add_all(
        [
            _event(model="llama3:latest", duration_ms=100.0),
            _event(model="llama3:latest", duration_ms=200.0),
        ]
    )
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    (stat,) = summary.latency_by_model
    assert stat.model == "llama3:latest"
    assert stat.request_count == 2
    assert stat.avg_duration_ms == 150.0
    # method="inclusive" keeps p95 within the observed min/max — see telemetry_service's own comment on why the
    # default "exclusive" method can extrapolate past the actual max with only a couple of samples.
    assert 100.0 <= stat.p95_duration_ms <= 200.0


@pytest.mark.asyncio
async def test_failed_events_are_excluded_from_latency_stats(db):
    db.add(_event(model="llama3:latest", success=False, error_message="boom", duration_ms=9999.0))
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert summary.latency_by_model == []
    assert summary.latency_time_series == []


@pytest.mark.asyncio
async def test_events_outside_the_lookback_window_are_excluded_from_latency_stats(db):
    old = utcnow() - timedelta(days=8)
    db.add(_event(model="llama3:latest", started_at=old, duration_ms=100.0))
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert summary.latency_by_model == []
    # But it still counts toward the plain totals/by_model breakdown, which have no lookback window.
    assert summary.total_requests == 1


@pytest.mark.asyncio
async def test_latency_time_series_buckets_by_hour(db):
    hour_one = utcnow().replace(minute=5, second=0, microsecond=0)
    hour_two = hour_one + timedelta(hours=1)
    db.add_all(
        [
            _event(model="llama3:latest", started_at=hour_one, duration_ms=100.0),
            _event(model="llama3:latest", started_at=hour_one.replace(minute=45), duration_ms=200.0),
            _event(model="llama3:latest", started_at=hour_two, duration_ms=50.0),
        ]
    )
    await db.commit()

    summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")

    assert len(summary.latency_time_series) == 2
    first, second = summary.latency_time_series
    assert first.request_count == 2
    assert first.avg_duration_ms == 150.0
    assert second.request_count == 1
    assert second.avg_duration_ms == 50.0


@pytest.mark.asyncio
async def test_a_matricxon_events_never_leak_into_ollamas_summary(db):
    """The real feature this covers: both engines write to the same TelemetryEvent table (see that model's own
    docstring) — asking for Ollama's summary must never count, group, or average in a Matricxon call, and vice
    versa, regardless of how similar the model/host names happen to be."""
    db.add_all(
        [
            _event(model="shared-name:latest", host="http://shared:1", engine="ollama"),
            _event(model="shared-name:latest", host="http://shared:1", engine="matricxon", success=False),
        ]
    )
    await db.commit()

    ollama_summary = await telemetry_service.get_engine_telemetry_summary(db, "ollama")
    matricxon_summary = await telemetry_service.get_engine_telemetry_summary(db, "matricxon")

    assert ollama_summary.total_requests == 1
    assert ollama_summary.error_count == 0
    assert matricxon_summary.total_requests == 1
    assert matricxon_summary.error_count == 1
