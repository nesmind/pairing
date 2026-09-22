"""Unit tests for app/services/retention_poller.py — the periodic sweep
that keeps telemetry_events/ollama_model_snapshots/system_metric_snapshots
bounded for a process's entire uptime, not just between restarts (see
that module's own docstring for why a startup-only prune wasn't enough).
Uses the shared `db` fixture; conftest.py already patches
retention_poller.AsyncSessionLocal at this module."""

import asyncio
from datetime import timedelta

import pytest
from sqlalchemy import select

from app.models import OllamaModelSnapshot, SystemMetricSnapshot, TelemetryEvent
from app.models._base import utcnow
from app.schemas import RetentionSettings
from app.services import retention_poller, retention_settings_service


def _old_telemetry_event() -> TelemetryEvent:
    return TelemetryEvent(
        kind="chat",
        host="http://x:11434",
        success=True,
        started_at=utcnow() - timedelta(days=retention_settings_service.DEFAULT_TELEMETRY_RETENTION_DAYS + 1),
        duration_ms=1.0,
    )


def _recent_telemetry_event() -> TelemetryEvent:
    return TelemetryEvent(kind="chat", host="http://x:11434", success=True, started_at=utcnow(), duration_ms=1.0)


@pytest.mark.asyncio
async def test_prune_once_deletes_old_rows_and_keeps_recent_ones_across_all_three_tables(db):
    old_cutoff_ollama = utcnow() - timedelta(days=retention_settings_service.DEFAULT_TELEMETRY_RETENTION_DAYS + 1)
    old_cutoff_metrics = utcnow() - timedelta(days=retention_settings_service.DEFAULT_SYSTEM_METRICS_RETENTION_DAYS + 1)
    db.add_all(
        [
            _old_telemetry_event(),
            _recent_telemetry_event(),
            OllamaModelSnapshot(host="http://x:11434", model_name="llama3:latest", polled_at=old_cutoff_ollama),
            OllamaModelSnapshot(host="http://x:11434", model_name="llama3:latest", polled_at=utcnow()),
            SystemMetricSnapshot(
                cpu_percent=1.0,
                mem_used_bytes=1,
                mem_total_bytes=2,
                disk_used_bytes=1,
                disk_total_bytes=2,
                polled_at=old_cutoff_metrics,
            ),
            SystemMetricSnapshot(
                cpu_percent=1.0,
                mem_used_bytes=1,
                mem_total_bytes=2,
                disk_used_bytes=1,
                disk_total_bytes=2,
                polled_at=utcnow(),
            ),
        ]
    )
    await db.commit()

    await retention_poller.prune_once()

    assert len((await db.execute(select(TelemetryEvent))).scalars().all()) == 1
    assert len((await db.execute(select(OllamaModelSnapshot))).scalars().all()) == 1
    assert len((await db.execute(select(SystemMetricSnapshot))).scalars().all()) == 1


@pytest.mark.asyncio
async def test_prune_once_honors_an_admin_configured_retention_window(db):
    """Not just the defaults — a shorter admin-set window (Settings > System) must take effect on the very next
    sweep, no restart needed (see retention_settings_service.get_retention_settings' own docstring on why this
    is read fresh every call rather than cached)."""
    await retention_settings_service.set_retention_settings(
        db, RetentionSettings(telemetry_days=1, system_metrics_days=1)
    )
    db.add(
        TelemetryEvent(
            kind="chat",
            host="http://x:11434",
            success=True,
            started_at=utcnow() - timedelta(days=2),
            duration_ms=1.0,
        )
    )
    await db.commit()

    await retention_poller.prune_once()

    assert (await db.execute(select(TelemetryEvent))).scalars().all() == []


@pytest.mark.asyncio
async def test_prune_once_is_a_cheap_noop_when_nothing_is_stale(db):
    db.add(_recent_telemetry_event())
    await db.commit()

    await retention_poller.prune_once()  # must not raise, must not touch the fresh row

    rows = (await db.execute(select(TelemetryEvent))).scalars().all()
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_run_retention_poller_survives_a_sweep_failure(monkeypatch):
    calls = 0

    async def fake_prune_once():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient failure")
        raise SystemExit  # stop the infinite loop cleanly once the second attempt has happened

    monkeypatch.setattr(retention_poller, "prune_once", fake_prune_once)
    monkeypatch.setattr(retention_poller, "SWEEP_INTERVAL_SECONDS", 0)

    with pytest.raises(SystemExit):
        await retention_poller._run_retention_poller()

    assert calls == 2


@pytest.mark.asyncio
async def test_start_retention_poller_tracks_its_own_task_with_a_strong_reference(monkeypatch):
    async def fake_run_forever():
        await asyncio.sleep(999)

    monkeypatch.setattr(retention_poller, "_run_retention_poller", fake_run_forever)

    retention_poller.start_retention_poller()

    assert len(retention_poller._background_tasks) == 1
    tasks = list(retention_poller._background_tasks)
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task
