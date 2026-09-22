"""Unit tests for app/services/system_metrics_service.py."""

from datetime import timedelta

import psutil
import pytest

from app.models import GpuMetricSnapshot, SystemMetricSnapshot
from app.models._base import utcnow
from app.services import system_metrics_service


def _snapshot(**kwargs) -> SystemMetricSnapshot:
    defaults = {
        "cpu_percent": 10.0,
        "mem_used_bytes": 4_000_000_000,
        "mem_total_bytes": 8_000_000_000,
        "disk_used_bytes": 20_000_000_000,
        "disk_total_bytes": 100_000_000_000,
        "load_avg_1m": 0.5,
        "polled_at": utcnow(),
    }
    defaults.update(kwargs)
    return SystemMetricSnapshot(**defaults)


def _gpu_snapshot(**kwargs) -> GpuMetricSnapshot:
    defaults = {
        "gpu_index": 0,
        "name": "RTX 4090",
        "utilization_percent": 50.0,
        "mem_used_bytes": 8_000_000_000,
        "mem_total_bytes": 24_000_000_000,
        "temperature_c": 60.0,
        "polled_at": utcnow(),
    }
    defaults.update(kwargs)
    return GpuMetricSnapshot(**defaults)


@pytest.mark.asyncio
async def test_summary_on_an_empty_database(db, monkeypatch):
    monkeypatch.setattr(psutil, "cpu_count", lambda: 8)

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert summary.cpu_percent == 0.0
    assert summary.cpu_count == 8
    assert summary.mem_used_bytes == 0
    assert summary.polled_at is None
    assert summary.history == []
    assert summary.gpus == []
    assert summary.gpu_history == []


@pytest.mark.asyncio
async def test_summary_reflects_the_newest_snapshot(db):
    older = utcnow() - timedelta(minutes=5)
    db.add_all(
        [
            _snapshot(cpu_percent=10.0, polled_at=older),
            _snapshot(cpu_percent=99.0, polled_at=utcnow()),
        ]
    )
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert summary.cpu_percent == 99.0


@pytest.mark.asyncio
async def test_history_is_bounded_to_the_lookback_window(db):
    stale = utcnow() - timedelta(hours=2)
    db.add_all(
        [
            _snapshot(cpu_percent=5.0, polled_at=stale),
            _snapshot(cpu_percent=15.0, polled_at=utcnow()),
        ]
    )
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    # The stale point is outside the 1-hour lookback window and excluded from history — but "latest" is still
    # correct (a separate query, not derived from the bounded/windowed history list — see that function's own
    # comment on why).
    assert len(summary.history) == 1
    assert summary.history[0].cpu_percent == 15.0
    assert summary.cpu_percent == 15.0


@pytest.mark.asyncio
async def test_history_computes_memory_percent(db):
    db.add(_snapshot(mem_used_bytes=2_000_000_000, mem_total_bytes=8_000_000_000))
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert summary.history[0].mem_percent == 25.0


@pytest.mark.asyncio
async def test_history_point_count_is_capped(db, monkeypatch):
    monkeypatch.setattr(system_metrics_service, "_MAX_HISTORY_POINTS", 3)
    now = utcnow()
    db.add_all([_snapshot(cpu_percent=float(i), polled_at=now - timedelta(seconds=i)) for i in range(10)])
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert len(summary.history) == 3
    # The cap must never cost us the true latest value — see the "latest" query's own comment.
    assert summary.cpu_percent == 0.0


@pytest.mark.asyncio
async def test_gpus_reflects_only_the_newest_poll_cycle(db):
    """Two GPUs sampled together (same polled_at) at an older poll, then a newer poll with updated values —
    only the newest poll's rows should come back, not a stale/newer mix."""
    older = utcnow() - timedelta(seconds=30)
    newer = utcnow()
    db.add_all(
        [
            _gpu_snapshot(gpu_index=0, utilization_percent=10.0, polled_at=older),
            _gpu_snapshot(gpu_index=1, utilization_percent=20.0, polled_at=older),
            _gpu_snapshot(gpu_index=0, utilization_percent=45.0, polled_at=newer),
            _gpu_snapshot(gpu_index=1, utilization_percent=55.0, polled_at=newer),
        ]
    )
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert len(summary.gpus) == 2
    assert [g.gpu_index for g in summary.gpus] == [0, 1]  # ordered by gpu_index
    assert [g.utilization_percent for g in summary.gpus] == [45.0, 55.0]


@pytest.mark.asyncio
async def test_gpus_includes_name_and_temperature(db):
    db.add(_gpu_snapshot(name="Tesla T4", temperature_c=None))
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert summary.gpus[0].name == "Tesla T4"
    assert summary.gpus[0].temperature_c is None


@pytest.mark.asyncio
async def test_gpu_history_is_bounded_to_the_lookback_window(db):
    stale = utcnow() - timedelta(hours=2)
    db.add_all(
        [
            _gpu_snapshot(utilization_percent=5.0, polled_at=stale),
            _gpu_snapshot(utilization_percent=90.0, polled_at=utcnow()),
        ]
    )
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert len(summary.gpu_history) == 1
    assert summary.gpu_history[0].utilization_percent == 90.0


@pytest.mark.asyncio
async def test_gpu_history_point_count_is_capped(db, monkeypatch):
    monkeypatch.setattr(system_metrics_service, "_MAX_GPU_HISTORY_POINTS", 3)
    now = utcnow()
    db.add_all([_gpu_snapshot(polled_at=now - timedelta(seconds=i)) for i in range(10)])
    await db.commit()

    summary = await system_metrics_service.get_system_metrics_summary(db)

    assert len(summary.gpu_history) == 3
