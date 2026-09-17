"""Aggregation behind GET /api/system-metrics/summary — the Stats
page's "System" view (see app/routers/system_metrics.py and
app/services/system_metrics_poller.py for how the underlying rows get
written). Unlike telemetry_service.py, there's no SQL-portability
concern here worth working around: the only queries are "give me the
newest row" and "give me a bounded, already-ordered range of rows" —
both fully portable ORDER BY + LIMIT, no percentiles or date-bucketing
needed, since the frontend chart draws every point directly rather than
pre-aggregating into buckets.
"""

from datetime import timedelta

import psutil
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import GpuMetricSnapshot, SystemMetricSnapshot
from app.models._base import utcnow
from app.schemas import GpuHistoryPoint, GpuStat, SystemMetricPoint, SystemMetricsSummary

_HISTORY_WINDOW = timedelta(hours=1)
_MAX_HISTORY_POINTS = 500
# Several GPUs each contribute a row per poll (unlike the single CPU/RAM row above), so this cap is scaled up to
# comfortably cover a handful of GPUs over the full history window without truncating in the common case.
_MAX_GPU_HISTORY_POINTS = 2000


def _mem_percent(snapshot: SystemMetricSnapshot) -> float:
    return (snapshot.mem_used_bytes / snapshot.mem_total_bytes * 100) if snapshot.mem_total_bytes else 0.0


async def _latest_gpu_stats(db: AsyncSession) -> list[GpuStat]:
    """Every GPU row from one poll cycle shares the exact same polled_at (see
    app.services.system_metrics_poller._sample_gpus_sync) — so "the latest full GPU list" is simply every row at
    the newest polled_at value, not a per-gpu_index dedupe the way app.services.ollama_snapshot_service needs
    (Ollama's loaded-model set isn't guaranteed to be sampled in one atomic pass the way this is)."""
    latest_polled_at = (await db.execute(select(func.max(GpuMetricSnapshot.polled_at)))).scalar_one_or_none()
    if latest_polled_at is None:
        return []
    rows = (
        (
            await db.execute(
                select(GpuMetricSnapshot)
                .where(GpuMetricSnapshot.polled_at == latest_polled_at)
                .order_by(GpuMetricSnapshot.gpu_index)
            )
        )
        .scalars()
        .all()
    )
    return [
        GpuStat(
            gpu_index=r.gpu_index,
            name=r.name,
            utilization_percent=r.utilization_percent,
            mem_used_bytes=r.mem_used_bytes,
            mem_total_bytes=r.mem_total_bytes,
            temperature_c=r.temperature_c,
            polled_at=r.polled_at,
        )
        for r in rows
    ]


async def _gpu_history(db: AsyncSession) -> list[GpuHistoryPoint]:
    cutoff = utcnow() - _HISTORY_WINDOW
    rows = (
        (
            await db.execute(
                select(GpuMetricSnapshot)
                .where(GpuMetricSnapshot.polled_at >= cutoff)
                .order_by(GpuMetricSnapshot.polled_at)
                .limit(_MAX_GPU_HISTORY_POINTS)
            )
        )
        .scalars()
        .all()
    )
    return [
        GpuHistoryPoint(polled_at=r.polled_at, gpu_index=r.gpu_index, utilization_percent=r.utilization_percent)
        for r in rows
    ]


async def get_system_metrics_summary(db: AsyncSession) -> SystemMetricsSummary:
    cpu_count = psutil.cpu_count() or 0

    # A separate query for "latest" rather than just taking the last element of the history list below: if a
    # deployment ever polls faster or keeps a longer window than _MAX_HISTORY_POINTS covers, that list's own
    # LIMIT would silently truncate off the newest rows (it's ordered oldest-first, for the chart) — the current
    # values shown above the chart must always be the real latest sample regardless of that cap.
    latest = (
        await db.execute(select(SystemMetricSnapshot).order_by(SystemMetricSnapshot.polled_at.desc()).limit(1))
    ).scalar_one_or_none()

    cutoff = utcnow() - _HISTORY_WINDOW
    rows = (
        (
            await db.execute(
                select(SystemMetricSnapshot)
                .where(SystemMetricSnapshot.polled_at >= cutoff)
                .order_by(SystemMetricSnapshot.polled_at)
                .limit(_MAX_HISTORY_POINTS)
            )
        )
        .scalars()
        .all()
    )
    history = [
        SystemMetricPoint(polled_at=r.polled_at, cpu_percent=r.cpu_percent, mem_percent=_mem_percent(r)) for r in rows
    ]

    # Computed regardless of whether a CPU/RAM snapshot exists yet — both are written in the same poll cycle in
    # practice, but there's no reason to make the GPU data depend on the other table having a row.
    gpus = await _latest_gpu_stats(db)
    gpu_history = await _gpu_history(db)

    if latest is None:
        return SystemMetricsSummary(
            cpu_percent=0.0,
            cpu_count=cpu_count,
            mem_used_bytes=0,
            mem_total_bytes=0,
            disk_used_bytes=0,
            disk_total_bytes=0,
            load_avg_1m=None,
            polled_at=None,
            history=[],
            gpus=gpus,
            gpu_history=gpu_history,
        )
    return SystemMetricsSummary(
        cpu_percent=latest.cpu_percent,
        cpu_count=cpu_count,
        mem_used_bytes=latest.mem_used_bytes,
        mem_total_bytes=latest.mem_total_bytes,
        disk_used_bytes=latest.disk_used_bytes,
        disk_total_bytes=latest.disk_total_bytes,
        load_avg_1m=latest.load_avg_1m,
        polled_at=latest.polled_at,
        history=history,
        gpus=gpus,
        gpu_history=gpu_history,
    )
