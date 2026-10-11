"""Periodically prunes the four append-only poller-written tables (telemetry_events, model_snapshots,
system_metric_snapshots, gpu_metric_snapshots; writers: ollama_ps_poller, system_metrics_poller,
ollama_telemetry). A prune only at startup bounds growth between restarts; a deployment up for weeks needs
it periodic too, or the tables grow for the whole uptime. Same detached-task/strong-ref-set idiom as the
two pollers above; primary-gated for the same reason (shared tables).

The windows are admin-configurable (Settings > System, retention_settings_service) and read fresh each
sweep, so a change applies at the next run, not after a restart.
"""

import asyncio
import logging
from datetime import timedelta

from sqlalchemy import delete

from app.database import AsyncSessionLocal
from app.models import GpuMetricSnapshot, ModelSnapshot, SystemMetricSnapshot, TelemetryEvent
from app.models._base import utcnow
from app.services import retention_settings_service

logger = logging.getLogger("llama_chat")

# Every 6 hours: an indexed DELETE over a usually empty range is cheap, so sweeping more than daily costs little
# and keeps each table's peak near its retention window instead of drifting up over a day.
SWEEP_INTERVAL_SECONDS = 6 * 60 * 60

_background_tasks: set[asyncio.Task] = set()


async def prune_once() -> None:
    async with AsyncSessionLocal() as db:
        settings = await retention_settings_service.get_retention_settings(db)
        telemetry_cutoff = utcnow() - timedelta(days=settings.telemetry_days)
        metrics_cutoff = utcnow() - timedelta(days=settings.system_metrics_days)
        deleted_events = (
            await db.execute(delete(TelemetryEvent).where(TelemetryEvent.started_at < telemetry_cutoff))
        ).rowcount
        deleted_ollama_snapshots = (
            await db.execute(delete(ModelSnapshot).where(ModelSnapshot.polled_at < telemetry_cutoff))
        ).rowcount
        deleted_metric_snapshots = (
            await db.execute(delete(SystemMetricSnapshot).where(SystemMetricSnapshot.polled_at < metrics_cutoff))
        ).rowcount
        deleted_gpu_snapshots = (
            await db.execute(delete(GpuMetricSnapshot).where(GpuMetricSnapshot.polled_at < metrics_cutoff))
        ).rowcount
        await db.commit()
    if deleted_events or deleted_ollama_snapshots or deleted_metric_snapshots or deleted_gpu_snapshots:
        logger.info(
            "Pruned %d old telemetry event(s), %d old ML engine model snapshot(s), %d old system metric snapshot(s), "
            "%d old GPU metric snapshot(s).",
            deleted_events,
            deleted_ollama_snapshots,
            deleted_metric_snapshots,
            deleted_gpu_snapshots,
        )


async def _run_retention_poller() -> None:
    while True:
        try:
            await prune_once()
        except Exception:
            # A bad sweep (e.g. a transient DB error) must never kill the loop for good.
            logger.exception("Retention sweep failed.")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


def start_retention_poller() -> None:
    """Called once from startup_service.run_startup_tasks (primary only): an initial sweep almost at once (before the
    first sleep), then every SWEEP_INTERVAL_SECONDS.
    """
    task = asyncio.create_task(_run_retention_poller())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
