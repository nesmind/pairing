"""Periodically prunes the four append-only, poller-written tables this
app accumulates over time (telemetry_events, ollama_model_snapshots,
system_metric_snapshots, gpu_metric_snapshots — see
app.services.ollama_ps_poller, app.services.system_metrics_poller, and
app.services.ollama_telemetry for what writes them). A one-shot prune
at process startup only bounds
growth *between restarts* — a self-hosted deployment that stays up for
weeks or months between restarts needs this to actually run
periodically too, or these tables grow unbounded for the process's
entire uptime. Same detached-task/strong-ref-set idiom as the two
pollers above; primary-gated the same way and for the same reason
(host-level/shared tables, not per-instance ones).

The retention windows themselves are admin-configurable (Settings >
System, see app.services.retention_settings_service) — read fresh at
the start of every sweep, not cached, so a change takes effect on the
next run rather than requiring a restart.
"""

import asyncio
import logging
from datetime import timedelta

from sqlalchemy import delete

from app.database import AsyncSessionLocal
from app.models import GpuMetricSnapshot, OllamaModelSnapshot, SystemMetricSnapshot, TelemetryEvent
from app.models._base import utcnow
from app.services import retention_settings_service

logger = logging.getLogger("llama_chat")

# Every 6 hours — an indexed DELETE against a range that's usually empty (nothing stale yet) is cheap, so there's
# no real cost to sweeping more often than daily, and it keeps each table's actual peak size closer to its
# intended retention window instead of drifting up over the course of a day.
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
            await db.execute(delete(OllamaModelSnapshot).where(OllamaModelSnapshot.polled_at < telemetry_cutoff))
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
            # A single bad sweep (e.g. a transient DB error) must never kill the loop permanently.
            logger.exception("Retention sweep failed.")
        await asyncio.sleep(SWEEP_INTERVAL_SECONDS)


def start_retention_poller() -> None:
    """Called once, from startup_service.run_startup_tasks (primary
    only) — runs an initial sweep almost immediately (first loop
    iteration, before the first sleep), then every SWEEP_INTERVAL_SECONDS
    for as long as the process stays up."""
    task = asyncio.create_task(_run_retention_poller())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
