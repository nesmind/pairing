"""Aggregation queries behind GET /api/telemetry/ollama/summary — the
Telemetry page's Ollama dashboard (see app/routers/telemetry.py and
app/static/js/telemetry.js). Request counts grouped by host/model are
plain SQL group_by() (small cardinality, matches
app/services/stats_service.py's own convention).

Latency-over-time buckets and the p95 figure are deliberately NOT SQL:
SQLite's strftime() has no MySQL equivalent (that's DATE_FORMAT, a
different function), and plain MySQL 8 has no portable percentile
function either. Instead, one bounded range query (_LOOKBACK_WINDOW +
_MAX_EVENTS_FETCHED as a hard safety cap — never an unbounded scan, and
telemetry_events' own 30-day retention already keeps the table itself
small) fetches the raw rows once, and both the hourly buckets and the
p95 are computed from that single result set in Python. This is a
narrow, deliberate exception to "always aggregate in SQL" — justified
specifically because the fetch is bounded, not because Python happens to
be easier here.
"""

import statistics
from datetime import timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import TelemetryEvent
from app.models._base import utcnow
from app.schemas import (
    CountByKey,
    HostStats,
    LatencyTimeBucket,
    ModelLatencyStats,
    OllamaTelemetrySummary,
)
from app.services import ollama_snapshot_service

_LOOKBACK_WINDOW = timedelta(days=7)
_MAX_EVENTS_FETCHED = 10_000


async def _counts_by_model(db: AsyncSession) -> list[CountByKey]:
    rows = (
        await db.execute(
            select(TelemetryEvent.model, func.count(TelemetryEvent.id))
            .where(TelemetryEvent.model.isnot(None))
            .group_by(TelemetryEvent.model)
            .order_by(func.count(TelemetryEvent.id).desc())
        )
    ).all()
    return [CountByKey(key=model, count=count) for model, count in rows]


async def _stats_by_host(db: AsyncSession) -> list[HostStats]:
    """Two small group_by queries (row count = one per configured host,
    always tiny) rather than one query with a conditional-sum for the
    error count — plain group_by(host) is portable and reads clearly."""
    totals = dict(
        (
            await db.execute(select(TelemetryEvent.host, func.count(TelemetryEvent.id)).group_by(TelemetryEvent.host))
        ).all()
    )
    error_rows = dict(
        (
            await db.execute(
                select(TelemetryEvent.host, func.count(TelemetryEvent.id))
                .where(TelemetryEvent.success.is_(False))
                .group_by(TelemetryEvent.host)
            )
        ).all()
    )
    return [
        HostStats(host=host, request_count=count, error_count=error_rows.get(host, 0)) for host, count in totals.items()
    ]


async def _fetch_recent_events(db: AsyncSession) -> list[TelemetryEvent]:
    cutoff = utcnow() - _LOOKBACK_WINDOW
    rows = (
        (
            await db.execute(
                select(TelemetryEvent)
                .where(TelemetryEvent.started_at >= cutoff, TelemetryEvent.success.is_(True))
                .order_by(TelemetryEvent.started_at.desc())
                .limit(_MAX_EVENTS_FETCHED)
            )
        )
        .scalars()
        .all()
    )
    return list(rows)


def _latency_by_model(events: list[TelemetryEvent]) -> list[ModelLatencyStats]:
    by_model: dict[str, list[float]] = {}
    for event in events:
        if event.model:
            by_model.setdefault(event.model, []).append(event.duration_ms)
    stats = []
    for model, durations in by_model.items():
        # method="inclusive" (rather than statistics' default "exclusive") keeps p95 within the observed
        # min/max — the default can extrapolate past the actual max with a handful of samples, which reads as
        # obviously wrong on a dashboard ("p95 higher than every request we've seen").
        p95 = statistics.quantiles(durations, n=100, method="inclusive")[94] if len(durations) >= 2 else durations[0]
        stats.append(
            ModelLatencyStats(
                model=model,
                avg_duration_ms=statistics.mean(durations),
                p95_duration_ms=p95,
                request_count=len(durations),
            )
        )
    return sorted(stats, key=lambda s: s.request_count, reverse=True)


def _latency_time_series(events: list[TelemetryEvent]) -> list[LatencyTimeBucket]:
    buckets: dict[str, list[float]] = {}
    for event in events:
        bucket_key = event.started_at.strftime("%Y-%m-%dT%H:00:00")
        buckets.setdefault(bucket_key, []).append(event.duration_ms)
    return [
        LatencyTimeBucket(bucket_start=key, avg_duration_ms=statistics.mean(vals), request_count=len(vals))
        for key, vals in sorted(buckets.items())
    ]


async def get_ollama_telemetry_summary(db: AsyncSession) -> OllamaTelemetrySummary:
    """The one entry point app/routers/telemetry.py calls."""
    total = (await db.execute(select(func.count(TelemetryEvent.id)))).scalar_one()
    errors = (
        await db.execute(select(func.count(TelemetryEvent.id)).where(TelemetryEvent.success.is_(False)))
    ).scalar_one()
    events = await _fetch_recent_events(db)
    return OllamaTelemetrySummary(
        total_requests=total,
        error_count=errors,
        by_model=await _counts_by_model(db),
        by_host=await _stats_by_host(db),
        latency_by_model=_latency_by_model(events),
        latency_time_series=_latency_time_series(events),
        loaded_models=await ollama_snapshot_service.get_latest_snapshots(db),
    )
