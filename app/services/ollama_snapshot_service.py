"""Reads the Telemetry page's "currently loaded models" table from the poll history ollama_ps_poller/
matricxon_ps_poller write. Both engines share the ModelSnapshot table (see the model), so every query filters
by `engine`, one dashboard at a time (as telemetry_service for TelemetryEvent). It avoids a GROUP BY host,
model_name + MAX(polled_at) window query (version-fragile like a SQL percentile, see telemetry_service):
it fetches the last couple of poll intervals (always a tiny set) and dedupes to the newest per (host, model_name) in
Python.
"""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import ModelSnapshot
from app.models._base import utcnow
from app.schemas import LoadedModelSnapshot
from app.services.ollama_ps_poller import POLL_INTERVAL_SECONDS

# A few poll intervals of lookback: tolerates one slow or skipped poll without pulling in stale history.
_LOOKBACK = timedelta(seconds=POLL_INTERVAL_SECONDS * 3)


async def get_latest_snapshots(db: AsyncSession, engine: str) -> list[LoadedModelSnapshot]:
    cutoff = utcnow() - _LOOKBACK
    rows = (
        (
            await db.execute(
                select(ModelSnapshot)
                .where(ModelSnapshot.engine == engine, ModelSnapshot.polled_at >= cutoff)
                .order_by(ModelSnapshot.polled_at.desc())
            )
        )
        .scalars()
        .all()
    )
    newest_per_key: dict[tuple[str, str], ModelSnapshot] = {}
    for row in rows:
        key = (row.host, row.model_name)
        if key not in newest_per_key:
            newest_per_key[key] = row
    return [
        LoadedModelSnapshot(
            host=row.host,
            model_name=row.model_name,
            size_bytes=row.size_bytes,
            size_vram_bytes=row.size_vram_bytes,
            expires_at=row.expires_at,
            polled_at=row.polled_at,
        )
        for row in newest_per_key.values()
    ]
