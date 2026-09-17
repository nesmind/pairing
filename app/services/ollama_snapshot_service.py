"""Reads back the "currently loaded models" table for the Telemetry
page from the raw poll history app.services.ollama_ps_poller writes.
Deliberately avoids a GROUP BY host, model_name + MAX(polled_at)
window-function query (version-fragile the same way a SQL percentile
would be, see app/services/telemetry_service.py's own docstring) —
instead fetches the last couple of poll intervals' worth of rows (a
result set that's always tiny: a handful of models across a handful of
hosts) and dedupes to the newest per (host, model_name) in Python.
"""

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import OllamaModelSnapshot
from app.models._base import utcnow
from app.schemas import LoadedModelSnapshot
from app.services.ollama_ps_poller import POLL_INTERVAL_SECONDS

# A few poll intervals' worth of lookback — enough to tolerate one slow
# or skipped poll without losing "currently loaded" visibility, without
# ever pulling in stale history.
_LOOKBACK = timedelta(seconds=POLL_INTERVAL_SECONDS * 3)


async def get_latest_snapshots(db: AsyncSession) -> list[LoadedModelSnapshot]:
    cutoff = utcnow() - _LOOKBACK
    rows = (
        (
            await db.execute(
                select(OllamaModelSnapshot)
                .where(OllamaModelSnapshot.polled_at >= cutoff)
                .order_by(OllamaModelSnapshot.polled_at.desc())
            )
        )
        .scalars()
        .all()
    )
    newest_per_key: dict[tuple[str, str], OllamaModelSnapshot] = {}
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
