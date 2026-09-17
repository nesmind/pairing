"""Unit tests for app/services/ollama_snapshot_service.py — dedupes
poll history down to the newest row per (host, model_name), rather than
a version-fragile SQL window-function query (see that module's own
docstring)."""

from datetime import timedelta

import pytest

from app.models import OllamaModelSnapshot
from app.models._base import utcnow
from app.services import ollama_snapshot_service


@pytest.mark.asyncio
async def test_returns_only_the_newest_row_per_host_and_model(db):
    now = utcnow()
    db.add_all(
        [
            OllamaModelSnapshot(
                host="http://x:11434", model_name="llama3:latest", size_bytes=1, polled_at=now - timedelta(seconds=30)
            ),
            OllamaModelSnapshot(host="http://x:11434", model_name="llama3:latest", size_bytes=2, polled_at=now),
        ]
    )
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db)

    assert len(snapshots) == 1
    assert snapshots[0].size_bytes == 2


@pytest.mark.asyncio
async def test_distinguishes_different_hosts_and_models(db):
    now = utcnow()
    db.add_all(
        [
            OllamaModelSnapshot(host="http://a:11434", model_name="llama3:latest", polled_at=now),
            OllamaModelSnapshot(host="http://b:11434", model_name="llama3:latest", polled_at=now),
            OllamaModelSnapshot(host="http://a:11434", model_name="moondream:1.8b", polled_at=now),
        ]
    )
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db)

    keys = {(s.host, s.model_name) for s in snapshots}
    assert keys == {
        ("http://a:11434", "llama3:latest"),
        ("http://b:11434", "llama3:latest"),
        ("http://a:11434", "moondream:1.8b"),
    }


@pytest.mark.asyncio
async def test_excludes_snapshots_outside_the_lookback_window(db):
    stale = utcnow() - timedelta(hours=1)
    db.add(OllamaModelSnapshot(host="http://x:11434", model_name="llama3:latest", polled_at=stale))
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db)

    assert snapshots == []


@pytest.mark.asyncio
async def test_empty_when_nothing_polled_yet(db):
    assert await ollama_snapshot_service.get_latest_snapshots(db) == []
