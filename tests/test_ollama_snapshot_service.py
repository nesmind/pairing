"""Tests for services/ollama_snapshot_service.py: dedupes poll history to the newest row per (host, model_name),
one engine at a time (see the module on why queries filter on engine), instead of a version-fragile SQL window query.
"""

from datetime import timedelta

import pytest

from app.models import ModelSnapshot
from app.models._base import utcnow
from app.services import ollama_snapshot_service


@pytest.mark.asyncio
async def test_returns_only_the_newest_row_per_host_and_model(db):
    now = utcnow()
    db.add_all(
        [
            ModelSnapshot(
                host="http://x:11434", model_name="llama3:latest", size_bytes=1, polled_at=now - timedelta(seconds=30)
            ),
            ModelSnapshot(host="http://x:11434", model_name="llama3:latest", size_bytes=2, polled_at=now),
        ]
    )
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db, "ollama")

    assert len(snapshots) == 1
    assert snapshots[0].size_bytes == 2


@pytest.mark.asyncio
async def test_distinguishes_different_hosts_and_models(db):
    now = utcnow()
    db.add_all(
        [
            ModelSnapshot(host="http://a:11434", model_name="llama3:latest", polled_at=now),
            ModelSnapshot(host="http://b:11434", model_name="llama3:latest", polled_at=now),
            ModelSnapshot(host="http://a:11434", model_name="moondream:1.8b", polled_at=now),
        ]
    )
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db, "ollama")

    keys = {(s.host, s.model_name) for s in snapshots}
    assert keys == {
        ("http://a:11434", "llama3:latest"),
        ("http://b:11434", "llama3:latest"),
        ("http://a:11434", "moondream:1.8b"),
    }


@pytest.mark.asyncio
async def test_excludes_snapshots_outside_the_lookback_window(db):
    stale = utcnow() - timedelta(hours=1)
    db.add(ModelSnapshot(host="http://x:11434", model_name="llama3:latest", polled_at=stale))
    await db.commit()

    snapshots = await ollama_snapshot_service.get_latest_snapshots(db, "ollama")

    assert snapshots == []


@pytest.mark.asyncio
async def test_empty_when_nothing_polled_yet(db):
    assert await ollama_snapshot_service.get_latest_snapshots(db, "ollama") == []


@pytest.mark.asyncio
async def test_a_matricxon_snapshot_never_leaks_into_ollamas_results(db):
    """Both engines write to the same ModelSnapshot table (see the model): a Matricxon poll for the same host/model as
    an Ollama poll must not leak into Ollama's "currently loaded" dashboard, nor the reverse.
    """
    now = utcnow()
    db.add_all(
        [
            ModelSnapshot(engine="ollama", host="http://x:11434", model_name="llama3:latest", polled_at=now),
            ModelSnapshot(engine="matricxon", host="http://x:8420", model_name="ministral-3:3b", polled_at=now),
        ]
    )
    await db.commit()

    ollama_snapshots = await ollama_snapshot_service.get_latest_snapshots(db, "ollama")
    matricxon_snapshots = await ollama_snapshot_service.get_latest_snapshots(db, "matricxon")

    assert [s.model_name for s in ollama_snapshots] == ["llama3:latest"]
    assert [s.model_name for s in matricxon_snapshots] == ["ministral-3:3b"]
