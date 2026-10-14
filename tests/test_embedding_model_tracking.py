"""Knowledge-base documents remember their embedding model: sync re-embeds after a model switch (or always when
forced), search skips vectors from another model, and Settings > System > RAG can see and fix it."""

import numpy as np
import pytest
from sqlalchemy import select

from app.models import Chunk, Document
from app.routers import documents
from app.services import document_ingest, document_retrieval, document_sync, model_catalog_service
from app.services.document_embedding_status import DocumentEmbeddingStatus
from app.services.document_extract import vector_to_bytes


class FakeEngine:
    """Embeds with a size per model, counting calls; stands in for inference_client.embed."""

    SIZES = {"model-a": 3, "model-b": 3, "model-small": 2}

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, _text: str, model: str) -> list[float]:
        self.calls += 1
        return [1.0] * self.SIZES[model]


@pytest.fixture
def engine(monkeypatch, tmp_path):
    fake = FakeEngine()
    monkeypatch.setattr(document_ingest, "embed", fake.embed)
    monkeypatch.setattr(document_retrieval, "embed", fake.embed)
    monkeypatch.setattr(document_sync, "KNOWLEDGE_DIR", tmp_path)
    (tmp_path / "global").mkdir()
    (tmp_path / "global" / "notes.txt").write_text("Some notes about the hoopoe.")
    return fake


def _use_model(monkeypatch, model: str | None) -> None:
    async def current(_db):
        return model

    monkeypatch.setattr(model_catalog_service, "get_default_embedding_model", current)


async def _sync(db, tmp_path, force: bool = False) -> dict:
    return await document_sync._drain_sync_stats(tmp_path / "global", None, db, force)


async def _only_document(db) -> Document:
    return (await db.execute(select(Document))).scalars().one()


@pytest.mark.asyncio
async def test_sync_records_the_model_and_reembeds_after_a_switch(db, engine, monkeypatch, tmp_path):
    _use_model(monkeypatch, "model-a")
    assert (await _sync(db, tmp_path))["added"] == 1
    assert (await _only_document(db)).embedding_model == "model-a"

    calls = engine.calls
    assert (await _sync(db, tmp_path))["skipped"] == 1 and engine.calls == calls  # same model: nothing to do

    _use_model(monkeypatch, "model-b")
    assert (await _sync(db, tmp_path))["updated"] == 1
    assert (await _only_document(db)).embedding_model == "model-b"


@pytest.mark.asyncio
async def test_force_reembeds_even_with_the_same_model(db, engine, monkeypatch, tmp_path):
    _use_model(monkeypatch, "model-a")
    await _sync(db, tmp_path)
    assert (await _sync(db, tmp_path, force=True))["updated"] == 1


@pytest.mark.asyncio
async def test_an_unrecorded_model_is_left_alone_unless_forced(db, engine, monkeypatch, tmp_path):
    _use_model(monkeypatch, "model-a")
    await _sync(db, tmp_path)
    document = await _only_document(db)
    document.embedding_model = None  # as before this was recorded
    await db.commit()
    assert (await _sync(db, tmp_path))["skipped"] == 1


async def _add(db, name: str, model: str | None, vector: list[float]) -> None:
    document = Document(filename=f"global/{name}", content_hash=name, embedding_model=model, chunk_count=1)
    db.add(document)
    await db.flush()
    db.add(Chunk(document_id=document.id, text=f"text of {name}", embedding=vector_to_bytes(np.asarray(vector))))
    await db.commit()


@pytest.mark.asyncio
async def test_search_uses_only_vectors_from_the_current_model(db, engine, monkeypatch):
    _use_model(monkeypatch, "model-a")
    await _add(db, "current.txt", "model-a", [1.0, 0.0, 0.0])
    await _add(db, "other-model.txt", "model-b", [1.0, 0.0, 0.0])
    await _add(db, "unknown-same-size.txt", None, [0.0, 1.0, 0.0])
    await _add(db, "unknown-other-size.txt", None, [1.0, 1.0])

    results = await document_retrieval.retrieve_context(db, "question", None, top_k=10)

    assert sorted(r["filename"] for r in results) == ["current.txt", "unknown-same-size.txt"]


@pytest.mark.asyncio
async def test_status_counts_current_outdated_and_unknown(db, engine, monkeypatch):
    _use_model(monkeypatch, "model-a")
    await _add(db, "a.txt", "model-a", [1.0, 0.0, 0.0])
    await _add(db, "b.txt", "model-b", [1.0, 0.0, 0.0])
    await _add(db, "c.txt", None, [1.0, 0.0, 0.0])

    status = await DocumentEmbeddingStatus(db).get()

    assert (status.current_model, status.total, status.current, status.outdated, status.unknown) == (
        "model-a",
        3,
        1,
        1,
        1,
    )


@pytest.mark.asyncio
async def test_reembed_route_reports_the_tally_and_new_status(db, engine, monkeypatch):
    _use_model(monkeypatch, "model-a")

    async def fake_reembed_all(_db):
        return {"added": 0, "updated": 2, "removed": 0, "skipped": 0, "errors": []}

    monkeypatch.setattr(document_sync, "reembed_all", fake_reembed_all)
    result = await documents.reembed_all_documents(db=db, _admin=None)
    assert result.updated == 2 and result.status.current_model == "model-a"
