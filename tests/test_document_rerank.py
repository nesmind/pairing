"""Knowledge base reranking: document_rerank.ChunkReranker, the reranker setting
(DefaultModelSettings.rerank) and Matricxon's /api/rerank client."""

from types import SimpleNamespace

import httpx
import pytest

from app.services import document_rerank, matricxon_pool
from app.services.default_model_settings import DefaultModelSettings
from app.services.document_rerank import ChunkReranker
from app.services.inference_client import InferenceError
from app.services.matricxon_rerank import MatricxonRerankClient


def _chunks(*texts: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(text=text) for text in texts]


def _engine(monkeypatch, rerank: bool) -> None:
    engine = SimpleNamespace(capabilities=SimpleNamespace(rerank=rerank))
    monkeypatch.setattr(document_rerank.registry, "active", lambda: engine)


def _model(monkeypatch, model: str | None) -> None:
    async def fake_rerank(_self):
        return model

    monkeypatch.setattr(DefaultModelSettings, "rerank", fake_rerank)


@pytest.mark.asyncio
async def test_the_reranker_picks_the_best_top_k(db, monkeypatch):
    _engine(monkeypatch, rerank=True)
    _model(monkeypatch, "reranker:latest")

    async def fake_scores(_query, documents, _model):
        return [{"a": 0.1, "b": 0.9, "c": 0.5}[d] for d in documents]

    monkeypatch.setattr(document_rerank, "rerank", fake_scores)
    best = await ChunkReranker(db).order("q", _chunks("a", "b", "c"), top_k=2)

    assert [c.text for c in best] == ["b", "c"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("engine_reranks", "model"), [(False, "reranker:latest"), (True, None)])
async def test_without_a_reranker_the_similarity_order_is_kept(db, monkeypatch, engine_reranks, model):
    _engine(monkeypatch, rerank=engine_reranks)
    _model(monkeypatch, model)

    async def must_not_call(*_a):
        pytest.fail("rerank must not be called")

    monkeypatch.setattr(document_rerank, "rerank", must_not_call)
    best = await ChunkReranker(db).order("q", _chunks("a", "b", "c"), top_k=2)

    assert [c.text for c in best] == ["a", "b"]


@pytest.mark.asyncio
async def test_a_failed_rerank_keeps_the_similarity_order(db, monkeypatch):
    _engine(monkeypatch, rerank=True)
    _model(monkeypatch, "reranker:latest")

    async def failing(*_a):
        raise InferenceError("down")

    monkeypatch.setattr(document_rerank, "rerank", failing)
    best = await ChunkReranker(db).order("q", _chunks("a", "b", "c"), top_k=2)

    assert [c.text for c in best] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_setting_falls_back_to_an_installed_reranker_and_can_be_switched_off(db, monkeypatch):
    async def installed(_capability):
        return ["rr-a:latest", "rr-b:latest"]

    monkeypatch.setattr(DefaultModelSettings, "_installed_with_capability", staticmethod(installed))
    settings = DefaultModelSettings(db)

    assert await settings.rerank() == "rr-a:latest"
    await settings.set_rerank("rr-b:latest")
    assert await settings.rerank() == "rr-b:latest"
    await settings.set_rerank(None)
    assert await settings.rerank() is None
    assert await settings.rerank_enabled() is False


class _FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        # Best first, as Matricxon returns them.
        return {"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}]}


class _FakeClient:
    def __init__(self, calls: list) -> None:
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url, json):
        self._calls.append((url, json))
        return _FakeResponse()


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_matricxon_client_returns_scores_in_input_order(monkeypatch):
    calls: list = []
    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://mx:8420")
    monkeypatch.setattr(matricxon_pool, "track_request", lambda _host: _NullCtx())
    monkeypatch.setattr(matricxon_pool, "mark_recovered", lambda _host: None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient(calls))

    scores = await MatricxonRerankClient().rerank("q", ["d0", "d1"], "rr:latest")

    assert scores == [0.2, 0.9]
    assert calls[0][0] == "http://mx:8420/api/rerank"
    assert calls[0][1] == {"model": "rr:latest", "query": "q", "documents": ["d0", "d1"]}
