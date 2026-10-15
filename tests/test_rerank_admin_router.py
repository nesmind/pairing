"""routers/rerank_admin.py: reading and choosing the Knowledge base reranker."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.routers import rerank_admin
from app.schemas.rerank import RerankModelUpdate
from app.services.default_model_settings import DefaultModelSettings


@pytest.fixture
def matricxon_like(monkeypatch):
    engine = SimpleNamespace(capabilities=SimpleNamespace(rerank=True))
    monkeypatch.setattr(rerank_admin.registry, "active", lambda: engine)

    async def installed(_capability):
        return ["rr:latest"]

    monkeypatch.setattr(DefaultModelSettings, "_installed_with_capability", staticmethod(installed))


@pytest.mark.asyncio
async def test_an_installed_reranker_is_used_by_default(db, matricxon_like):
    setting = await rerank_admin.get_rerank_model(db=db, _admin=None)
    assert (setting.model, setting.enabled, setting.installed) == ("rr:latest", True, ["rr:latest"])


@pytest.mark.asyncio
async def test_switching_off(db, matricxon_like):
    setting = await rerank_admin.set_rerank_model(RerankModelUpdate(model=None), db=db, _admin=None)
    assert (setting.model, setting.enabled) == (None, False)


@pytest.mark.asyncio
async def test_choosing_a_model_that_is_not_installed_is_refused(db, matricxon_like):
    with pytest.raises(HTTPException) as exc:
        await rerank_admin.set_rerank_model(RerankModelUpdate(model="nope"), db=db, _admin=None)
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_an_engine_without_rerank_reports_nothing(db, monkeypatch):
    engine = SimpleNamespace(capabilities=SimpleNamespace(rerank=False))
    monkeypatch.setattr(rerank_admin.registry, "active", lambda: engine)
    setting = await rerank_admin.get_rerank_model(db=db, _admin=None)
    assert (setting.engine_supports, setting.model, setting.installed) == (False, None, [])


QWEN3_RERANKER = "hf.co/ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF:Q8_0"


def test_the_curated_reranker_is_in_the_catalog():
    from app.model_catalog import CATALOG

    entry = CATALOG.find_rerank(QWEN3_RERANKER)
    assert entry is not None and entry.architecture == "qwen3" and entry.locally_runnable


@pytest.mark.asyncio
async def test_catalog_marks_an_installed_reranker_as_in_use(db, monkeypatch):
    engine = SimpleNamespace(capabilities=SimpleNamespace(rerank=True))
    monkeypatch.setattr(rerank_admin.registry, "active", lambda: engine)
    from app.services import rerank_model_catalog_service

    monkeypatch.setattr(rerank_model_catalog_service.registry, "active", lambda: engine)

    async def installed(_capability):
        return [QWEN3_RERANKER]

    monkeypatch.setattr(DefaultModelSettings, "_installed_with_capability", staticmethod(installed))
    catalog = await rerank_admin.get_rerank_model_catalog(db=db, _admin=None)

    entry = next(e for e in catalog.entries if e.tag == QWEN3_RERANKER)
    assert (catalog.engine_supports, entry.installed, entry.in_use) == (True, True, True)


@pytest.mark.asyncio
async def test_catalog_without_rerank_support_shows_nothing_installed(db, monkeypatch):
    engine = SimpleNamespace(capabilities=SimpleNamespace(rerank=False))
    from app.services import rerank_model_catalog_service

    monkeypatch.setattr(rerank_model_catalog_service.registry, "active", lambda: engine)
    catalog = await rerank_admin.get_rerank_model_catalog(db=db, _admin=None)

    assert catalog.engine_supports is False
    assert not any(e.installed or e.in_use for e in catalog.entries)
