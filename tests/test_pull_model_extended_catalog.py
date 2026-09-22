"""Unit test for app/routers/settings.py's pull_model endpoint accepting an extended-catalog tag, not just a
static app/model_catalog.py one (see that endpoint's own docstring) — called directly, not through a
TestClient/ASGI app (see tests/test_instance_proxy_http.py's own docstring on why this project doesn't use that
pattern). The full SSE download itself is exercised by app/services/ollama_admin.py's own tests; this only
checks the entry-lookup gate that decides whether pull_model proceeds at all."""

import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app import hardware
from app.routers import settings as settings_router
from app.schemas import PullModelRequest
from app.services import (
    default_model_settings,
    duplicate_install_detector,
    embedding_model_catalog_service,
    extended_model_catalog_service,
    gguf_probe,
)
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch


async def _async_return(value):
    return value


def _stub_repo_files(monkeypatch, repo: dict) -> None:
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(repo))
    )


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """Same "couldn't reach Hugging Face" stub as tests/test_extended_model_catalog_service.py's own fixture
    (see its docstring) — ExtendedModelCatalog.add below now also probes the real GGUF header; this keeps that
    from making a real network call here too."""

    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return None

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)


@pytest.mark.asyncio
async def test_pull_model_rejects_a_tag_in_neither_catalog(db, admin_user):
    with pytest.raises(HTTPException) as exc_info:
        await settings_router.pull_model(PullModelRequest(tag="nope:latest"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 400
    assert "Unknown model tag" in exc_info.value.detail


@pytest.mark.asyncio
async def test_pull_model_accepts_an_extended_catalog_tag(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(
        monkeypatch,
        {"family": "qwen2", "parameter_size": "14.0B", "files": [{"filename": "x.gguf", "download_gb": 9.0}]},
    )
    await ExtendedModelCatalog(db).add("Qwen/Qwen2.5-14B-Instruct-GGUF", "x.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("Qwen/Qwen2.5-14B-Instruct-GGUF", "x.gguf")
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)

    result = await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)

    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_pull_model_accepts_an_embedding_catalog_tag(db, admin_user, monkeypatch):
    """Same entry-lookup gate as the extended-catalog case above, but for app/model_catalog.py's
    CATALOG.embedding_models (see pull_model's own three-way CATALOG.find/ExtendedModelCatalog.find/
    CATALOG.find_embedding chain)."""
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)
    tag = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"

    result = await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)

    assert isinstance(result, StreamingResponse)


_MINISTRAL_TAG = "hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M"


@pytest.mark.asyncio
async def test_pull_model_returns_409_when_a_duplicate_is_already_installed(db, admin_user, monkeypatch):
    """Real regression guard (2026-09-21): moondream2-gguf and moondream-2b-2025-04-14-4bit turned out to be the
    exact same file installed under two different repo names — see DuplicateInstallDetector's own docstring."""
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)

    async def fake_list_models():
        return [{"name": "some-other-tag", "details": {"family": "mistral3"}, "size": 2_100_000_000}]

    monkeypatch.setattr(duplicate_install_detector, "list_models", fake_list_models)

    with pytest.raises(HTTPException) as exc_info:
        await settings_router.pull_model(PullModelRequest(tag=_MINISTRAL_TAG), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail["duplicate_of"] == "some-other-tag"


@pytest.mark.asyncio
async def test_pull_model_proceeds_when_the_duplicate_is_confirmed(db, admin_user, monkeypatch):
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)

    async def fake_list_models():
        return [{"name": "some-other-tag", "details": {"family": "mistral3"}, "size": 2_100_000_000}]

    monkeypatch.setattr(duplicate_install_detector, "list_models", fake_list_models)

    result = await settings_router.pull_model(
        PullModelRequest(tag=_MINISTRAL_TAG, confirm_duplicate=True), db=db, _admin=admin_user
    )

    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_pull_model_still_enforces_hardware_gating_for_an_extended_tag(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(
        monkeypatch,
        {"family": "qwen2", "parameter_size": "72B", "files": [{"filename": "x.gguf", "download_gb": 200.0}]},
    )
    await ExtendedModelCatalog(db).add("Qwen/Qwen2.5-72B-Instruct-GGUF", "x.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("Qwen/Qwen2.5-72B-Instruct-GGUF", "x.gguf")
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 16.0)

    with pytest.raises(HTTPException) as exc_info:
        await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 400
    assert "16.0GB" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_embedding_model_catalog_returns_both_entries(db, user, monkeypatch):
    """GET /api/settings/embedding-model-catalog — any logged-in user, not just an admin (see the endpoint's
    own docstring on why installed status isn't admin-secret)."""
    monkeypatch.setattr(embedding_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(default_model_settings, "list_models", lambda: _async_return([]))

    response = await settings_router.get_embedding_model_catalog(db=db, _user=user)

    assert {e.tag for e in response.entries} == {
        "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0",
    }
