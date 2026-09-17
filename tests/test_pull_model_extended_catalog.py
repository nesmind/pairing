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
from app.services import extended_model_catalog_service


async def _async_return(value):
    return value


@pytest.mark.asyncio
async def test_pull_model_rejects_a_tag_in_neither_catalog(db, admin_user):
    with pytest.raises(HTTPException) as exc_info:
        await settings_router.pull_model(PullModelRequest(tag="nope:latest"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 400
    assert "Unknown model tag" in exc_info.value.detail


@pytest.mark.asyncio
async def test_pull_model_accepts_an_extended_catalog_tag(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return(
            {"family": "qwen2", "parameter_size": "14.0B", "files": [{"filename": "x.gguf", "download_gb": 9.0}]}
        ),
    )
    await extended_model_catalog_service.add_model(db, "Qwen/Qwen2.5-14B-Instruct-GGUF", "x.gguf", proxy_url=None)
    tag = extended_model_catalog_service.build_tag("Qwen/Qwen2.5-14B-Instruct-GGUF", "x.gguf")
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)

    result = await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)

    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_pull_model_accepts_an_embedding_catalog_tag(db, admin_user, monkeypatch):
    """Same entry-lookup gate as the extended-catalog case above, but for app/model_catalog.py's
    EMBEDDING_CATALOG (see pull_model's own three-way find_entry/find_entry/find_embedding_entry chain)."""
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 100.0)
    tag = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"

    result = await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)

    assert isinstance(result, StreamingResponse)


@pytest.mark.asyncio
async def test_pull_model_still_enforces_hardware_gating_for_an_extended_tag(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return(
            {"family": "qwen2", "parameter_size": "72B", "files": [{"filename": "x.gguf", "download_gb": 200.0}]}
        ),
    )
    await extended_model_catalog_service.add_model(db, "Qwen/Qwen2.5-72B-Instruct-GGUF", "x.gguf", proxy_url=None)
    tag = extended_model_catalog_service.build_tag("Qwen/Qwen2.5-72B-Instruct-GGUF", "x.gguf")
    monkeypatch.setattr(hardware, "available_capacity_gb", lambda: 16.0)

    with pytest.raises(HTTPException) as exc_info:
        await settings_router.pull_model(PullModelRequest(tag=tag), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 400
    assert "16.0GB" in exc_info.value.detail


@pytest.mark.asyncio
async def test_get_embedding_model_catalog_returns_both_entries(user, monkeypatch):
    """GET /api/settings/embedding-model-catalog — any logged-in user, not just an admin (see the endpoint's
    own docstring on why installed status isn't admin-secret)."""
    from app.services import model_catalog_service

    monkeypatch.setattr(model_catalog_service, "list_models", lambda: _async_return([]))

    response = await settings_router.get_embedding_model_catalog(_user=user)

    assert {e.tag for e in response.entries} == {
        "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0",
        "hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0",
    }
