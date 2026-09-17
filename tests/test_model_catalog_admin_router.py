"""Unit tests for app/routers/model_catalog_admin.py — called directly, not through a TestClient/ASGI app (see
tests/test_instance_proxy_http.py's own docstring on why this project doesn't use that pattern)."""

import pytest
from fastapi import HTTPException

from app.routers import model_catalog_admin
from app.schemas import (
    AddExtendedModelRequest,
    HfRepoFilesRequest,
    RemoveExtendedModelRequest,
    SearchHfModelsRequest,
)
from app.services import extended_model_catalog_service
from app.services.huggingface_client import HuggingFaceLookupError

_REPO = {
    "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
    "family": "qwen2",
    "parameter_size": "14.0B",
    "context_length": 32768,
    "gated": False,
    "license": "apache-2.0",
    "files": [{"filename": "qwen2.5-14b-instruct-q4_k_m.gguf", "download_gb": 9.0}],
}


async def _async_return(value):
    return value


@pytest.mark.asyncio
async def test_get_extended_catalog_starts_empty(db, user):
    result = await model_catalog_admin.get_extended_catalog(db=db, user=user)
    assert result.entries == []


@pytest.mark.asyncio
async def test_search_hf_models_returns_results(db, admin_user, monkeypatch):
    monkeypatch.setattr(
        model_catalog_admin,
        "search_models",
        lambda query, proxy_url: _async_return(
            [
                {
                    "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
                    "downloads": 100,
                    "likes": 5,
                    "gated": False,
                    "license": None,
                }
            ]
        ),
    )

    result = await model_catalog_admin.search_hf_models(
        SearchHfModelsRequest(query="qwen2.5"), db=db, _admin=admin_user
    )

    assert len(result.results) == 1
    assert result.results[0].repo_id == "Qwen/Qwen2.5-14B-Instruct-GGUF"


@pytest.mark.asyncio
async def test_search_hf_models_returns_503_when_hugging_face_is_unreachable(db, admin_user, monkeypatch):
    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError("Could not reach Hugging Face: check your internet connection.")

    monkeypatch.setattr(model_catalog_admin, "search_models", _raise)

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.search_hf_models(SearchHfModelsRequest(query="qwen2.5"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_get_hf_repo_files_returns_the_file_list(db, admin_user, monkeypatch):
    monkeypatch.setattr(model_catalog_admin, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO))

    result = await model_catalog_admin.get_hf_repo_files(
        HfRepoFilesRequest(repo_id=_REPO["repo_id"]), db=db, _admin=admin_user
    )

    assert result.repo_id == _REPO["repo_id"]
    assert result.files[0].filename == "qwen2.5-14b-instruct-q4_k_m.gguf"


@pytest.mark.asyncio
async def test_get_hf_repo_files_returns_503_for_an_unknown_repo(db, admin_user, monkeypatch):
    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError('"org/nope" was not found on Hugging Face — check the repo is correct.')

    monkeypatch.setattr(model_catalog_admin, "get_repo_files", _raise)

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.get_hf_repo_files(HfRepoFilesRequest(repo_id="org/nope"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_add_extended_model_verifies_and_adds(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO)
    )

    result = await model_catalog_admin.add_extended_model(
        AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"]),
        db=db,
        admin=admin_user,
    )

    assert len(result.entries) == 1
    assert result.entries[0].tag == extended_model_catalog_service.build_tag(
        _REPO["repo_id"], _REPO["files"][0]["filename"]
    )


@pytest.mark.asyncio
async def test_add_extended_model_returns_400_for_a_duplicate(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO)
    )
    body = AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"])
    await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_add_extended_model_returns_503_when_hugging_face_is_unreachable(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError("Could not reach Hugging Face: check your internet connection.")

    monkeypatch.setattr(extended_model_catalog_service, "get_repo_files", _raise)

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.add_extended_model(
            AddExtendedModelRequest(repo_id="org/a", filename="x.gguf"), db=db, admin=admin_user
        )
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_remove_extended_model_removes_it(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO)
    )
    body = AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"])
    await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)
    tag = extended_model_catalog_service.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])

    result = await model_catalog_admin.remove_extended_model(
        RemoveExtendedModelRequest(tag=tag), db=db, admin=admin_user
    )

    assert result.entries == []


@pytest.mark.asyncio
async def test_get_extended_catalog_hides_admin_hidden_entries_from_a_regular_user(db, user, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        extended_model_catalog_service, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO)
    )
    body = AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"])
    await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)
    tag = extended_model_catalog_service.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])

    from app.services.model_catalog_service import set_hidden_tags

    await set_hidden_tags(db, {tag})

    regular_result = await model_catalog_admin.get_extended_catalog(db=db, user=user)
    admin_result = await model_catalog_admin.get_extended_catalog(db=db, user=admin_user)

    assert regular_result.entries == []
    assert len(admin_result.entries) == 1
