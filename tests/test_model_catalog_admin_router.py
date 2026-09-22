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
from app.services import extended_model_catalog_service, gguf_probe
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError
from app.services.model_catalog_service import HiddenModelTags

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


def _stub_repo_files(monkeypatch, repo: dict = _REPO) -> None:
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(repo))
    )


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """Same "couldn't reach Hugging Face" stub as tests/test_extended_model_catalog_service.py's own fixture
    (see its docstring) — add_extended_model here goes through the same ExtendedModelCatalog.add, which now
    also probes the real GGUF header; this keeps that probe from making a real network call in these
    router-level tests too."""

    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return None

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)


@pytest.mark.asyncio
async def test_get_extended_catalog_starts_empty(db, user):
    result = await model_catalog_admin.get_extended_catalog(db=db, user=user)
    assert result.entries == []


@pytest.mark.asyncio
async def test_search_hf_models_returns_results(db, admin_user, monkeypatch):
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "search",
        staticmethod(
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
            )
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

    monkeypatch.setattr(HuggingFaceCatalogSearch, "search", staticmethod(_raise))

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.search_hf_models(SearchHfModelsRequest(query="qwen2.5"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_get_hf_repo_files_returns_the_file_list(db, admin_user, monkeypatch):
    _stub_repo_files(monkeypatch)

    result = await model_catalog_admin.get_hf_repo_files(
        HfRepoFilesRequest(repo_id=_REPO["repo_id"]), db=db, _admin=admin_user
    )

    assert result.repo_id == _REPO["repo_id"]
    assert result.files[0].filename == "qwen2.5-14b-instruct-q4_k_m.gguf"


@pytest.mark.asyncio
async def test_get_hf_repo_files_returns_503_for_an_unknown_repo(db, admin_user, monkeypatch):
    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError('"org/nope" was not found on Hugging Face — check the repo is correct.')

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.get_hf_repo_files(HfRepoFilesRequest(repo_id="org/nope"), db=db, _admin=admin_user)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_add_extended_model_verifies_and_adds(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch)

    result = await model_catalog_admin.add_extended_model(
        AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"]),
        db=db,
        admin=admin_user,
    )

    assert len(result.entries) == 1
    assert result.entries[0].tag == ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    assert result.already_installed is False


@pytest.mark.asyncio
async def test_add_extended_model_flags_already_installed_and_omits_it_from_entries(db, admin_user, monkeypatch):
    """Confirmed live: adding a tag that's already installed used to come back looking identical to a genuine
    new addition (no way for the frontend to tell it apart from the entries list alone, since
    ExtendedModelCatalog.build excludes any installed tag on purpose — see its own docstring) — an admin who
    just re-added an already-installed model had nowhere left to find it and pull it, misled by a flat "Added"."""
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([{"name": tag}]))
    _stub_repo_files(monkeypatch)

    result = await model_catalog_admin.add_extended_model(
        AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"]),
        db=db,
        admin=admin_user,
    )

    assert result.already_installed is True
    assert result.entries == []


@pytest.mark.asyncio
async def test_add_extended_model_returns_400_for_a_duplicate(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch)
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

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))

    with pytest.raises(HTTPException) as exc_info:
        await model_catalog_admin.add_extended_model(
            AddExtendedModelRequest(repo_id="org/a", filename="x.gguf"), db=db, admin=admin_user
        )
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_remove_extended_model_removes_it(db, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch)
    body = AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"])
    await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])

    result = await model_catalog_admin.remove_extended_model(
        RemoveExtendedModelRequest(tag=tag), db=db, admin=admin_user
    )

    assert result.entries == []


@pytest.mark.asyncio
async def test_get_extended_catalog_hides_admin_hidden_entries_from_a_regular_user(db, user, admin_user, monkeypatch):
    monkeypatch.setattr(extended_model_catalog_service, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch)
    body = AddExtendedModelRequest(repo_id=_REPO["repo_id"], filename=_REPO["files"][0]["filename"])
    await model_catalog_admin.add_extended_model(body, db=db, admin=admin_user)
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])

    await HiddenModelTags(db).set({tag})

    regular_result = await model_catalog_admin.get_extended_catalog(db=db, user=user)
    admin_result = await model_catalog_admin.get_extended_catalog(db=db, user=admin_user)

    assert regular_result.entries == []
    assert len(admin_result.entries) == 1
