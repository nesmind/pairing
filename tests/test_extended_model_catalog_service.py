"""Unit tests for app/services/extended_model_catalog_service.py. Every network-touching dependency
(list_models, get_repo_files) is monkeypatched on this module's own imported bindings (not their origin modules —
see the "module-split import-binding gotcha" this codebase already tracks), so no real Ollama or Hugging Face
call happens here."""

import pytest

from app.services import extended_model_catalog_service as svc
from app.services.huggingface_client import HuggingFaceLookupError
from app.services.model_catalog_service import set_hidden_tags

_REPO = {
    "repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
    "family": "qwen2",
    "parameter_size": "14.0B",
    "context_length": 32768,
    "gated": False,
    "license": "apache-2.0",
    "files": [{"filename": "qwen2.5-14b-instruct-q4_k_m.gguf", "download_gb": 9.0}],
}


def test_build_tag_uses_the_full_filename_minus_extension():
    tag = svc.build_tag("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf")
    assert tag == "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m"


@pytest.mark.asyncio
async def test_get_extended_catalog_defaults_to_empty(db):
    assert await svc.get_extended_catalog(db) == []


@pytest.mark.asyncio
async def test_add_model_verifies_and_stores_a_new_entry(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(svc, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO))

    entry = await svc.add_model(db, _REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry == {
        "tag": "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m",
        "family": "qwen2",
        "parameter_size": "14.0B",
        "download_gb": 9.0,
        "note": None,
    }
    assert await svc.get_extended_catalog(db) == [entry]


@pytest.mark.asyncio
async def test_add_model_raises_when_the_filename_is_no_longer_in_the_repo(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(svc, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO))

    with pytest.raises(ValueError, match="no longer available"):
        await svc.add_model(db, _REPO["repo_id"], "does-not-exist.gguf", proxy_url=None)
    assert await svc.get_extended_catalog(db) == []


@pytest.mark.asyncio
async def test_add_model_skips_verification_when_already_installed(db, monkeypatch):
    tag = svc.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("get_repo_files should never be called for an already-installed tag")

    monkeypatch.setattr(svc, "get_repo_files", _fail_if_called)

    entry = await svc.add_model(db, _REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry["tag"] == tag
    assert entry["note"] == svc.ALREADY_INSTALLED_NOTE
    assert entry["download_gb"] is None


@pytest.mark.asyncio
async def test_add_model_rejects_a_duplicate_tag(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(svc, "get_repo_files", lambda repo_id, proxy_url: _async_return(_REPO))
    await svc.add_model(db, _REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    with pytest.raises(ValueError, match="already in the catalog"):
        await svc.add_model(db, _REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)


@pytest.mark.asyncio
async def test_add_model_propagates_a_huggingface_lookup_error(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError('"nope/nope" was not found on Hugging Face — check the repo is correct.')

    monkeypatch.setattr(svc, "get_repo_files", _raise)

    with pytest.raises(HuggingFaceLookupError):
        await svc.add_model(db, "nope/nope", "nope.gguf", proxy_url=None)
    assert await svc.get_extended_catalog(db) == []


@pytest.mark.asyncio
async def test_remove_model_removes_only_the_given_tag(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        svc,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return(
            {**_REPO, "repo_id": repo_id, "files": [{"filename": "x.gguf", "download_gb": 1.0}]}
        ),
    )
    await svc.add_model(db, "org/a", "x.gguf", proxy_url=None)
    await svc.add_model(db, "org/b", "x.gguf", proxy_url=None)
    tag_a = svc.build_tag("org/a", "x.gguf")
    tag_b = svc.build_tag("org/b", "x.gguf")

    await svc.remove_model(db, tag_a)

    tags = {e["tag"] for e in await svc.get_extended_catalog(db)}
    assert tags == {tag_b}


@pytest.mark.asyncio
async def test_find_entry_returns_none_for_an_unknown_tag(db):
    assert await svc.find_entry(db, "hf.co/nope/nope:nope") is None


@pytest.mark.asyncio
async def test_find_entry_returns_a_pull_model_compatible_shape(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        svc,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return({**_REPO, "files": [{"filename": "x.gguf", "download_gb": 10.0}]}),
    )
    await svc.add_model(db, "org/a", "x.gguf", proxy_url=None)
    tag = svc.build_tag("org/a", "x.gguf")

    entry = await svc.find_entry(db, tag)

    assert entry["locally_runnable"] is True
    assert entry["min_ram_gb"] == pytest.approx(12.5)  # 10.0 * 1.25, same heuristic as app/model_catalog.py
    assert entry["unavailable_reason"] is None


@pytest.mark.asyncio
async def test_build_extended_catalog_marks_hardware_gated_entries(db, user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        svc,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return(
            {**_REPO, "parameter_size": "72B", "files": [{"filename": "x.gguf", "download_gb": 40.0}]}
        ),
    )
    await svc.add_model(db, "org/big", "x.gguf", proxy_url=None)
    monkeypatch.setattr(svc.hardware, "available_capacity_gb", lambda: 16.0)

    entries = await svc.build_extended_catalog(db, user)

    assert len(entries) == 1
    assert entries[0].tag == svc.build_tag("org/big", "x.gguf")
    assert entries[0].installed is False
    assert entries[0].hardware_ok is False  # 40GB * 1.25 = 50GB needed, only 16GB available


@pytest.mark.asyncio
async def test_build_extended_catalog_marks_a_not_installed_entry_removable_for_admin_only(
    db, user, admin_user, monkeypatch
):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        svc,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return({**_REPO, "files": [{"filename": "x.gguf", "download_gb": 1.0}]}),
    )
    await svc.add_model(db, "org/a", "x.gguf", proxy_url=None)

    admin_entries = await svc.build_extended_catalog(db, admin_user)
    regular_entries = await svc.build_extended_catalog(db, user)

    assert admin_entries[0].removable is True
    assert regular_entries[0].removable is False


@pytest.mark.asyncio
async def test_build_extended_catalog_omits_an_installed_entry_entirely(db, admin_user, monkeypatch):
    """An installed extended-catalog entry is left out of this response entirely (not shown with installed=True)
    — app.services.model_catalog_service.build_model_catalog's own "installed but not in the static catalog"
    branch already surfaces it in the *default* list once pulled, so showing it here too would duplicate the
    same row across both lists (see build_extended_catalog's own docstring)."""
    tag = svc.build_tag("org/a", "x.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag, "capabilities": ["completion"]}]))

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("get_repo_files should never be called for an already-installed tag")

    monkeypatch.setattr(svc, "get_repo_files", _fail_if_called)
    await svc.add_model(db, "org/a", "x.gguf", proxy_url=None)  # already installed at add time -> skips lookup too

    entries = await svc.build_extended_catalog(db, admin_user)

    assert entries == []
    # Still stored, though — uninstalling later would make it reappear here (not exercised in this unit test,
    # since that only depends on list_models()'s live result at read time, already covered by the "not
    # installed" tests above).
    assert [e["tag"] for e in await svc.get_extended_catalog(db)] == [tag]


@pytest.mark.asyncio
async def test_build_extended_catalog_hides_admin_hidden_tags_from_a_regular_user(db, user, admin_user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        svc,
        "get_repo_files",
        lambda repo_id, proxy_url: _async_return({**_REPO, "files": [{"filename": "x.gguf", "download_gb": 1.0}]}),
    )
    await svc.add_model(db, "org/hidden", "x.gguf", proxy_url=None)
    tag = svc.build_tag("org/hidden", "x.gguf")
    await set_hidden_tags(db, {tag})

    regular_entries = await svc.build_extended_catalog(db, user)
    admin_entries = await svc.build_extended_catalog(db, admin_user)

    assert regular_entries == []
    assert len(admin_entries) == 1
    assert admin_entries[0].hidden is True


async def _async_return(value):
    return value
