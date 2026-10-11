"""Tests for services/extended_model_catalog_service.py. Network dependencies (list_models,
HuggingFaceCatalogSearch.repo_files) are patched on this module's binding, or (repo_files, a staticmethod)
on the shared class, not origin modules (module-split gotcha), so no real Ollama or Hugging Face call.
"""

import pytest

from app.services import engine_service, extended_model_catalog_service as svc, gguf_probe
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError
from app.services.model_catalog_service import HiddenModelTags


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


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


def _stub_repo_files(monkeypatch, repo: dict) -> None:
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(repo))
    )


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """Every test gets a "couldn't reach Hugging Face" GGUF probe by default (the None probe_metadata returns on a
    network failure), so add falls back to repo-level metadata and filename heuristics. Patched at the origin
    module since HuggingFaceModelProbe.probe calls `gguf_probe.probe_metadata(...)` at call time; filename-only
    quantization guessing still runs. Tests wanting real-probe behavior override it.
    """

    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return None

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)


def test_build_tag_uses_the_full_filename_minus_extension():
    tag = ExtendedModelCatalog.build_tag("Qwen/Qwen2.5-14B-Instruct-GGUF", "qwen2.5-14b-instruct-q4_k_m.gguf")
    assert tag == "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m"


@pytest.mark.parametrize(
    "repo_id,expected",
    [
        ("Qwen/Qwen2.5-14B-Instruct-GGUF", "Qwen2.5-14B-Instruct"),
        ("nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF", "NVIDIA-Nemotron-3-Nano-4B"),
        ("moondream/moondream-2b-2025-04-14-4bit", "moondream-2b-2025-04-14-4bit"),
        ("org/GGUF", "GGUF"),  # stripping the whole name would leave nothing to show: keep it
    ],
)
def test_repo_display_name_strips_the_noisy_gguf_suffix(repo_id, expected):
    assert ExtendedModelCatalog._repo_display_name(repo_id) == expected


@pytest.mark.asyncio
async def test_list_defaults_to_empty(db):
    assert await ExtendedModelCatalog(db).list() == []


@pytest.mark.asyncio
async def test_add_verifies_and_stores_a_new_entry(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, _REPO)
    engine_service._cached_engine = "ollama"
    catalog = ExtendedModelCatalog(db)

    entry = await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry == {
        "tag": "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m",
        # The repo's name (trailing "-GGUF" stripped, _repo_display_name), not repo["family"] ("qwen2"): in-file
        # metadata now serves only the live Matricxon-support check, not display (it can mislead; see add).
        "family": "Qwen2.5-14B-Instruct",
        "parameter_size": "14.0B",
        "download_gb": 9.0,
        "is_projector": False,
        "vision": False,
        # None: the stubbed header probe "failed" like a network hiccup. quantizations is pure filename-only guessing
        # (HuggingFaceModelProbe.guess_quantizations_from_filename), unaffected by the stub.
        "architecture": None,
        "quantizations": ["Q4_K"],
        "note": None,
        # Whichever engine was active at add time: every entry gets this, including a real admin search/add like this
        # one.
        "discovered_via_engine": "ollama",
    }
    assert await catalog.list() == [entry]


@pytest.mark.asyncio
async def test_add_labels_a_projector_file_with_the_repo_name_instead_of_inheriting_the_repo_family(db, monkeypatch):
    """Confirmed live twice: (1) a "moondream2" mmproj was added with its sibling text model's repo-level "phi2"/"1.4B"
    (repo_files' family/parameter_size describe one file per repo), looking like a 1.4B Phi-2 chat model; (2) a
    bare "Vision projector" label was just as unidentifiable with several admin-added projectors: the repo name has to
    be in it.
    """
    repo = {
        **_REPO,
        "repo_id": "moondream/moondream-2b-2025-04-14-4bit",
        "family": "phi2",
        "parameter_size": "1.4B",
        "files": [{"filename": "mmproj-f16.gguf", "download_gb": 0.9, "is_projector": True}],
    }
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, repo)

    entry = await ExtendedModelCatalog(db).add(repo["repo_id"], "mmproj-f16.gguf", proxy_url=None)

    # No "(vision projector)" suffix any more (a badge: renderModelRow's is_projector in settings.js), but the repo name
    # still identifies *which* projector.
    assert entry["family"] == "moondream-2b-2025-04-14-4bit"
    assert entry["parameter_size"] is None
    assert entry["download_gb"] == 0.9
    assert entry["is_projector"] is True
    assert entry["vision"] is False


@pytest.mark.asyncio
async def test_add_flags_a_model_file_whose_repo_ships_a_projector_as_vision(db, user, monkeypatch):
    repo = {
        **_REPO,
        "files": [
            {"filename": "model-q4_k_m.gguf", "download_gb": 2.5, "is_projector": False},
            {"filename": "mmproj-f16.gguf", "download_gb": 0.9, "is_projector": True},
        ],
    }
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, repo)
    catalog = ExtendedModelCatalog(db)

    entry = await catalog.add(repo["repo_id"], "model-q4_k_m.gguf", proxy_url=None)

    assert entry["vision"] is True
    assert (await catalog.build(user))[0].vision is True


@pytest.mark.asyncio
async def test_add_raises_when_the_filename_is_no_longer_in_the_repo(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, _REPO)
    catalog = ExtendedModelCatalog(db)

    with pytest.raises(ValueError, match="no longer available"):
        await catalog.add(_REPO["repo_id"], "does-not-exist.gguf", proxy_url=None)
    assert await catalog.list() == []


@pytest.mark.asyncio
async def test_add_still_enriches_an_already_installed_tag_when_the_lookup_succeeds(db, monkeypatch):
    """Bug: an already-installed tag skipped the Hugging Face lookup, leaving null
    family/parameter_size
    and, once displayed, Matricxon's generic family (e.g. "Llama") instead of "Llama-3.2-3B-Instruct". It now gets
    the same lookup as any tag; `note` still marks it already-installed (no download), the rest is real.
    """
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))
    _stub_repo_files(monkeypatch, _REPO)

    entry = await ExtendedModelCatalog(db).add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry["tag"] == tag
    assert entry["note"] == ExtendedModelCatalog.ALREADY_INSTALLED_NOTE
    assert entry["family"] == "Qwen2.5-14B-Instruct"
    assert entry["parameter_size"] == "14.0B"
    assert entry["download_gb"] == 9.0


@pytest.mark.asyncio
async def test_add_falls_back_to_the_no_internet_lookup_shape_when_already_installed_and_the_lookup_fails(
    db, monkeypatch
):
    """The safety net the old skip existed for: silencing "not in catalog" for something on disk must never need
    connectivity. A HuggingFaceLookupError (network down, repo gone, rate limit) falls back to the null-data-plus-note
    shape.
    """
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError("network unreachable")

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))

    entry = await ExtendedModelCatalog(db).add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry["tag"] == tag
    assert entry["note"] == ExtendedModelCatalog.ALREADY_INSTALLED_NOTE
    assert entry["family"] is None
    assert entry["download_gb"] is None


@pytest.mark.asyncio
async def test_add_re_adding_an_already_installed_tag_is_a_harmless_no_op(db, monkeypatch):
    """Confirmed live: re-searching a repo both installed and catalogued used to dead-end on "already in the catalog"
    with no way to see or remove the stale entry (build hides installed tags). Re-adding must succeed and, as an
    installed tag gets the real lookup (test_add_still_enriches_an_already_installed_tag_when_the_lookup_succeeds),
    refresh a stale entry.
    """
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, _REPO)
    catalog = ExtendedModelCatalog(db)
    await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))
    updated_repo = {**_REPO, "parameter_size": "14.2B"}
    _stub_repo_files(monkeypatch, updated_repo)

    entry = await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry["note"] == ExtendedModelCatalog.ALREADY_INSTALLED_NOTE
    stored = await catalog.list()
    assert [e["tag"] for e in stored] == [tag]
    assert stored[0]["parameter_size"] == "14.2B"


@pytest.mark.asyncio
async def test_add_rejects_a_duplicate_tag(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, _REPO)
    catalog = ExtendedModelCatalog(db)
    await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    with pytest.raises(ValueError, match="already in the catalog"):
        await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)


@pytest.mark.asyncio
async def test_add_propagates_a_huggingface_lookup_error(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError('"nope/nope" was not found on Hugging Face — check the repo is correct.')

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))
    catalog = ExtendedModelCatalog(db)

    with pytest.raises(HuggingFaceLookupError):
        await catalog.add("nope/nope", "nope.gguf", proxy_url=None)
    assert await catalog.list() == []


@pytest.mark.asyncio
async def test_remove_removes_only_the_given_tag(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(
            lambda repo_id, proxy_url: _async_return(
                {**_REPO, "repo_id": repo_id, "files": [{"filename": "x.gguf", "download_gb": 1.0}]}
            )
        ),
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)
    await catalog.add("org/b", "x.gguf", proxy_url=None)
    tag_a = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    tag_b = ExtendedModelCatalog.build_tag("org/b", "x.gguf")

    await catalog.remove(tag_a)

    tags = {e["tag"] for e in await catalog.list()}
    assert tags == {tag_b}


@pytest.mark.asyncio
async def test_find_returns_none_for_an_unknown_tag(db):
    assert await ExtendedModelCatalog(db).find("hf.co/nope/nope:nope") is None


@pytest.mark.asyncio
async def test_find_returns_a_pull_model_compatible_shape(db, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, {**_REPO, "files": [{"filename": "x.gguf", "download_gb": 10.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("org/a", "x.gguf")

    entry = await catalog.find(tag)

    assert entry.locally_runnable is True
    assert entry.min_ram_gb == pytest.approx(12.5)
    assert entry.unavailable_reason is None


@pytest.mark.asyncio
async def test_build_marks_hardware_gated_entries(db, user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(
        monkeypatch, {**_REPO, "parameter_size": "72B", "files": [{"filename": "x.gguf", "download_gb": 40.0}]}
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/big", "x.gguf", proxy_url=None)
    monkeypatch.setattr(svc.hardware, "available_capacity_gb", lambda: 16.0)

    entries = await catalog.build(user)

    assert len(entries) == 1
    assert entries[0].tag == ExtendedModelCatalog.build_tag("org/big", "x.gguf")
    assert entries[0].installed is False
    assert entries[0].hardware_ok is False
    assert entries[0].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_gives_a_projector_its_own_accurate_unsupported_reason(db, user, monkeypatch):
    """A vision projector was never a standalone chat model, so the generic "architecture unknown, not verified yet"
    reason of other admin-added entries misleads: it implies verification might say True, which never happens here.
    """
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(
        monkeypatch, {**_REPO, "files": [{"filename": "mmproj.gguf", "download_gb": 0.9, "is_projector": True}]}
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "mmproj.gguf", proxy_url=None)

    entries = await catalog.build(user)

    assert entries[0].matricxon_supported is False
    assert "never run as a chat model" in entries[0].matricxon_unsupported_reason
    # Lets the frontend skip the "Not supported" badge for this one (see CatalogEntry.is_projector:
    # matricxon_supported=False alone reads backwards).
    assert entries[0].is_projector is True


@pytest.mark.asyncio
async def test_build_marks_a_not_installed_entry_removable_for_admin_only(db, user, admin_user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, {**_REPO, "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)

    admin_entries = await catalog.build(admin_user)
    regular_entries = await catalog.build(user)

    assert admin_entries[0].removable is True
    assert regular_entries[0].removable is False


@pytest.mark.asyncio
async def test_build_omits_an_installed_entry_entirely(db, admin_user, monkeypatch):
    """An installed extended entry is omitted entirely (not shown installed=True): ChatModelCatalogBuilder's
    "installed but not in the static catalog" branch already lists it in the default list, and showing it here
    would duplicate the row (see build).
    """
    tag = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    installed = [{"name": tag, "capabilities": ["completion"]}]
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(installed))
    catalog = ExtendedModelCatalog(db)

    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    await catalog.add("org/a", "x.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert entries == []
    # Still stored, though: uninstalling later would make it reappear here (not exercised; it depends on list_models
    # at read time, covered by the "not installed" tests).
    assert [e["tag"] for e in await catalog.list()] == [tag]


@pytest.mark.asyncio
async def test_build_omits_an_installed_entry_with_no_completion_capability(db, admin_user, monkeypatch):
    """Confirmed live: a vision mmproj file (or any non-"completion" install, e.g. embedding-only) pulled via
    "Browse more models" never appears in ChatModelCatalogBuilder's chat list (no "completion" capability), but must
    still leave *this* list once installed. The exclusion only looked at "completion"-capable installs, so such a
    tag stayed here forever with the repo's main-file family/parameter_size and the hardcoded matricxon_supported=False.
    """
    tag = ExtendedModelCatalog.build_tag("org/a", "mmproj.gguf")
    installed = [{"name": tag, "capabilities": []}]
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(installed))
    catalog = ExtendedModelCatalog(db)

    _stub_repo_files(
        monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "mmproj.gguf", "download_gb": 0.5}]}
    )
    await catalog.add("org/a", "mmproj.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert entries == []


@pytest.mark.asyncio
async def test_build_omits_an_entry_added_while_a_different_engine_was_active(db, admin_user, monkeypatch):
    """Every entry (self-registered or admin-added by search; add treats them alike) records the engine active
    when added. A tag added under Matricxon must not resurface as a "Browse more models" suggestion once Ollama is
    active (confirmed, reported twice: an auto-discovered install and a manually searched add).
    """
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    engine_service._cached_engine = "matricxon"
    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)

    engine_service._cached_engine = "ollama"
    entries = await catalog.build(admin_user)

    assert entries == []


@pytest.mark.asyncio
async def test_build_keeps_an_entry_once_the_engine_that_added_it_is_active_again(db, admin_user, monkeypatch):
    tag = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    engine_service._cached_engine = "matricxon"
    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert [e.tag for e in entries] == [tag]


@pytest.mark.asyncio
async def test_build_hides_admin_hidden_tags_from_a_regular_user(db, user, admin_user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, {**_REPO, "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/hidden", "x.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("org/hidden", "x.gguf")
    await HiddenModelTags(db).set({tag})

    regular_entries = await catalog.build(user)
    admin_entries = await catalog.build(admin_user)

    assert regular_entries == []
    assert len(admin_entries) == 1
    assert admin_entries[0].hidden is True


@pytest.mark.asyncio
async def test_build_skips_an_auto_registered_entry_once_it_is_no_longer_installed(db, admin_user, monkeypatch):
    """Auto-discovery registers an installed model with ALREADY_INSTALLED_NOTE; after uninstall it must not resurface as
    a
    "Browse more models" suggestion (the long stale list on an empty engine).
    """
    tag = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))
    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)

    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    assert await catalog.build(admin_user) == []


@pytest.mark.asyncio
async def test_clear_drops_manual_entries_but_keeps_the_installed_model_metadata_cache(db, monkeypatch):
    """Clearing on an engine switch must not force a Hugging Face re-lookup of every installed model."""
    installed = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": installed}]))
    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "x.gguf", proxy_url=None)
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/b", "files": [{"filename": "y.gguf", "download_gb": 1.0}]})
    await catalog.add("org/b", "y.gguf", proxy_url=None)

    await catalog.clear()

    assert [e["tag"] for e in await catalog.list()] == [installed]
