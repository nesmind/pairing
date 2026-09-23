"""Unit tests for app/services/extended_model_catalog_service.py. Every network-touching dependency
(list_models, HuggingFaceCatalogSearch.repo_files) is monkeypatched on this module's own imported binding, or
(for repo_files, a staticmethod) directly on the shared class object (not their origin modules — see the
"module-split import-binding gotcha" this codebase already tracks), so no real Ollama or Hugging Face call
happens here."""

import pytest

from app.services import extended_model_catalog_service as svc, gguf_probe
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


def _stub_repo_files(monkeypatch, repo: dict) -> None:
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(repo))
    )


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """Every test here gets a "couldn't reach Hugging Face" GGUF header probe by default — same None
    gguf_probe.probe_metadata itself returns on a genuine network failure, so add() falls back to its own
    repo-level-metadata/filename heuristics exactly as it did before this probe existed. Patched at its real
    origin module (see this file's own docstring on the import-binding gotcha) since
    extended_model_catalog_enrichment.HuggingFaceModelProbe.probe calls it via `gguf_probe.probe_metadata(...)`,
    a plain module attribute lookup at call time — real, still-pure quantization guessing (filename-only, no
    network) keeps running normally either way. Tests that specifically want real-probe behavior override this
    themselves."""

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
        ("moondream/moondream-2b-2025-04-14-4bit", "moondream-2b-2025-04-14-4bit"),  # no -GGUF suffix to strip
        ("org/GGUF", "GGUF"),  # stripping the whole name bare would leave nothing to show — keep it instead
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
    catalog = ExtendedModelCatalog(db)

    entry = await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry == {
        "tag": "hf.co/Qwen/Qwen2.5-14B-Instruct-GGUF:qwen2.5-14b-instruct-q4_k_m",
        # The repo's own name (trailing "-GGUF" stripped — see _repo_display_name), not repo["family"]
        # ("qwen2") — real in-file metadata is only ever used for the live Matricxon-support check now, never
        # for display (see add's own docstring on why: confirmed live that trusting it for display can be
        # actively unhelpful).
        "family": "Qwen2.5-14B-Instruct",
        "parameter_size": "14.0B",
        "download_gb": 9.0,
        "is_projector": False,
        "vision": False,
        # None: the (stubbed) GGUF header probe "failed", same as a real network hiccup — quantizations is
        # still real, pure, filename-only guessing (see
        # extended_model_catalog_enrichment.HuggingFaceModelProbe.guess_quantizations_from_filename) —
        # unaffected by the stub, since it never touches the network at all.
        "architecture": None,
        "quantizations": ["Q4_K"],
        "note": None,
    }
    assert await catalog.list() == [entry]


@pytest.mark.asyncio
async def test_add_labels_a_projector_file_with_the_repo_name_instead_of_inheriting_the_repo_family(db, monkeypatch):
    """Confirmed live twice: (1) a "moondream2" mmproj file was added with its sibling text model's own
    repo-level "phi2"/"1.4B" (HuggingFaceCatalogSearch.repo_files' family/parameter_size describe one file per
    repo, not the one actually picked — see that method's own docstring), showing as if the projector itself
    were a 1.4B Phi-2 chat model; (2) a bare "Vision projector" label fixing that was just as unidentifiable
    once there was more than one admin-added projector — the repo's own name has to be in it."""
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

    assert entry["family"] == "moondream-2b-2025-04-14-4bit (vision projector)"
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
    """Real bug found live (2026-09-22): an already-installed tag used to always skip the Hugging Face lookup
    entirely, leaving it stuck with null family/parameter_size forever — falling back, once displayed, to
    Matricxon's own generic engine-reported family (e.g. "Llama") instead of the real repo-derived name (e.g.
    "Llama-3.2-3B-Instruct"), even though the real data was one lookup away. Now it gets the same real lookup
    every other tag gets; `note` still marks it as already-installed (no download happens), but the rest of the
    entry is real, not null."""
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
    """The safety net the old unconditional skip existed for in the first place: silencing the "not in
    catalog" badge for something already on disk must never require connectivity — a HuggingFaceLookupError
    (network down, repo gone, rate-limited) falls back to the old null-data-plus-note shape instead of
    propagating and blocking the add outright."""
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
    """Confirmed live: re-searching a repo already both installed *and* catalogued (e.g. to double-check a pull
    "took", or just out of habit) used to dead-end on "already in the catalog" — with no way to see or remove
    the stale entry first, since build() already hides any installed tag regardless (see its own docstring).
    Re-adding must succeed instead, and (now that an already-installed tag gets the same real lookup as any
    other, see test_add_still_enriches_an_already_installed_tag_when_the_lookup_succeeds above) also self-heals
    a stale/wrong entry to fresh real data rather than leaving whatever the first add happened to store."""
    tag = ExtendedModelCatalog.build_tag(_REPO["repo_id"], _REPO["files"][0]["filename"])
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(monkeypatch, _REPO)
    catalog = ExtendedModelCatalog(db)
    await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag}]))
    updated_repo = {**_REPO, "parameter_size": "14.2B"}  # a slightly different real value than the first add
    _stub_repo_files(monkeypatch, updated_repo)

    entry = await catalog.add(_REPO["repo_id"], _REPO["files"][0]["filename"], proxy_url=None)

    assert entry["note"] == ExtendedModelCatalog.ALREADY_INSTALLED_NOTE
    stored = await catalog.list()
    assert [e["tag"] for e in stored] == [tag]  # replaced in place, not duplicated
    assert stored[0]["parameter_size"] == "14.2B"  # refreshed, not left at the first add's stale value


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
    assert entry.min_ram_gb == pytest.approx(12.5)  # 10.0 * 1.25, same heuristic as app/model_catalog.py
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
    assert entries[0].hardware_ok is False  # 40GB * 1.25 = 50GB needed, only 16GB available
    assert entries[0].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_gives_a_projector_its_own_accurate_unsupported_reason(db, user, monkeypatch):
    """A vision projector was never a standalone chat model to begin with, so the generic "architecture
    unknown, not verified yet" reason every other admin-added entry gets is actively misleading here — it
    implies verification might someday say True, which never happens for a file like this."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    _stub_repo_files(
        monkeypatch, {**_REPO, "files": [{"filename": "mmproj.gguf", "download_gb": 0.9, "is_projector": True}]}
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/a", "mmproj.gguf", proxy_url=None)

    entries = await catalog.build(user)

    assert entries[0].matricxon_supported is False
    assert "never run as a chat model" in entries[0].matricxon_unsupported_reason
    # Lets the frontend skip the "Not supported" badge outright for this one — see CatalogEntry.is_projector's
    # own docstring on why matricxon_supported=False alone reads as backwards for a file like this.
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
    """An installed extended-catalog entry is left out of this response entirely (not shown with installed=True)
    — app.services.model_catalog_service.ChatModelCatalogBuilder's own "installed but not in the static
    catalog" branch already surfaces it in the *default* list once pulled, so showing it here too would
    duplicate the same row across both lists (see build's own docstring)."""
    tag = ExtendedModelCatalog.build_tag("org/a", "x.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag, "capabilities": ["completion"]}]))
    catalog = ExtendedModelCatalog(db)

    _stub_repo_files(monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "x.gguf", "download_gb": 1.0}]})
    await catalog.add("org/a", "x.gguf", proxy_url=None)  # already installed at add time too

    entries = await catalog.build(admin_user)

    assert entries == []
    # Still stored, though — uninstalling later would make it reappear here (not exercised in this unit test,
    # since that only depends on list_models()'s live result at read time, already covered by the "not
    # installed" tests above).
    assert [e["tag"] for e in await catalog.list()] == [tag]


@pytest.mark.asyncio
async def test_build_omits_an_installed_entry_with_no_completion_capability(db, admin_user, monkeypatch):
    """Confirmed live: a vision mmproj/projector file (or any other non-"completion" install, e.g. an
    embedding-only one) pulled via "Browse more models" never appears in ChatModelCatalogBuilder's own
    chat-model list (it correctly has no "completion" capability) — but it must still disappear from *this*
    list once installed, the same as a real chat model would. Before this fix, the exclusion check below only
    looked at "completion"-capable installs, so a tag like this stayed stuck here forever, still showing
    whatever repo-level family/parameter_size Hugging Face reported for the repo's main (different) GGUF file,
    plus the hardcoded matricxon_supported=False every not-yet-installed entry gets."""
    tag = ExtendedModelCatalog.build_tag("org/a", "mmproj.gguf")
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag, "capabilities": []}]))
    catalog = ExtendedModelCatalog(db)

    _stub_repo_files(
        monkeypatch, {**_REPO, "repo_id": "org/a", "files": [{"filename": "mmproj.gguf", "download_gb": 0.5}]}
    )
    await catalog.add("org/a", "mmproj.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert entries == []


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
