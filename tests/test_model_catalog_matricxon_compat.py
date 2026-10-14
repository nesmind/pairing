"""Tests for the Matricxon-compatibility fields (matricxon_supported/matricxon_unsupported_reason) of
CatalogEntry/EmbeddingCatalogEntry: MatricxonSupportChecker (the live check against Matricxon's GET /api/health,
matricxon_client.get_capabilities) flowing through ChatModelCatalogBuilder/EmbeddingModelCatalogService, and
ExtendedModelCatalog.build's same check for an admin-added entry (once its architecture is known:
extended_model_catalog_enrichment.HuggingFaceModelProbe.probe; else the old always-unverified default).
Network dependencies are patched on each service module's binding (module-split gotcha): no real
Ollama/Matricxon/Hugging Face call, including matricxon_client.get_capabilities/list_models and
gguf_probe.probe_metadata.
"""

import pytest

from app.services import (
    embedding_model_catalog_service as embedding_svc,
    engine_service,
    extended_model_catalog_service as extended_svc,
    gguf_probe,
    matricxon_client,
    model_catalog_service as svc,
)
from app.services.embedding_model_catalog_service import EmbeddingModelCatalogService
from app.services.extended_model_catalog_service import ExtendedModelCatalog
from app.services.huggingface_client import HuggingFaceCatalogSearch, HuggingFaceLookupError
from app.services.model_catalog_service import ChatModelCatalogBuilder

_MINISTRAL_TAG = "hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M"
_GEMMA_E2B_TAG = "hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it"
_GEMMA_12B_TAG = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:gemma-4-12b-it-qat-q4_0"
_LLAVA_TAG = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:llava-v1.6-vicuna-7b-Q4_K_M"
# llama, vision: True. No longer curated (removed), exercised only as an auto-discovered install.
_NOMIC_TAG = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"

# Matricxon's support surface *before* the "llama"/"gemma4" additions this file tests (../matricxon/ROADMAP.md
# "Expose supported architectures/quantizations": asked live, not hand-copied).
_ORIGINAL_CAPABILITIES = {
    "supported_architectures": ["mistral3", "bert", "nomic-bert"],
    "supported_quantizations": ["F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"],
}


async def _async_return(value):
    return value


@pytest.fixture(autouse=True)
def reset_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


@pytest.fixture(autouse=True)
def _fake_matricxon_capabilities(monkeypatch):
    """Every test not overriding this gets Matricxon's original (pre-M10) support surface; the pure method's full
    matrix is in test_matricxon_support_checker.py (MatricxonSupportChecker.verdict_for); this covers it flowing through
    the builders.
    """

    async def fake_get_capabilities():
        return _ORIGINAL_CAPABILITIES

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake_get_capabilities)


@pytest.fixture(autouse=True)
def _fake_matricxon_installed(monkeypatch):
    """Matricxon's current per-tag capabilities (its GET /api/tags, via MatricxonSupportChecker._installed_info_or_none,
    asked apart from the active-engine `list_models` other tests patch). Empty by default ("nothing installed");
    overridden where a test needs a tag's RAM estimate or proven-install status.
    """

    async def fake_list_models():
        return []

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models)


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """The "couldn't reach Hugging Face" stub of tests/test_extended_model_catalog_service.py: ExtendedModelCatalog.add
    also probes the GGUF header; overridden where the real architecture-to-verdict flow is tested.
    """

    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return None

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)


@pytest.mark.asyncio
async def test_build_model_catalog_marks_ministral_3_matricxon_supported(db, user, monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_MINISTRAL_TAG].matricxon_supported is True
    assert by_tag[_MINISTRAL_TAG].matricxon_unsupported_reason is None
    # A hand-curated app/model_catalog.py entry, not a stray install (CatalogEntry.is_auto_discovered).
    assert by_tag[_MINISTRAL_TAG].is_auto_discovered is False


@pytest.mark.asyncio
async def test_build_model_catalog_marks_gemma_matricxon_unsupported_with_a_reason(db, user, monkeypatch):
    """With Matricxon's original support surface (gemma4 not implemented), a non-vision gemma4 entry is unsupported on
    architecture."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_E2B_TAG].matricxon_supported is False
    assert by_tag[_GEMMA_E2B_TAG].matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_model_catalog_marks_gemma_matricxon_supported_once_the_architecture_ships(db, user, monkeypatch):
    """The gap the live check closes: once Matricxon reports "gemma4" (../matricxon/ROADMAP.md), a non-vision
    gemma4/Q4_0 entry becomes supported with no code change here; a hand-kept static flag would stay stale.
    """
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    async def fake_get_capabilities():
        return {
            "supported_architectures": ["mistral3", "bert", "nomic-bert", "llama", "gemma4"],
            "supported_quantizations": ["Q4_0", "Q4_K", "Q6_K", "Q8_0"],
        }

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake_get_capabilities)

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_E2B_TAG].matricxon_supported is True
    assert by_tag[_GEMMA_E2B_TAG].matricxon_unsupported_reason is None


@pytest.mark.asyncio
async def test_build_model_catalog_vision_entry_support_ignores_install_and_vision_pairing_state(db, user, monkeypatch):
    """gemma4:12b (still curated) and LLaVA (removed from default_models.json; tested via auto-discovery)
    are vision-capable. matricxon_supported must track only the base architecture/quantization check, as for any
    non-vision entry, whether or not the tag is installed or Matricxon's capabilities include "vision": gating on
    pairing made an installed, working chat model (moondream2) show "Not supported"
    (MatricxonSupportChecker.verdict_for).

    Every case patches `svc.list_models` (this module's binding; module-split gotcha), never
    `matricxon_client.list_models`: once `svc.list_models` is rebound it stays so, and patching the client later has
    no effect on `build`. That inconsistency let the last assertion check nothing for years: LLaVA's curated
    entry masked it as "not installed" until its removal exposed a bare KeyError.
    """

    async def fake_get_capabilities():
        return {
            "supported_architectures": ["mistral3", "bert", "nomic-bert", "llama", "gemma4"],
            "supported_quantizations": ["Q4_0", "Q4_K", "Q6_K", "Q8_0"],
        }

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake_get_capabilities)

    # Not installed at all.
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    response = await ChatModelCatalogBuilder(db, user).build()
    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_12B_TAG].matricxon_supported is True
    assert by_tag[_GEMMA_12B_TAG].matricxon_unsupported_reason is None

    # Installed, but Matricxon's per-tag capabilities report no "vision".
    async def fake_list_models_no_vision():
        return [{"name": _GEMMA_12B_TAG, "capabilities": ["completion"], "details": {"family": "gemma4"}}]

    monkeypatch.setattr(svc, "list_models", fake_list_models_no_vision)
    response = await ChatModelCatalogBuilder(db, user).build()
    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_12B_TAG].matricxon_supported is True
    assert by_tag[_GEMMA_12B_TAG].matricxon_unsupported_reason is None

    # Installed, with Matricxon's per-tag capabilities confirming "vision".
    async def fake_list_models_with_vision():
        return [{"name": _LLAVA_TAG, "capabilities": ["completion", "vision"], "details": {"family": "llama"}}]

    monkeypatch.setattr(svc, "list_models", fake_list_models_with_vision)
    response = await ChatModelCatalogBuilder(db, user).build()
    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_LLAVA_TAG].matricxon_supported is True
    assert by_tag[_LLAVA_TAG].matricxon_unsupported_reason is None
    assert by_tag[_LLAVA_TAG].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_model_catalog_treats_an_unreachable_matricxon_as_unsupported(db, user, monkeypatch):
    """MatricxonSupportChecker.verdict_for's None-capabilities branch through the real builder: Matricxon being
    down must not crash the Model tab, just leave every entry's support unverified.
    """
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    from app.services.matricxon_client import MatricxonError

    async def fake_get_capabilities():
        raise MatricxonError("down")

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake_get_capabilities)

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_MINISTRAL_TAG].matricxon_supported is False
    assert by_tag[_MINISTRAL_TAG].matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_model_catalog_auto_discovered_entry_is_honestly_unverified_with_no_real_architecture(
    db, user, monkeypatch
):
    """A model installed outside the curated CATALOG (e.g. pulled manually) with no architecture data anywhere (no
    ExtendedModelCatalog registration, no engine-reported details.family) must not be assumed supported just
    because Matricxon is active. Bug: Matricxon's /api/tags lists anything with a valid sidecar
    whether or not it ever loaded, so "installed while Matricxon is active" proved nothing about support.
    """
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": "custom:latest", "capabilities": ["completion"]}]),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag["custom:latest"].matricxon_supported is False
    assert by_tag["custom:latest"].matricxon_unsupported_reason
    # Not in app/model_catalog.py's curated list (CatalogEntry.is_auto_discovered).
    assert by_tag["custom:latest"].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_model_catalog_auto_discovered_entry_gets_a_real_verdict_when_matricxon_is_active(
    db, user, monkeypatch
):
    """The fix: checker.verdict_for is the one check every other entry (curated, extended, embedding) uses, and an
    auto-discovered entry shares it, taking the architecture from the engine that reported it installed (both
    Ollama's and Matricxon's /api/tags report the GGUF general.architecture as details.family).
    """
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {
                    "name": "hf.co/org/repo:model-Q4_K_M",
                    "capabilities": ["completion"],
                    "details": {"family": "mistral3"},
                }
            ]
        ),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    entry = {e.tag: e for e in response.entries}["hf.co/org/repo:model-Q4_K_M"]
    assert entry.matricxon_supported is True
    assert entry.matricxon_unsupported_reason is None


@pytest.mark.asyncio
async def test_build_model_catalog_surfaces_matricxons_chat_format_unverified_capability(db, user, monkeypatch):
    """Matricxon's "chat_format_unverified" capability (../matricxon/app/models/capabilities.py) must reach
    CatalogEntry.chat_format_unverified, live-only like "vision". Covers a gap:
    a Hebrew Mistral fine-tune
    gave incoherent, non-chat output with any prompt format, and nothing in the catalog set it apart from well-behaved
    models.
    """
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": "custom:latest", "capabilities": ["completion", "chat_format_unverified"]}]),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    assert {e.tag: e for e in response.entries}["custom:latest"].chat_format_unverified is True


@pytest.mark.asyncio
async def test_build_model_catalog_does_not_flag_a_model_matricxon_confirmed(db, user, monkeypatch):
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": "custom:latest", "capabilities": ["completion"]}]),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    assert {e.tag: e for e in response.entries}["custom:latest"].chat_format_unverified is False


@pytest.mark.asyncio
async def test_build_model_catalog_auto_discovered_entry_is_unverified_when_ollama_is_active_even_with_real_data(
    db, user, monkeypatch
):
    """MatricxonSupportChecker.load deliberately asks Matricxon nothing while Ollama is active (a perf cost avoided
    for every entry type; see its docstring), so even a known-supported architecture reads "not verified": the honest
    "we never asked" reason, not a reachability claim either way.
    """
    engine_service._cached_engine = "ollama"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {
                    "name": "hf.co/org/repo:model-Q4_K_M",
                    "capabilities": ["completion"],
                    "details": {"family": "mistral3"},
                }
            ]
        ),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    entry = {e.tag: e for e in response.entries}["hf.co/org/repo:model-Q4_K_M"]
    assert entry.matricxon_supported is False
    assert "Could not reach Matricxon" in entry.matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_model_catalog_an_extended_catalog_tag_is_not_auto_discovered_once_installed(
    db, admin_user, monkeypatch
):
    """Bug found: a model deliberately added via "Browse more models" (ExtendedModelCatalog's list)
    and pulled still showed the "Not in catalog" badge of a stray install, as if something had gone wrong.
    is_auto_discovered must be False once the tag is in ExtendedModelCatalog's registered list too, not just the curated
    one.
    """
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(
            lambda repo_id, proxy_url: _async_return(
                {
                    "repo_id": repo_id,
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "context_length": 131072,
                    "gated": False,
                    "license": None,
                    "files": [{"filename": "model-q3_k_m.gguf", "download_gb": 1.7}],
                }
            )
        ),
    )
    await ExtendedModelCatalog(db).add("org/repo", "model-q3_k_m.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("org/repo", "model-q3_k_m.gguf")

    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": tag, "capabilities": ["completion"], "details": {"family": "llama"}}]),
    )

    response = await ChatModelCatalogBuilder(db, admin_user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[tag].installed is True
    assert by_tag[tag].is_auto_discovered is False
    # Real ExtendedModelCatalog data ("repo", "3.2B"), not Matricxon's raw report or a hardcoded "Other" vendor;
    # this scenario showed "Llama unknown — Other" (same day).
    assert by_tag[tag].family == "repo"
    assert by_tag[tag].parameter_size == "3.2B"
    assert by_tag[tag].vendor == "org"


@pytest.mark.asyncio
async def test_build_model_catalog_treats_matricxons_literal_unknown_as_missing_data(db, admin_user, monkeypatch):
    """Bug found: Matricxon's /api/tags reports the literal "unknown" for parameter_size when it
    doesn't know; truthy, so `details.get("parameter_size") or "?"` never caught it and it rendered "Llama unknown".
    Reproduces: a tag registered via ExtendedModelCatalog's "already installed" path whose Hugging Face lookup
    failed (add still tries the lookup, falling back to null family/parameter_size only if it can't), leaving
    Matricxon's raw report as the only source. The tag ("file", no quant suffix) also can't feed _quant_from_tag,
    so "?" is the last resort (see
    test_build_model_catalog_falls_back_to_the_tags_own_quant_before_a_bare_question_mark).
    """
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([{"name": "hf.co/org/repo:file"}]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError('"org/repo" was not found on Hugging Face — check the repo is correct.')

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))
    await ExtendedModelCatalog(db).add("org/repo", "file.gguf", proxy_url=None)
    tag = ExtendedModelCatalog.build_tag("org/repo", "file.gguf")

    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {
                    "name": tag,
                    "capabilities": ["completion"],
                    "details": {"family": "llama", "parameter_size": "unknown"},
                }
            ]
        ),
    )

    response = await ChatModelCatalogBuilder(db, admin_user).build()

    entry = {e.tag: e for e in response.entries}[tag]
    assert entry.parameter_size == "?"
    assert entry.family == "Llama"


@pytest.mark.asyncio
async def test_build_model_catalog_falls_back_to_the_tags_own_quant_before_a_bare_question_mark(
    db, admin_user, monkeypatch
):
    """A stray install (no real data, and no Hugging Face lookup on a catalog view) still has the exact quantization
    in its own tag (ChatModelCatalogBuilder._quant_from_tag): more informative than "?" (Q3_K_M vs Q8_0 of one
    model), and free to derive.
    """
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))

    def _raise(*_a, **_kw):
        raise HuggingFaceLookupError("network unreachable")

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_raise))
    tag = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [{"name": tag, "capabilities": ["completion"], "details": {"family": "llama", "parameter_size": "unknown"}}]
        ),
    )

    response = await ChatModelCatalogBuilder(db, admin_user).build()

    entry = {e.tag: e for e in response.entries}[tag]
    assert entry.parameter_size == "Q3_K_M"
    assert entry.is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_model_catalog_never_calls_hugging_face_for_a_stray_install(db, admin_user, monkeypatch):
    """Metadata is captured when a model is added or installed, never on a catalog view: a stray install (e.g. pulled
    on the terminal) renders with the engine-reported fallback, with no Hugging Face call or write.
    """

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("repo_files must not be called while building the catalog")

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_fail_if_called))
    tag = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [{"name": tag, "capabilities": ["completion"], "details": {"family": "llama", "parameter_size": "unknown"}}]
        ),
    )

    response = await ChatModelCatalogBuilder(db, admin_user).build()

    entry = {e.tag: e for e in response.entries}[tag]
    assert entry.is_auto_discovered is True
    assert await ExtendedModelCatalog(db).list() == []


@pytest.mark.asyncio
async def test_build_model_catalog_backfills_discovered_via_engine_for_pre_existing_extended_entries(
    db, admin_user, monkeypatch
):
    """A tag self-registered before ExtendedModelCatalog.add recorded discovered_via_engine gets it
    backfilled when an admin's catalog view next finds it installed: the one moment with live proof of which
    engine has it (ExtendedModelCatalog.backfill_discovered_via_engine: unrecoverable once uninstalled).
    """
    tag = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M"
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(
            lambda repo_id, proxy_url: _async_return(
                {
                    "repo_id": repo_id,
                    "family": "llama",
                    "parameter_size": "3.2B",
                    "context_length": 131072,
                    "gated": False,
                    "license": None,
                    "files": [{"filename": "Llama-3.2-3B-Instruct-Q3_K_M.gguf", "download_gb": 1.7}],
                }
            )
        ),
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q3_K_M.gguf", proxy_url=None)
    # Simulates pre- data: the key never existed, not merely None.
    stored = await catalog.list()
    del stored[0]["discovered_via_engine"]
    await catalog._save(stored)

    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([{"name": tag, "capabilities": ["completion"]}]))

    await ChatModelCatalogBuilder(db, admin_user).build()

    backfilled = await ExtendedModelCatalog(db).list()
    assert backfilled[0]["discovered_via_engine"] == "matricxon"


@pytest.mark.asyncio
async def test_build_model_catalog_never_writes_for_a_non_admin(db, user, monkeypatch):
    """A regular user's catalog view must never write to the admin-curated extended catalog
    (ChatModelCatalogBuilder.build)."""
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))

    def _fail_if_called(*_a, **_kw):
        raise AssertionError("repo_files must not be called for a non-admin's own catalog view")

    monkeypatch.setattr(HuggingFaceCatalogSearch, "repo_files", staticmethod(_fail_if_called))
    tag = "hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [{"name": tag, "capabilities": ["completion"], "details": {"family": "llama", "parameter_size": "unknown"}}]
        ),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    entry = {e.tag: e for e in response.entries}[tag]
    assert entry.is_auto_discovered is True
    assert await ExtendedModelCatalog(db).list() == []


@pytest.mark.asyncio
async def test_build_model_catalog_surfaces_an_installed_vision_projector(db, user, monkeypatch):
    """Confirmed live: a vision-projector (mmproj) file pulled as its own Matricxon tag was invisible
    everywhere once installed: no "completion" capability excludes it from the installed-chat loop, and it's
    excluded from "browse more models" once installed there (extended_model_catalog_service's tests). It is now a
    read-only-ish entry (installed_projector_catalog's tests cover the pure behavior; this smoke-tests end to end).
    """
    projector_tag = "hf.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf:llama-joycaption-mmproj-f16"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [{"name": projector_tag, "size": 877771808, "capabilities": [], "details": {"family": "clip"}}]
        ),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert projector_tag in by_tag
    assert by_tag[projector_tag].is_projector is True
    assert by_tag[projector_tag].text_capable is False
    assert by_tag[projector_tag].installed is True
    assert by_tag[projector_tag].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_marks_nomic_matricxon_supported(monkeypatch, db):
    monkeypatch.setattr(embedding_svc, "list_models", lambda: _async_return([]))

    response = await EmbeddingModelCatalogService(db).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_NOMIC_TAG].matricxon_supported is True


@pytest.mark.asyncio
async def test_build_extended_catalog_marks_an_entry_unverified_when_the_gguf_probe_cant_determine_it(
    db, admin_user, monkeypatch
):
    """Falls back to the always-unverified framing only when HuggingFaceModelProbe.probe couldn't read an architecture
    (a network hiccup at add time, or a corrupt/unusual file; see _stub_gguf_probe). The test below covers a real
    architecture.
    """
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(
            lambda repo_id, proxy_url: _async_return(
                {
                    "repo_id": repo_id,
                    "family": "qwen2",
                    "parameter_size": "7B",
                    "context_length": 32768,
                    "gated": False,
                    "license": "apache-2.0",
                    "files": [{"filename": "x.gguf", "download_gb": 4.0}],
                }
            )
        ),
    )
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/repo", "x.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert len(entries) == 1
    assert entries[0].matricxon_supported is False
    assert "architecture unknown" in entries[0].matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_extended_catalog_gives_a_real_verdict_once_the_gguf_probe_finds_a_supported_architecture(
    db, admin_user, monkeypatch
):
    """The gap HuggingFaceModelProbe.probe/MatricxonSupportChecker.verdict_for close: an admin-added entry always
    showed "not verified" (a hardcoded False, misleading for a supported model). Once its GGUF header names a
    supported architecture it gets the same live verdict_for as a curated CATALOG entry.
    """
    monkeypatch.setattr(extended_svc, "list_models", lambda: _async_return([]))
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(
            lambda repo_id, proxy_url: _async_return(
                {
                    "repo_id": repo_id,
                    "family": "qwen2",
                    "parameter_size": "7B",
                    "context_length": 32768,
                    "gated": False,
                    "license": "apache-2.0",
                    "files": [{"filename": "x-q4_k_m.gguf", "download_gb": 4.0}],
                }
            )
        ),
    )

    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return {"architecture": "mistral3", "name": "Hf"}

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/repo", "x-q4_k_m.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert len(entries) == 1
    # The repo name ("repo" from "org/repo"), not probed["name"] ("Hf"): a popular model's general.name can be such
    # a placeholder, so display never trusts it; the architecture is used only for the live support check.
    assert entries[0].family == "repo"
    # mistral3/Q4_K are in _ORIGINAL_CAPABILITIES (this file's default fixture): a supported combination, unlike the
    # always-False admin-added entries used to get.
    assert entries[0].matricxon_supported is True
    assert entries[0].matricxon_unsupported_reason is None


@pytest.mark.asyncio
async def test_build_marks_a_curated_projector_installed_despite_having_no_completion_capability(
    db, admin_user, monkeypatch
):
    """default_models.json's SmolVLM2 mmproj entry: a projector never reports "completion", so it must be looked up
    among all installed tags, not just chat-capable ones, and flagged is_projector.
    """
    tag = "hf.co/ggml-org/SmolVLM2-2.2B-Instruct-GGUF:mmproj-SmolVLM2-2.2B-Instruct-Q8_0"
    monkeypatch.setattr(
        svc, "list_models", lambda: _async_return([{"name": tag, "capabilities": [], "details": {"family": "clip"}}])
    )

    response = await ChatModelCatalogBuilder(db, admin_user).build()

    projector = next(e for e in response.entries if e.tag == tag)
    assert projector.installed and projector.is_projector and not projector.text_capable
    assert len([e for e in response.entries if e.tag == tag]) == 1


@pytest.mark.asyncio
async def test_a_connector_models_own_name_and_vendor_are_shown(db, user, monkeypatch):
    """A connector engine (cloudflare_client) gives a readable name, vendor and context: its tags aren't hf.co paths."""
    cloudflare_model = {
        "name": "@cf/meta/llama-3.1-8b-instruct-fp8",
        "capabilities": ["completion"],
        "display_name": "Llama 3.1 8B Instruct FP8",
        "vendor": "Meta",
        "details": {"context_length": 32000},
    }
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([cloudflare_model]))

    response = await ChatModelCatalogBuilder(db, user).build()

    entry = next(e for e in response.entries if e.tag == cloudflare_model["name"])
    assert entry.family == "Llama 3.1 8B Instruct FP8"
    assert entry.parameter_size == ""  # no "?", and no tag-based "FP8" repeated
    assert entry.vendor == "Meta" and entry.context_length == 32000
