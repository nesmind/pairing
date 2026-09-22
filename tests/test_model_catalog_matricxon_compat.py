"""Unit tests for the Matricxon-compatibility fields (matricxon_supported/matricxon_unsupported_reason) added to
CatalogEntry/EmbeddingCatalogEntry — covers app.services.matricxon_support_checker.MatricxonSupportChecker (the
live check against Matricxon's own GET /api/health, see app.services.matricxon_client.get_capabilities) flowing
through ChatModelCatalogBuilder/EmbeddingModelCatalogService, and
app.services.extended_model_catalog_service.ExtendedModelCatalog.build's own same live check for an admin-added
entry (once its real per-file architecture is known — see
extended_model_catalog_enrichment.HuggingFaceModelProbe.probe — or the old always-unverified default when it
isn't). Every network-touching dependency is monkeypatched on each service module's own imported binding (see
the "module-split import-binding gotcha" this codebase already tracks), so no real Ollama/Matricxon/Hugging Face
call happens here — including matricxon_client.get_capabilities/list_models and gguf_probe.probe_metadata, any
of which would otherwise hit a real, possibly-running local Matricxon or the real internet."""

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
_GEMMA_12B_TAG = "hf.co/google/gemma-4-12B-it-qat-q4_0-gguf:gemma-4-12b-it-qat-q4_0"  # vision: True, gemma4
_LLAVA_TAG = "hf.co/second-state/Llava-v1.6-Vicuna-7B-GGUF:llava-v1.6-vicuna-7b-Q4_K_M"  # vision: True, llama
_NOMIC_TAG = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"

# Matricxon's own real support surface *before* the "llama"/"gemma4" architecture additions this file's tests
# below exercise the transition for — see ../matricxon/ROADMAP.md's "Expose supported architectures/
# quantizations" entry for why this is asked live instead of hand-copied.
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
    """Every test in this file that doesn't explicitly override this gets Matricxon's original (pre-M10) support
    surface by default — see MatricxonSupportChecker.verdict_for's own tests in test_matricxon_support_checker.py
    for the pure method's own full behavior matrix; this file only covers it flowing through the catalog
    builders."""

    async def fake_get_capabilities():
        return _ORIGINAL_CAPABILITIES

    monkeypatch.setattr(matricxon_client, "get_capabilities", fake_get_capabilities)


@pytest.fixture(autouse=True)
def _fake_matricxon_installed(monkeypatch):
    """Matricxon's own real, current per-tag capabilities (GET /api/tags on Matricxon specifically, via
    MatricxonSupportChecker._installed_info_or_none — see that method's own docstring for why this is asked
    separately from the active-engine-dispatched `list_models` this file's other tests already monkeypatch per
    catalog builder). Empty by default - "Matricxon has nothing installed" - overridden per-test below where a
    test needs a specific installed tag's own real per-tag RAM estimate or proven-install status."""

    async def fake_list_models():
        return []

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models)


@pytest.fixture(autouse=True)
def _stub_gguf_probe(monkeypatch):
    """Same "couldn't reach Hugging Face" stub as tests/test_extended_model_catalog_service.py's own fixture
    (see its docstring) — ExtendedModelCatalog.add now also probes the real GGUF header; overridden per-test
    below for the real architecture-flows-through-to-a-live-verdict case this file exists to cover."""

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
    # A hand-curated app/model_catalog.py entry, not a stray/manual install — see
    # CatalogEntry.is_auto_discovered's own docstring.
    assert by_tag[_MINISTRAL_TAG].is_auto_discovered is False


@pytest.mark.asyncio
async def test_build_model_catalog_marks_gemma_matricxon_unsupported_with_a_reason(db, user, monkeypatch):
    """With Matricxon's original support surface (gemma4 not yet implemented), a non-vision gemma4 entry is
    unsupported on architecture grounds."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_E2B_TAG].matricxon_supported is False
    assert by_tag[_GEMMA_E2B_TAG].matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_model_catalog_marks_gemma_matricxon_supported_once_the_architecture_ships(db, user, monkeypatch):
    """The real gap the live check closes: once Matricxon's own capabilities report "gemma4" (see
    ../matricxon/ROADMAP.md), a non-vision gemma4/Q4_0 entry becomes supported with no code change needed on
    this side — a hand-maintained static flag would have stayed stale until someone remembered to update it."""
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
    """gemma4:12b/LLaVA are vision-capable per this app's own catalog — matricxon_supported for them must track
    only the base architecture/quantization check, the same as any non-vision entry, regardless of whether the
    tag is installed at all or whether Matricxon's own real per-tag capabilities happen to include "vision":
    confirmed live (2026-09-21) that gating this field on vision-pairing too made an installed, demonstrably-
    working chat model (moondream2, text half confirmed running) show a plain "Not supported" badge, which reads
    as "this doesn't work at all" and is wrong for a model whose text chat clearly does (see
    MatricxonSupportChecker.verdict_for's own docstring)."""

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

    # Installed, but Matricxon's own real per-tag capabilities report no "vision" for it.
    async def fake_list_models_no_vision():
        return [{"name": _GEMMA_12B_TAG, "capabilities": ["completion"]}]

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models_no_vision)
    response = await ChatModelCatalogBuilder(db, user).build()
    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_GEMMA_12B_TAG].matricxon_supported is True
    assert by_tag[_GEMMA_12B_TAG].matricxon_unsupported_reason is None

    # Installed, with Matricxon's own real per-tag capabilities confirming "vision" too.
    async def fake_list_models_with_vision():
        return [{"name": _LLAVA_TAG, "capabilities": ["completion", "vision"]}]

    monkeypatch.setattr(matricxon_client, "list_models", fake_list_models_with_vision)
    response = await ChatModelCatalogBuilder(db, user).build()
    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_LLAVA_TAG].matricxon_supported is True
    assert by_tag[_LLAVA_TAG].matricxon_unsupported_reason is None


@pytest.mark.asyncio
async def test_build_model_catalog_treats_an_unreachable_matricxon_as_unsupported(db, user, monkeypatch):
    """MatricxonSupportChecker.verdict_for's own None-capabilities branch, exercised through the real catalog
    builder — Matricxon being down must not crash the whole Model tab, just leave every entry's support
    unverified."""
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
async def test_build_model_catalog_installed_extra_model_is_unverified_when_ollama_is_active(db, user, monkeypatch):
    """A model installed outside the curated CATALOG (e.g. pulled manually) is proven to work on whichever
    engine actually served list_models() — with Ollama active, that says nothing about Matricxon."""
    engine_service._cached_engine = "ollama"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": "custom:latest", "capabilities": ["completion"]}]),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag["custom:latest"].matricxon_supported is False
    assert by_tag["custom:latest"].matricxon_unsupported_reason


@pytest.mark.asyncio
async def test_build_model_catalog_installed_extra_model_is_proven_when_matricxon_is_active(db, user, monkeypatch):
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": "custom:latest", "capabilities": ["completion"]}]),
    )

    response = await ChatModelCatalogBuilder(db, user).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag["custom:latest"].matricxon_supported is True
    # Not in app/model_catalog.py's hand-curated list — see CatalogEntry.is_auto_discovered's own docstring.
    assert by_tag["custom:latest"].is_auto_discovered is True


@pytest.mark.asyncio
async def test_build_model_catalog_an_extended_catalog_tag_is_not_auto_discovered_once_installed(
    db, admin_user, monkeypatch
):
    """Real bug found live (2026-09-22): an admin deliberately added a model via "Browse more models"
    (ExtendedModelCatalog's own stored list), pulled it, and it still showed the same "Not in catalog" badge
    as a genuinely stray/unregistered install — reading as if something had gone wrong, when it was exactly
    what the admin asked for. is_auto_discovered must be False once the tag is found in ExtendedModelCatalog's
    own registered list too, not just app/model_catalog.py's hand-curated one."""
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
    # Real ExtendedModelCatalog-stored data ("repo", "3.2B"), not Matricxon's own raw report or a hardcoded
    # "Other" vendor — confirmed live the same day, this exact scenario showed "Llama unknown — Other".
    assert by_tag[tag].family == "repo"
    assert by_tag[tag].parameter_size == "3.2B"
    assert by_tag[tag].vendor == "org"


@pytest.mark.asyncio
async def test_build_model_catalog_treats_matricxons_literal_unknown_as_missing_data(db, admin_user, monkeypatch):
    """Real bug found live (2026-09-22): Matricxon's own /api/tags reports the literal string "unknown" for
    parameter_size when it genuinely doesn't know it — a truthy string, so the old `details.get("parameter_size")
    or "?"` fallback never caught it, and it rendered verbatim as "Llama unknown". Reproduces the exact
    conditions that triggered it: a tag registered via ExtendedModelCatalog's "already installed" path, but
    whose Hugging Face lookup failed (see ExtendedModelCatalog.add's own docstring — ADD still attempts the
    real lookup now, gracefully falling back to null family/parameter_size only when it can't reach it), so the
    fallback to Matricxon's own raw report is the only source left. The tag itself ("file", no recognizable
    quant suffix) also can't feed _quant_from_tag's own fallback, so "?" is genuinely the last resort here —
    see test_build_model_catalog_falls_back_to_the_tags_own_quant_before_a_bare_question_mark below for the
    case where it can."""
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
    """A genuinely stray install whose self-heal (see ChatModelCatalogBuilder._self_register) can't find real
    data either — the Hugging Face lookup itself fails here, same as it failing for any other reason — still
    has one real, useful thing to show: the exact quantization already sitting in its own tag (see
    ChatModelCatalogBuilder._quant_from_tag) — more informative than a bare "?" for a real difference the user
    can see (a Q3_K_M vs. a Q8_0 of the same model), and it costs nothing to derive."""
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
    assert entry.is_auto_discovered is True  # the self-heal's own lookup failed too — badge correctly stays


@pytest.mark.asyncio
async def test_build_model_catalog_self_heals_a_stray_install_for_an_admin(db, admin_user, monkeypatch):
    """The durable fix for "I keep seeing this bug every new model/engine switch" (reported live, 2026-09-22):
    rather than requiring a one-off manual ExtendedModelCatalog.add() every time some install bypasses the
    app's own "Browse more models" flow (a direct Ollama/Matricxon pull outside the UI, say), an admin's own
    catalog view now registers it automatically the first time it's seen — real family/parameter_size and all,
    on this exact same render, not just "next time"."""
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
    assert entry.is_auto_discovered is False
    # The repo's own real display name (see ExtendedModelCatalog._repo_display_name), not Matricxon's generic
    # engine-reported "Llama" (details.family above) — the whole point of this self-heal.
    assert entry.family == "Llama-3.2-3B-Instruct"
    assert entry.parameter_size == "3.2B"
    stored = await ExtendedModelCatalog(db).list()
    assert [e["tag"] for e in stored] == [tag]  # persisted — a future view won't need to look it up again


@pytest.mark.asyncio
async def test_build_model_catalog_never_self_heals_for_a_non_admin(db, user, monkeypatch):
    """A regular user's own catalog view must never write to the admin-curated extended catalog — see
    ChatModelCatalogBuilder.build's is_admin gate around _self_register."""
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
    """Confirmed live (2026-09-21): a real vision-projector (mmproj) file, pulled as its own independent
    Matricxon tag, used to be completely invisible everywhere once installed — no "completion" capability of
    its own means it's excluded from the normal installed-chat-model loop above, and (separately) it's excluded
    from "browse more models" too once installed there (see extended_model_catalog_service's own tests). Now
    surfaced as its own read-only-ish entry instead (see app.services.installed_projector_catalog's own tests
    for the pure-function behavior this only smoke-tests end to end)."""
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
    """Falls back to the old, always-unverified framing only when
    extended_model_catalog_enrichment.HuggingFaceModelProbe.probe genuinely couldn't read a real architecture (a
    network hiccup at add time, or a corrupt/unusual file — see this file's own _stub_gguf_probe fixture) — see
    the test right below for the real-architecture case this replaced as the default."""
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
    """The real gap HuggingFaceModelProbe.probe/MatricxonSupportChecker.verdict_for closes: an admin-added entry
    used to always show "not verified" regardless of what it actually was (a hardcoded False, confirmed live as
    actively misleading for a real, Matricxon-supported model) — once its real GGUF header names a supported
    architecture, it now gets the exact same live verdict_for check a hand-curated CATALOG entry already gets."""
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
        return {"architecture": "mistral3", "name": "Hf"}  # general.name is real but unreliable — see below

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)
    catalog = ExtendedModelCatalog(db)
    await catalog.add("org/repo", "x-q4_k_m.gguf", proxy_url=None)

    entries = await catalog.build(admin_user)

    assert len(entries) == 1
    # The repo's own name ("repo", from "org/repo"), not probed["name"] ("Hf") — confirmed live that a real,
    # popular model's own general.name can be exactly this kind of unhelpful placeholder, so display never
    # trusts it; the real architecture below is still used, just for the live support check, not display.
    assert entries[0].family == "repo"
    # mistral3/Q4_K are both in _ORIGINAL_CAPABILITIES (this file's own default fixture) — a real supported
    # combination, unlike the always-False every admin-added entry used to get regardless.
    assert entries[0].matricxon_supported is True
    assert entries[0].matricxon_unsupported_reason is None
