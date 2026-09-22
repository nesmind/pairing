"""Unit tests for app/services/model_catalog_service.py — installed_embedding_models and resolve_installed_model,
the parts of this module with no coverage elsewhere (ChatModelCatalogBuilder itself is covered end to end by
tests/test_model_catalog_matricxon_compat.py; HiddenModelTags/DefaultModelSettings delegation by
tests/test_default_model_settings.py). Every network-touching dependency is monkeypatched on this module's own
imported binding (not its origin module — see the "module-split import-binding gotcha" this codebase already
tracks), so no real Ollama call happens here."""

import pytest

from app import model_catalog
from app.services import model_catalog_service as svc

_NOMIC_TAG = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"


async def _async_return(value):
    return value


def test_find_entry_and_find_embedding_entry_never_cross_match():
    """The cheapest possible regression guard for the "embedding models can never leak into the chat
    picker" invariant CuratedCatalog.embedding_models' own docstring describes — the two collections must stay
    disjoint."""
    assert model_catalog.CATALOG.find(_NOMIC_TAG) is None
    assert model_catalog.CATALOG.find_embedding(_NOMIC_TAG) is not None
    for entry in model_catalog.CATALOG.chat_models:
        assert model_catalog.CATALOG.find_embedding(entry.tag) is None


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M", "unsloth"),
        ("hf.co/google/gemma-4-E2B-it-qat-q4_0-gguf:gemma-4-E2B_q4_0-it", "google"),
        ("custom:latest", "Other"),  # a plain Ollama-library tag has no real org to read
    ],
)
def test_vendor_from_tag(tag, expected):
    assert svc.ChatModelCatalogBuilder._vendor_from_tag(tag) == expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("llama", "llama"),
        ("unknown", None),  # Matricxon's own literal sentinel for "genuinely don't know" — see this method's
        # own docstring, confirmed live, 2026-09-22
        ("UNKNOWN", None),  # case-insensitive — same reasoning
        (None, None),
        ("", None),
    ],
)
def test_clean_engine_reported(value, expected):
    assert svc.ChatModelCatalogBuilder._clean_engine_reported(value) == expected


@pytest.mark.parametrize(
    "tag,expected",
    [
        ("hf.co/unsloth/Llama-3.2-3B-Instruct-GGUF:Llama-3.2-3B-Instruct-Q3_K_M", "Q3_K_M"),
        ("hf.co/cjpais/llava-1.6-mistral-7b-gguf:llava-v1.6-mistral-7b.Q3_K_M", "Q3_K_M"),
        ("hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0", "Q8_0"),
        ("hf.co/unsloth/MiniMax-M3-GGUF:UD-IQ1_M", "IQ1_M"),
        ("custom:latest", None),  # a plain Ollama-library tag has nothing quant-shaped to find
    ],
)
def test_quant_from_tag(tag, expected):
    """Confirmed live, 2026-09-22: a bare "?" for parameter_size is less useful than the exact quantization
    already sitting right there in the tag when nothing else is known — see _QUANT_TOKEN_RE's own module-level
    comment for why this is deliberately more precise than the existing coarse Matricxon-compatibility guess."""
    assert svc.ChatModelCatalogBuilder._quant_from_tag(tag) == expected


@pytest.mark.asyncio
async def test_installed_embedding_models_filters_by_capability(monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return(
            [
                {"name": _NOMIC_TAG, "capabilities": ["embedding"]},
                {"name": "some-chat-model", "capabilities": ["completion"]},
            ]
        ),
    )
    assert await svc.installed_embedding_models() == [_NOMIC_TAG]


# ---- resolve_installed_model -----------------------------------------------
# The real bug this covers: an admin switching the active engine (see
# app.services.engine_service) leaves every existing conversation's own
# `model` column pointed at a tag that engine may never have heard of —
# Ollama and Matricxon each have their own separate catalog. See
# app.services.chat_service.build_reply_stream for the only caller.

_CHAT_TAG = "some-chat-model"


@pytest.mark.asyncio
async def test_resolve_installed_model_returns_it_unchanged_when_already_installed(monkeypatch, db, user):
    monkeypatch.setattr(svc, "installed_chat_models", lambda: _async_return([_CHAT_TAG]))
    assert await svc.resolve_installed_model(db, user, _CHAT_TAG) == _CHAT_TAG


@pytest.mark.asyncio
async def test_resolve_installed_model_falls_back_to_the_users_own_default(monkeypatch, db, user):
    monkeypatch.setattr(svc, "installed_chat_models", lambda: _async_return([_CHAT_TAG]))
    await svc.settings_service.set_default_model(db, user.id, _CHAT_TAG)
    assert await svc.resolve_installed_model(db, user, "stale-tag") == _CHAT_TAG


@pytest.mark.asyncio
async def test_resolve_installed_model_falls_back_to_the_system_default_when_the_users_own_isnt_installed_either(
    monkeypatch, db, user
):
    monkeypatch.setattr(svc, "installed_chat_models", lambda: _async_return([_CHAT_TAG]))
    # user's own default ("some other tag never pulled") isn't installed, so this must skip past it rather than
    # returning it anyway.
    await svc.settings_service.set_default_model(db, user.id, "never-pulled-tag")
    await svc.set_default_model_for_new_users(db, _CHAT_TAG)
    assert await svc.resolve_installed_model(db, user, "stale-tag") == _CHAT_TAG


@pytest.mark.asyncio
async def test_resolve_installed_model_falls_back_to_whatever_is_installed_first_as_a_last_resort(
    monkeypatch, db, user
):
    monkeypatch.setattr(svc, "installed_chat_models", lambda: _async_return([_CHAT_TAG]))
    assert await svc.resolve_installed_model(db, user, "stale-tag") == _CHAT_TAG


@pytest.mark.asyncio
async def test_resolve_installed_model_returns_the_original_tag_when_nothing_at_all_is_installed(monkeypatch, db, user):
    """No fallback exists — returns the original (still-invalid) tag rather than raising, same as before this
    existed: the caller's own attempt at using it fails the normal way, with a clear error, not a crash here."""
    monkeypatch.setattr(svc, "installed_chat_models", lambda: _async_return([]))
    assert await svc.resolve_installed_model(db, user, "stale-tag") == "stale-tag"
