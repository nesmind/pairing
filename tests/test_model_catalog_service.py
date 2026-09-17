"""Unit tests for app/services/model_catalog_service.py's embedding-model catalog — the only part of that
service with no existing coverage. Every network-touching dependency (list_models) is monkeypatched on this
module's own imported binding (not its origin module — see the "module-split import-binding gotcha" this
codebase already tracks), so no real Ollama call happens here."""

import pytest

from app import model_catalog
from app.config import EMBEDDING_MODEL
from app.services import model_catalog_service as svc

_NOMIC_TAG = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"
_MINILM_TAG = "hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0"


async def _async_return(value):
    return value


def test_find_entry_and_find_embedding_entry_never_cross_match():
    """The cheapest possible regression guard for the "embedding models can never leak into the chat
    picker" invariant EMBEDDING_CATALOG's own docstring describes — the two lists must stay disjoint."""
    assert model_catalog.find_entry(_NOMIC_TAG) is None
    assert model_catalog.find_embedding_entry(_NOMIC_TAG) is not None
    for entry in model_catalog.CATALOG:
        assert model_catalog.find_embedding_entry(entry["tag"]) is None


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_returns_both_entries_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    response = await svc.build_embedding_model_catalog()

    assert {e.tag for e in response.entries} == {_NOMIC_TAG, _MINILM_TAG}
    assert all(not e.installed for e in response.entries)


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_marks_the_matching_entry_installed(monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": _NOMIC_TAG, "capabilities": ["embedding"]}]),
    )

    response = await svc.build_embedding_model_catalog()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_NOMIC_TAG].installed is True
    assert by_tag[_MINILM_TAG].installed is False


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_ignores_a_non_embedding_capable_match(monkeypatch):
    """A model whose name happens to match a catalog tag but lacks the "embedding" capability (e.g. still
    mid-pull, or a same-named chat model) must not be reported as installed."""
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": _NOMIC_TAG, "capabilities": ["completion"]}]),
    )

    response = await svc.build_embedding_model_catalog()

    assert all(not e.installed for e in response.entries)


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_default_tag_matches_config(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))

    response = await svc.build_embedding_model_catalog()

    assert response.default_tag == EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_build_embedding_model_catalog_hardware_ok_exempts_an_installed_entry(monkeypatch):
    monkeypatch.setattr(
        svc,
        "list_models",
        lambda: _async_return([{"name": _NOMIC_TAG, "capabilities": ["embedding"]}]),
    )
    monkeypatch.setattr(svc.hardware, "available_capacity_gb", lambda: 0.0)

    response = await svc.build_embedding_model_catalog()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_NOMIC_TAG].hardware_ok is True  # installed, so exempt from the (trivially tiny) RAM gate
    assert by_tag[_MINILM_TAG].hardware_ok is False  # not installed, and 0GB available
