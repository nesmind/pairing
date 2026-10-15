"""Tests for services/embedding_model_catalog_service.py, the embedding analogue of ChatModelCatalogBuilder
(tests/test_model_catalog_service.py). list_models is monkeypatched on each module's own binding, including
default_model_settings' (its installed-check makes an independent call; import-binding gotcha), so no real Ollama call.
"""

import pytest

from app.config import EMBEDDING_MODEL
from app.services import default_model_settings, embedding_model_catalog_service as svc
from app.services.embedding_model_catalog_service import EmbeddingModelCatalogService

_NOMIC_TAG = "hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0"
_MINILM_TAG = "hf.co/leliuga/all-MiniLM-L6-v2-GGUF:Q8_0"


async def _async_return(value):
    return value


def _stub_list_models(monkeypatch, installed: list[dict]) -> None:
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(installed))
    monkeypatch.setattr(default_model_settings, "list_models", lambda: _async_return(installed))


@pytest.mark.asyncio
async def test_build_returns_both_entries_when_nothing_installed(monkeypatch, db):
    _stub_list_models(monkeypatch, [])

    response = await EmbeddingModelCatalogService(db).build()

    assert {e.tag for e in response.entries} == {_NOMIC_TAG, _MINILM_TAG}
    assert all(not e.installed for e in response.entries)


@pytest.mark.asyncio
async def test_build_marks_the_matching_entry_installed(monkeypatch, db):
    _stub_list_models(monkeypatch, [{"name": _NOMIC_TAG, "capabilities": ["embedding"]}])

    response = await EmbeddingModelCatalogService(db).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_NOMIC_TAG].installed is True
    assert by_tag[_MINILM_TAG].installed is False


@pytest.mark.asyncio
async def test_build_ignores_a_non_embedding_capable_match(monkeypatch, db):
    """A model whose name matches a catalog tag but lacks the "embedding" capability (still mid-pull, or a same-named
    chat model) must not count as installed."""
    _stub_list_models(monkeypatch, [{"name": _NOMIC_TAG, "capabilities": ["completion"]}])

    response = await EmbeddingModelCatalogService(db).build()

    assert all(not e.installed for e in response.entries)


@pytest.mark.asyncio
async def test_build_default_tag_falls_back_to_config_when_unconfigured_and_uninstalled(monkeypatch, db):
    _stub_list_models(monkeypatch, [])

    response = await EmbeddingModelCatalogService(db).build()

    assert response.default_tag == EMBEDDING_MODEL


@pytest.mark.asyncio
async def test_build_default_tag_reflects_the_admin_configured_choice(monkeypatch, db):
    _stub_list_models(
        monkeypatch,
        [{"name": _NOMIC_TAG, "capabilities": ["embedding"]}, {"name": _MINILM_TAG, "capabilities": ["embedding"]}],
    )
    await default_model_settings.DefaultModelSettings(db).set_embedding(_MINILM_TAG)

    response = await EmbeddingModelCatalogService(db).build()

    assert response.default_tag == _MINILM_TAG


@pytest.mark.asyncio
async def test_build_hardware_ok_exempts_an_installed_entry(monkeypatch, db):
    _stub_list_models(monkeypatch, [{"name": _NOMIC_TAG, "capabilities": ["embedding"]}])
    monkeypatch.setattr(svc.hardware, "available_capacity_gb", lambda: 0.0)

    response = await EmbeddingModelCatalogService(db).build()

    by_tag = {e.tag: e for e in response.entries}
    assert by_tag[_NOMIC_TAG].hardware_ok is True
    assert by_tag[_MINILM_TAG].hardware_ok is False


@pytest.mark.asyncio
async def test_build_marks_only_the_system_choice_in_use(monkeypatch, db):
    _stub_list_models(
        monkeypatch,
        [{"name": _NOMIC_TAG, "capabilities": ["embedding"]}, {"name": _MINILM_TAG, "capabilities": ["embedding"]}],
    )
    await default_model_settings.DefaultModelSettings(db).set_embedding(_MINILM_TAG)

    by_tag = {e.tag: e for e in (await EmbeddingModelCatalogService(db).build()).entries}
    assert by_tag[_MINILM_TAG].in_use is True
    assert by_tag[_NOMIC_TAG].in_use is False
