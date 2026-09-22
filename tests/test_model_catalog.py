"""Unit tests for app/model_catalog.py — CuratedCatalogLoader reading/validating default_models.json. See
tests/test_model_catalog_service.py::test_find_entry_and_find_embedding_entry_never_cross_match for the
CuratedCatalog.find/find_embedding disjoint-collections behavior, unchanged by this loader and not repeated
here."""

import json

import pytest
from pydantic import ValidationError

from app.model_catalog import _DEFAULT_MODELS_PATH, CATALOG, CuratedCatalogLoader

_VALID_ENTRY = {
    "family": "Test Model",
    "vendor": "Test Vendor",
    "tag": "hf.co/test/test-repo:test-file",
    "parameter_size": "1B",
    "context_length": 2048,
    "download_gb": 1.0,
    "min_ram_gb": 2,
    "locally_runnable": True,
    "architecture": "llama",
    "quantizations": ["Q4_K"],
}


def _write(tmp_path, data: dict):
    path = tmp_path / "default_models.json"
    path.write_text(json.dumps(data))
    return path


def test_load_reads_the_real_shipped_file():
    """A genuine regression guard that default_models.json itself — the file ships with, not a fixture —
    stays valid and carries the expected default entries."""
    catalog = CuratedCatalogLoader.load(_DEFAULT_MODELS_PATH)

    assert len(catalog.chat_models) == 9
    assert len(catalog.embedding_models) == 2
    assert catalog.find("hf.co/mistralai/Ministral-3-3B-Instruct-2512-GGUF:Ministral-3-3B-Instruct-2512-Q4_K_M")
    assert catalog.find_embedding("hf.co/nomic-ai/nomic-embed-text-v1.5-GGUF:Q8_0")


def test_catalog_is_loaded_once_at_import_and_matches_the_file():
    """CATALOG (the module-level instance every caller actually uses) is exactly what a fresh load() of the
    same file produces — proving the eager import-time load isn't doing anything different."""
    reloaded = CuratedCatalogLoader.load(_DEFAULT_MODELS_PATH)
    assert [m.tag for m in CATALOG.chat_models] == [m.tag for m in reloaded.chat_models]
    assert [m.tag for m in CATALOG.embedding_models] == [m.tag for m in reloaded.embedding_models]


def test_load_accepts_and_drops_the_note_field(tmp_path):
    """`note` is metadata for a hand-editing admin only (see this module's own docstring) — present in the
    file, never surfaces on the resulting CuratedModel."""
    path = _write(tmp_path, {"chat_models": [{**_VALID_ENTRY, "note": "some provenance note"}], "embedding_models": []})
    catalog = CuratedCatalogLoader.load(path)
    assert not hasattr(catalog.chat_models[0], "note")


def test_load_raises_on_invalid_json_syntax(tmp_path):
    path = tmp_path / "default_models.json"
    path.write_text("{not valid json")
    with pytest.raises(json.JSONDecodeError):
        CuratedCatalogLoader.load(path)


def test_load_raises_on_a_missing_required_field(tmp_path):
    broken = {k: v for k, v in _VALID_ENTRY.items() if k != "architecture"}
    path = _write(tmp_path, {"chat_models": [broken], "embedding_models": []})
    with pytest.raises(ValidationError, match="architecture"):
        CuratedCatalogLoader.load(path)


def test_load_raises_on_a_wrong_field_type(tmp_path):
    broken = {**_VALID_ENTRY, "download_gb": "not-a-number"}
    path = _write(tmp_path, {"chat_models": [broken], "embedding_models": []})
    with pytest.raises(ValidationError, match="download_gb"):
        CuratedCatalogLoader.load(path)


def test_load_raises_on_a_missing_top_level_key(tmp_path):
    path = _write(tmp_path, {"chat_models": [_VALID_ENTRY]})  # no "embedding_models" key at all
    with pytest.raises(KeyError):
        CuratedCatalogLoader.load(path)
