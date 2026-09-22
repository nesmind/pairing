"""Unit tests for app/services/duplicate_install_detector.py. list_models is monkeypatched on this module's own
imported binding (not inference_client's origin — see the "module-split import-binding gotcha" this codebase
already tracks), so no real Ollama/Matricxon call happens here."""

import pytest

from app.services import duplicate_install_detector as svc
from app.services.duplicate_install_detector import DuplicateInstallDetector
from app.services.inference_client import InferenceError

_INSTALLED = [{"name": "hf.co/moondream/moondream2-gguf:x", "details": {"family": "phi2"}, "size": 2_839_534_976}]


async def _async_return(value):
    return value


@pytest.mark.asyncio
async def test_find_returns_none_when_nothing_installed(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return([]))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.84) is None


@pytest.mark.asyncio
async def test_find_returns_none_when_architecture_is_unknown(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", None, 2.84) is None


@pytest.mark.asyncio
async def test_find_returns_none_when_download_gb_is_unknown(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", None) is None


@pytest.mark.asyncio
async def test_find_matches_the_same_architecture_and_size_within_tolerance(monkeypatch):
    """The real case this exists for: the exact same GGUF file (byte-identical size) pulled under a different
    Hugging Face repo name — confirmed live (2026-09-21) with moondream2-gguf vs moondream-2b-2025-04-14-4bit."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.84) == "hf.co/moondream/moondream2-gguf:x"


@pytest.mark.asyncio
async def test_find_absorbs_a_hand_typed_catalog_figures_rounding(monkeypatch):
    """app/model_catalog.py's own hand-typed download_gb is rounded to one decimal — a real ~2.84GB file typed
    as "2.8" must still match, not get missed over rounding noise."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.8) == "hf.co/moondream/moondream2-gguf:x"


@pytest.mark.asyncio
async def test_find_returns_none_when_architecture_matches_but_size_is_far_off(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 14.0) is None


@pytest.mark.asyncio
async def test_find_returns_none_when_size_matches_but_architecture_differs(monkeypatch):
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "llama", 2.84) is None


@pytest.mark.asyncio
async def test_find_excludes_the_tag_itself(monkeypatch):
    """Re-pulling an already-installed tag is the same install, not a duplicate under a different name."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("hf.co/moondream/moondream2-gguf:x", "phi2", 2.84) is None


@pytest.mark.asyncio
async def test_find_returns_none_when_the_engine_is_unreachable(monkeypatch):
    """Best-effort: an unreachable engine must never block a pull that would otherwise succeed (or fail later
    with its own clear error) — same "can't verify, so don't fail closed" stance
    MatricxonSupportChecker._capabilities_or_none already takes."""

    async def fake_list_models():
        raise InferenceError("down")

    monkeypatch.setattr(svc, "list_models", fake_list_models)
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.84) is None
