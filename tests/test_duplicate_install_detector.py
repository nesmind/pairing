"""Tests for services/duplicate_install_detector.py. list_models is monkeypatched on this module's binding (not
inference_client's origin; module-split import-binding gotcha), so no real engine call.
"""

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
    """The case it exists for: the same GGUF file (byte-identical size) under a different Hugging Face repo name
    (moondream2-gguf vs moondream-2b--4bit).
    """
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.84) == "hf.co/moondream/moondream2-gguf:x"


@pytest.mark.asyncio
async def test_find_absorbs_a_hand_typed_catalog_figures_rounding(monkeypatch):
    """model_catalog.py's hand-typed download_gb is rounded to one decimal: a ~2.84GB file typed "2.8" must still
    match."""
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
    """Re-pulling an installed tag is the same install, not a duplicate under another name."""
    monkeypatch.setattr(svc, "list_models", lambda: _async_return(_INSTALLED))
    assert await DuplicateInstallDetector.find("hf.co/moondream/moondream2-gguf:x", "phi2", 2.84) is None


@pytest.mark.asyncio
async def test_find_returns_none_when_the_engine_is_unreachable(monkeypatch):
    """Best-effort: an unreachable engine must not block a pull that would otherwise succeed (or fail later clearly),
    as MatricxonSupportChecker._capabilities_or_none.
    """

    async def fake_list_models():
        raise InferenceError("down")

    monkeypatch.setattr(svc, "list_models", fake_list_models)
    assert await DuplicateInstallDetector.find("new-tag", "phi2", 2.84) is None
