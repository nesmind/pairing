"""Unit tests for app/services/engine_support_checker.py — the generic AlwaysSupportedChecker/
OllamaSupportChecker defaults, and EngineSupportSet (the "only the active engine's checker does real I/O" rule
that used to live inside MatricxonSupportChecker.load itself — see that module's own test file for the parts of
its behavior that are still checker-specific)."""

import pytest

from app.services import engine_service, matricxon_client
from app.services.engine_support_checker import (
    AlwaysSupportedChecker,
    EngineSupportSet,
    OllamaSupportChecker,
    SupportVerdict,
)


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


def test_always_supported_checker_supports_everything():
    checker = AlwaysSupportedChecker()
    assert checker.verdict_for("anything", ["Q4_K"]) == SupportVerdict(True)
    assert checker.verdict_for(None, None, is_projector=True) == SupportVerdict(True)


def test_always_supported_checker_estimates_no_ram():
    assert AlwaysSupportedChecker().estimated_ram_gb("some-tag") is None


def test_ollama_checker_estimates_ram_from_download_size():
    checker = OllamaSupportChecker()
    assert checker.estimated_ram_gb("tag", download_gb=4.0) == 5.0  # 4.0 * 1.25


def test_ollama_checker_falls_back_when_download_size_is_unknown():
    checker = OllamaSupportChecker()
    assert checker.estimated_ram_gb("tag", download_gb=None, fallback_min_ram_gb=2.5) == 2.5


@pytest.mark.asyncio
async def test_engine_support_set_only_gives_the_active_engine_a_real_load(monkeypatch):
    """The real bug class this guards against: asking a non-active engine (possibly a paid remote API) about
    its support surface on every catalog load — the rule MatricxonSupportChecker.load used to enforce on
    itself, now shared by every engine uniformly."""
    engine_service._cached_engine = "ollama"

    async def _fail_if_called():
        raise AssertionError("Matricxon must not be queried while Ollama is active")

    monkeypatch.setattr(matricxon_client, "get_capabilities", _fail_if_called)
    monkeypatch.setattr(matricxon_client, "list_models", _fail_if_called)

    support_set = await EngineSupportSet.load()

    result = support_set.support_for(tag="some-tag", architecture="mistral3", quantizations=["Q4_K"])
    assert result["ollama"].supported is True
    assert result["matricxon"].supported is False
    assert "Could not reach Matricxon" in result["matricxon"].reason


@pytest.mark.asyncio
async def test_engine_support_set_gives_the_active_engine_a_real_load(monkeypatch):
    engine_service._cached_engine = "matricxon"
    caps = {"supported_architectures": ["mistral3"], "supported_quantizations": ["Q4_K"]}

    async def _capabilities():
        return caps

    async def _list_models():
        return [{"name": "some-tag", "capabilities": ["completion"], "estimated_ram_gb": 3.8}]

    monkeypatch.setattr(matricxon_client, "get_capabilities", _capabilities)
    monkeypatch.setattr(matricxon_client, "list_models", _list_models)

    support_set = await EngineSupportSet.load()

    result = support_set.support_for(tag="some-tag", architecture="mistral3", quantizations=["Q4_K"])
    assert result["matricxon"].supported is True
    assert result["matricxon"].min_ram_gb == 3.8


def test_engine_support_set_min_ram_falls_back_when_the_engine_has_no_estimate():
    support_set = EngineSupportSet({"ollama": OllamaSupportChecker()})

    result = support_set.support_for(
        tag="tag", architecture="x", quantizations=["Q4_K"], download_gb=None, min_ram_gb=1.2
    )

    assert result["ollama"].min_ram_gb == 1.2
