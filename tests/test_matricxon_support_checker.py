"""Tests for services/matricxon_support_checker.py. verdict_for is pure, covered on a hand-built checker; load
separately with matricxon_client/engine_service patched on this module's binding (module-split gotcha).
"""

import pytest

from app.services import engine_service, matricxon_client
from app.services.matricxon_support_checker import MatricxonSupportChecker, SupportVerdict

_CAPS = {
    "supported_architectures": ["mistral3", "bert", "nomic-bert"],
    "supported_quantizations": ["F32", "F16", "Q8_0", "Q4_0", "Q4_1", "Q4_K", "Q5_K", "Q6_K"],
}


async def _async_return(value):
    return value


# verdict_for (pure)


def test_verdict_for_true_when_architecture_and_every_quantization_are_supported():
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for("mistral3", ["Q4_K", "Q6_K"])
    assert verdict == SupportVerdict(True)


def test_verdict_for_false_for_an_unsupported_architecture():
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for("llama", ["Q4_0"])
    assert verdict.supported is False
    assert "llama" in verdict.reason


def test_verdict_for_false_when_any_underlying_quantization_is_unsupported():
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for("mistral3", ["Q4_K", "Q2_K"])
    assert verdict.supported is False
    assert "Q2_K" in verdict.reason


def test_verdict_for_false_for_an_unrecognized_quantization_format():
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for("mistral3", None)
    assert verdict.supported is False
    assert "quantization" in verdict.reason.lower()


def test_verdict_for_false_when_matricxon_was_unreachable():
    checker = MatricxonSupportChecker(None, None)
    verdict = checker.verdict_for("mistral3", ["Q4_K"])
    assert verdict.supported is False
    assert verdict.reason


def test_verdict_for_always_rejects_a_projector_regardless_of_architecture():
    """is_projector wins even with a supported architecture: a projector sidecar was never a standalone chat model."""
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for("mistral3", ["Q4_K"], is_projector=True)
    assert verdict.supported is False
    assert "never run as a chat model" in verdict.reason


def test_verdict_for_gives_an_honest_reason_when_architecture_is_unknown():
    """A None architecture (an admin-added entry whose GGUF header couldn't be probed) gets its own reason, not the
    misleading 'doesn't implement architecture None'."""
    checker = MatricxonSupportChecker(_CAPS, None)
    verdict = checker.verdict_for(None, None)
    assert verdict.supported is False
    assert "architecture unknown" in verdict.reason


def test_verdict_for_ignores_vision_pairing_entirely():
    """matricxon_supported is just the architecture+quantization check, whether or not the entry is a vision model or
    its mmproj is paired. Gating on pairing made an installed, working chat model (moondream2) show "Not supported".
    Same verdict whether or not Matricxon's installed capabilities include "vision" for this tag.
    """
    checker = MatricxonSupportChecker(_CAPS, {"some-tag": {"capabilities": ["completion"]}})
    assert checker.verdict_for("mistral3", ["Q4_K"]) == SupportVerdict(True)

    checker = MatricxonSupportChecker(_CAPS, {"some-tag": {"capabilities": ["completion", "vision"]}})
    assert checker.verdict_for("mistral3", ["Q4_K"]) == SupportVerdict(True)


# estimated_ram_gb


def test_estimated_ram_gb_reads_the_installed_info():
    checker = MatricxonSupportChecker(_CAPS, {"some-tag": {"estimated_ram_gb": 7.5}})
    assert checker.estimated_ram_gb("some-tag") == 7.5


def test_estimated_ram_gb_none_for_an_uninstalled_tag():
    checker = MatricxonSupportChecker(_CAPS, {})
    assert checker.estimated_ram_gb("some-tag") is None


# load


@pytest.fixture(autouse=True)
def _reset_engine_cache():
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE
    yield
    engine_service._cached_engine = engine_service.DEFAULT_ENGINE


def test_unloaded_gives_an_honest_could_not_reach_verdict():
    """The "not active" instance a caller uses instead of load when the engine isn't active
    (EngineSupportSet.load now owns the "only the active engine does real I/O" rule this class enforced in load).
    """
    checker = MatricxonSupportChecker.unloaded()

    assert checker.verdict_for("mistral3", ["Q4_K"]).supported is False


@pytest.mark.asyncio
async def test_load_fetches_both_when_matricxon_is_active(monkeypatch):
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(matricxon_client, "get_capabilities", lambda: _async_return(_CAPS))
    monkeypatch.setattr(
        matricxon_client,
        "list_models",
        lambda: _async_return([{"name": "some-tag", "capabilities": ["completion"], "estimated_ram_gb": 3.8}]),
    )

    checker = await MatricxonSupportChecker.load()

    assert checker.verdict_for("mistral3", ["Q4_K"]) == SupportVerdict(True)
    assert checker.estimated_ram_gb("some-tag") == 3.8


@pytest.mark.asyncio
async def test_load_degrades_cleanly_when_matricxon_is_unreachable(monkeypatch):
    engine_service._cached_engine = "matricxon"

    async def _raise():
        raise matricxon_client.MatricxonError("down")

    monkeypatch.setattr(matricxon_client, "get_capabilities", _raise)
    monkeypatch.setattr(matricxon_client, "list_models", _raise)

    checker = await MatricxonSupportChecker.load()

    verdict = checker.verdict_for("mistral3", ["Q4_K"])
    assert verdict.supported is False
    assert "Could not reach Matricxon" in verdict.reason


@pytest.mark.asyncio
async def test_load_skips_its_own_list_models_call_when_given_an_already_fetched_list(monkeypatch):
    """Bug found: every catalog builder (ChatModelCatalogBuilder, EmbeddingModelCatalogService,
    ExtendedModelCatalog.build) already fetches list_models, and load fetched it *again*, doubling /api/tags on
    every catalog page load (and, since it can be slow mid-generation, the chance of the page timing out). Passing the
    caller's list must skip it.
    """
    engine_service._cached_engine = "matricxon"
    monkeypatch.setattr(matricxon_client, "get_capabilities", lambda: _async_return(_CAPS))

    async def _fail_if_called():
        raise AssertionError("list_models must not be called again when installed_models is already given")

    monkeypatch.setattr(matricxon_client, "list_models", _fail_if_called)

    checker = await MatricxonSupportChecker.load(
        [{"name": "some-tag", "capabilities": ["completion"], "estimated_ram_gb": 3.8}]
    )

    assert checker.estimated_ram_gb("some-tag") == 3.8
