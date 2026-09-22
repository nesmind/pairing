"""Unit tests for app/services/extended_model_catalog_enrichment.py. gguf_probe.probe_metadata is monkeypatched
at its real origin module (see the "module-split import-binding gotcha" this codebase already tracks — this
module calls it via a plain `gguf_probe.probe_metadata(...)` attribute lookup, not a `from x import y` binding),
so no real Hugging Face call happens here."""

import pytest

from app.services import gguf_probe
from app.services.extended_model_catalog_enrichment import HuggingFaceModelProbe

# ---- guess_quantizations_from_filename --------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Ministral-3-3B-Instruct-2512-Q4_K_M.gguf", ["Q4_K"]),
        ("qwen2.5-14b-instruct-q4_k_s.gguf", ["Q4_K"]),
        ("model-q6_k.gguf", ["Q6_K"]),
        ("model-q8_0.gguf", ["Q8_0"]),
        ("model-fp16.gguf", ["F16"]),
        ("model-bf16.gguf", ["BF16"]),
        ("model.F32.gguf", ["F32"]),
    ],
)
def test_guess_quantizations_from_filename_recognizes_common_conventions(filename, expected):
    assert HuggingFaceModelProbe.guess_quantizations_from_filename(filename) == expected


def test_guess_quantizations_from_filename_returns_none_for_an_unrecognized_name():
    assert HuggingFaceModelProbe.guess_quantizations_from_filename("weird-name-nozzle.gguf") is None


# ---- probe ---------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_probe_combines_the_real_probe_with_the_filename_quant_guess(monkeypatch):
    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return {"architecture": "phi2", "name": "moondream2"}

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)

    result = await HuggingFaceModelProbe.probe("org/repo", "model-q4_k_m.gguf", proxy_url=None)

    assert result == {"architecture": "phi2", "name": "moondream2", "quantizations": ["Q4_K"]}


@pytest.mark.asyncio
async def test_probe_degrades_cleanly_when_the_real_probe_fails(monkeypatch):
    async def fake_probe_metadata(_repo_id, _filename, _proxy_url):
        return None

    monkeypatch.setattr(gguf_probe, "probe_metadata", fake_probe_metadata)

    result = await HuggingFaceModelProbe.probe("org/repo", "model-q4_k_m.gguf", proxy_url=None)

    # quantizations is still real, pure, filename-only guessing — unaffected by the network-side probe failing.
    assert result == {"architecture": None, "name": None, "quantizations": ["Q4_K"]}
