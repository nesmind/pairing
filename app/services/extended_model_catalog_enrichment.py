"""Per-file enrichment for extended_model_catalog_service's admin-added entries, split out for the line cap.

Beyond Hugging Face's repo-level metadata (unreliable per file, see
HuggingFaceCatalogSearch.repo_files), it reads a display name and architecture from the file's own GGUF
header (gguf_probe). The Matricxon support verdict is computed by MatricxonSupportChecker directly; this
module's old wrapper was a thin pass-through.
"""

import re

from app.services import gguf_probe

# Ordered so a "_m"/"_s"/"_l" K-quant suffix is absorbed into the base token. These map to the coarse family
# model_catalog.py's `quantizations` use ("Q4_K", not "Q4_K_M"), matching Matricxon's supported_quantizations
# and settings.js's isRecommendedQuantFile ("q4_k_m" in the filename).
_QUANT_PATTERNS = [
    (re.compile(r"q4_k[_-]?[sml]?", re.IGNORECASE), "Q4_K"),
    (re.compile(r"q5_k[_-]?[sml]?", re.IGNORECASE), "Q5_K"),
    (re.compile(r"q3_k[_-]?[sml]?", re.IGNORECASE), "Q3_K"),
    (re.compile(r"q2_k[_-]?s?", re.IGNORECASE), "Q2_K"),
    (re.compile(r"q6_k", re.IGNORECASE), "Q6_K"),
    (re.compile(r"q8_0", re.IGNORECASE), "Q8_0"),
    (re.compile(r"q5_[01]", re.IGNORECASE), "Q5_0"),
    (re.compile(r"q4_[01]", re.IGNORECASE), "Q4_0"),
    (re.compile(r"bf16", re.IGNORECASE), "BF16"),
    # Negative lookbehind: "bf16" would also match this pattern, containing "f16" (without it every bf16 file guessed
    # both).
    (re.compile(r"(?<!b)f16|fp16", re.IGNORECASE), "F16"),
    (re.compile(r"f32|fp32", re.IGNORECASE), "F32"),
    # I-quants: real GGUF types (matricxon's app.gguf.constants.GGMLQuantizationType, plus llama.cpp's IQ2/IQ3
    # mix suffixes;../matricxon/ROADMAP.md "I-quant and ternary quantization types have no real dequant kernel")
    # that Matricxon has no dequant kernel for. Without these, an IQ2_M file with a supported architecture
    # (nemotron_h) showed the misleading "isn't a recognized GGUF type" instead of "Matricxon doesn't support the
    # IQ2_M quantization". Each maps to itself; I-quants have no convention for collapsing to a family.
    *[
        (re.compile(re.escape(name), re.IGNORECASE), name)
        for name in (
            "IQ1_S",
            "IQ1_M",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ2_M",
            "IQ3_XXS",
            "IQ3_S",
            "IQ3_M",
            "IQ4_NL",
            "IQ4_XS",
        )
    ],
]


class HuggingFaceModelProbe:
    @staticmethod
    def guess_quantizations_from_filename(filename: str) -> list[str] | None:
        """None if nothing recognizable matched ("couldn't guess", unlike "no quant token"). K-quant/F-type patterns
        map to Matricxon-recognized families; I-quants are real GGUF types Matricxon can't load, which shapes
        MatricxonSupportChecker.verdict_for's message.
        """
        matched = {base for pattern, base in _QUANT_PATTERNS if pattern.search(filename)}
        return sorted(matched) or None

    @staticmethod
    async def probe(repo_id: str, filename: str, proxy_url: str | None) -> dict:
        """Best-effort metadata for one file in `repo_id`: {"architecture", "name", "quantizations"}, with
        "architecture"/"name" None when gguf_probe.probe_metadata can't tell. Never raises, so a caller can fall
        back to Hugging Face's coarser repo-level guess instead of failing "Add".
        """
        probed = await gguf_probe.probe_metadata(repo_id, filename, proxy_url)
        return {
            "architecture": probed["architecture"] if probed else None,
            "name": probed["name"] if probed else None,
            "quantizations": HuggingFaceModelProbe.guess_quantizations_from_filename(filename),
        }
