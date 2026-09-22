"""
Real per-file enrichment for app.services.extended_model_catalog_service's admin-added "browse more models"
entries — split out purely to keep that file under CLAUDE.md's line cap.

Buys one real thing over trusting Hugging Face's own repo-level metadata alone (see
app.services.huggingface_client.HuggingFaceCatalogSearch.repo_files' own docstring on why that's unreliable
per-file, confirmed live against a real multi-file repo): a real display name/architecture read directly from
the specific file's own GGUF header (app.services.gguf_probe). The real Matricxon-support verdict for that
architecture is computed by app.services.matricxon_support_checker.MatricxonSupportChecker directly now — this
module used to also wrap that check, but it was never more than a thin pass-through.
"""

import re

from app.services import gguf_probe

# Ordered so a "_m"/"_s"/"_l" K-quant suffix is absorbed into the same base token rather than left unmatched —
# every token maps to the same coarse family app/model_catalog.py's own hand-curated `quantizations` lists
# already use ("Q4_K", never "Q4_K_M"/"Q4_K_S" separately), since Matricxon's own supported_quantizations list
# is keyed the same coarse way. The same "q4_k_m in the filename" convention
# app/static/js/settings.js's own isRecommendedQuantFile already relies on for its "Recommended" badge.
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
    # Negative lookbehind: "bf16" (its own pattern above) would otherwise also match this one, since it
    # literally contains the substring "f16" — confirmed live, without it every bf16 file guessed both.
    (re.compile(r"(?<!b)f16|fp16", re.IGNORECASE), "F16"),
    (re.compile(r"f32|fp32", re.IGNORECASE), "F32"),
]


class HuggingFaceModelProbe:
    @staticmethod
    def guess_quantizations_from_filename(filename: str) -> list[str] | None:
        """None if nothing recognizable matched — lets a caller tell "genuinely couldn't guess" apart from
        "known to use no listed quant token" (never actually returned here, since every pattern maps to a real,
        Matricxon-recognized family)."""
        matched = {base for pattern, base in _QUANT_PATTERNS if pattern.search(filename)}
        return sorted(matched) or None

    @staticmethod
    async def probe(repo_id: str, filename: str, proxy_url: str | None) -> dict:
        """Best-effort real metadata for one specific file in `repo_id` — {"architecture", "name",
        "quantizations"}, "architecture"/"name" None whenever gguf_probe.probe_metadata couldn't determine them
        (network hiccup, corrupt/unusual file, ...). Never raises: a caller always has *something* to fall back
        to (Hugging Face's own coarser repo-level guess) instead of failing the whole "Add" action over this
        best-effort enrichment."""
        probed = await gguf_probe.probe_metadata(repo_id, filename, proxy_url)
        return {
            "architecture": probed["architecture"] if probed else None,
            "name": probed["name"] if probed else None,
            "quantizations": HuggingFaceModelProbe.guess_quantizations_from_filename(filename),
        }
