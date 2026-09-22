"""
Catches the same underlying GGUF file pulled twice under two different Hugging Face repo names — confirmed
live (2026-09-21): moondream/moondream2-gguf (curated) and moondream/moondream-2b-2025-04-14-4bit (admin-added)
turned out to be byte-identical, just mirrored repos, and the second copy sat in the Model list as a confusing
"Phi2 · unknown" row with no real display metadata since it's not in app/model_catalog.py's curated list.
"""

from app.services.inference_client import InferenceError, list_models

# Loose enough to absorb app/model_catalog.py's hand-typed, rounded download_gb figures (e.g. "1.7" for a real
# ~1.696GB file); still tight enough that two genuinely different models essentially never land inside it by
# chance — the real moondream duplicate matched exactly (0% difference).
_SIZE_TOLERANCE = 0.03


class DuplicateInstallDetector:
    @staticmethod
    async def find(tag: str, architecture: str | None, download_gb: float | None) -> str | None:
        """The `name` of an already-installed model that looks like the same file as `tag` (same architecture
        the active engine itself reports, same real download size within `_SIZE_TOLERANCE`) — or None if
        `architecture`/`download_gb` aren't known (nothing to compare) or no installed model matches. Never
        matches `tag` against itself — re-pulling an already-installed tag isn't a duplicate under a different
        name, it's the same install. Best-effort: an unreachable engine here must never block a pull that would
        otherwise succeed (or fail later with its own clear error) — same "can't verify, so don't fail closed"
        stance as app.services.matricxon_support_checker.MatricxonSupportChecker._capabilities_or_none."""
        if architecture is None or not download_gb:
            return None
        target_bytes = download_gb * 1_000_000_000
        try:
            installed = await list_models()
        except InferenceError:
            return None
        for model in installed:
            if model["name"] == tag:
                continue
            if model.get("details", {}).get("family") != architecture:
                continue
            size = model.get("size") or 0
            if size and abs(size - target_bytes) / target_bytes <= _SIZE_TOLERANCE:
                return model["name"]
        return None
