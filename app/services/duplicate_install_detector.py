"""Catches the same GGUF file pulled twice under different Hugging Face repo names:
moondream/moondream2-gguf (curated) and moondream/moondream-2b--4bit (admin-added) were
byte-identical mirrors, and the second showed as a confusing "Phi2 · unknown" row, being absent from
app/model_catalog.py.
"""

from app.services.inference_client import InferenceError, list_models

# Loose enough to absorb model_catalog.py's rounded download_gb (e.g. "1.7" for ~1.696GB), tight enough that
# different models almost never match (the moondream duplicate differed by 0%).
_SIZE_TOLERANCE = 0.03


class DuplicateInstallDetector:
    @staticmethod
    async def find(tag: str, architecture: str | None, download_gb: float | None) -> str | None:
        """The `name` of an installed model that looks like the same file as `tag` (same engine-reported architecture,
        same download size within `_SIZE_TOLERANCE`), or None if `architecture`/`download_gb` are unknown or
        nothing matches. Never matches `tag` itself (re-pulling is the same install). Best-effort: an unreachable
        engine must not block a pull (like MatricxonSupportChecker._capabilities_or_none).
        """
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
