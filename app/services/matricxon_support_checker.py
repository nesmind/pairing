"""Whether Matricxon (the sibling from-scratch GGUF runtime../matricxon, matricxon_client) can run a model,
computed live against its current support surface rather than a hand-kept flag that drifts when its
registries change (../matricxon/ROADMAP.md "Expose supported architectures/quantizations").

One `MatricxonSupportChecker` is built per catalog build (`await MatricxonSupportChecker.load`: one
network round trip, or none if Matricxon isn't active, see `load`) and `.verdict_for(...)` reused for every
entry: the "fetch once, reuse" contract the three catalog builders (ChatModelCatalogBuilder,
EmbeddingModelCatalogService, ExtendedModelCatalog) each duplicated before this class.
"""

import asyncio

from app.services import matricxon_client
from app.services.engine_support_checker import EngineSupportChecker, SupportVerdict
from app.services.matricxon_client import MatricxonError

__all__ = ["MatricxonSupportChecker", "SupportVerdict"]


class MatricxonSupportChecker(EngineSupportChecker):
    def __init__(self, capabilities: dict | None, installed_info: dict[str, dict] | None) -> None:
        self._capabilities = capabilities
        self._installed_info = installed_info or {}

    @classmethod
    def unloaded(cls) -> "MatricxonSupportChecker":
        """The "not active / unreachable" instance, the same shape `load` falls back to when Matricxon can't be
        reached, so callers needn't tell them apart (verdict_for's "Could not reach Matricxon..." branch).
        """
        return cls(None, None)

    @classmethod
    async def load(cls, installed_models: list[dict] | None = None) -> "MatricxonSupportChecker":
        """A round trip to Matricxon's GET /api/health (and GET /api/tags unless `installed_models` is given). The
        caller EngineSupportSet.load decides whether this engine is active at all (asking an inactive, possibly
        paid remote engine on every catalog load is needless cost).

        `installed_models`: pass the caller's list_models result (all three catalog builders already call it) to skip
        a redundant GET /api/tags: with Matricxon active, that call is matricxon_client.list_models (via
        inference_client.list_models). On one Settings > Model load made up to six /api/tags requests
        for three pieces of data: extra latency, and a risk of timing out since /api/tags can be slow while Matricxon
        generates. The two remaining calls (this and capabilities) run concurrently.
        """
        if installed_models is not None:
            return cls(await cls._capabilities_or_none(), cls._installed_info_from(installed_models))
        capabilities, installed_info = await asyncio.gather(cls._capabilities_or_none(), cls._installed_info_or_none())
        return cls(capabilities, installed_info)

    @staticmethod
    async def _capabilities_or_none() -> dict | None:
        """matricxon_client.get_capabilities's response, or None if Matricxon is unreachable. Best-effort: the rest
        of the catalog still renders; verdict_for treats None as "not installed".
        """
        try:
            return await matricxon_client.get_capabilities()
        except MatricxonError:
            return None

    @staticmethod
    def _installed_info_from(installed: list[dict]) -> dict[str, dict]:
        """{tag: {"capabilities": [...], "estimated_ram_gb": float}} from a models list in hand (load's
        `installed_models`) or fetched (_installed_info_or_none). Feeds Matricxon's own per-tag RAM estimate: it
        dequantizes to bf16, so it runs well above this app's static, Ollama-shaped min_ram_gb (a Ministral-3B file
        needs ~7.5GB, not ~3GB).
        """
        return {
            m["name"]: {"capabilities": m.get("capabilities", []), "estimated_ram_gb": m.get("estimated_ram_gb")}
            for m in installed
        }

    @staticmethod
    async def _installed_info_or_none() -> dict[str, dict] | None:
        """GET /api/tags on Matricxon specifically (not inference_client's active-engine list_models); only when the
        caller had no models list (every real caller today passes one to load).
        """
        try:
            installed = await matricxon_client.list_models()
        except MatricxonError:
            return None
        return MatricxonSupportChecker._installed_info_from(installed)

    def estimated_ram_gb(
        self,
        tag: str | None,
        *,
        download_gb: float | None = None,
        fallback_min_ram_gb: float = 0.0,
    ) -> float | None:
        """Matricxon's per-tag RAM estimate; None if the tag isn't confirmed installed (an un-pulled tag's need can't be
        computed without its GGUF tensor shapes, never guessed). `download_gb`/`fallback_min_ram_gb` are part of the
        EngineSupportChecker signature (OllamaSupportChecker uses them) and ignored here.
        """
        if tag is None:
            return None
        return self._installed_info.get(tag, {}).get("estimated_ram_gb")

    def verdict_for(
        self,
        architecture: str | None,
        quantizations: list[str] | None,
        *,
        is_projector: bool = False,
    ) -> SupportVerdict:
        """Whether Matricxon implements `architecture`/`quantizations`: the base text-chat requirement, the same for
        vision entries (CatalogEntry.vision). matricxon_supported is just this check, not a claim about image
        support: gating it on vision pairing made an installed, working chat model (moondream2) show "Not
        supported".

        `is_projector` (an mmproj sidecar, never a standalone chat model; huggingface_client's is_projector) always
        wins: architecture/quantization support is meaningless for a file never meant to run alone.
        `architecture=None` (an admin-added entry whose GGUF header probe failed,
        extended_model_catalog_enrichment.HuggingFaceModelProbe) gets an honest "unknown" reason, not a
        misleading "doesn't implement architecture 'None'".
        """
        if is_projector:
            return SupportVerdict(
                False,
                "A vision projector, pulled alongside its paired text model — never run as a chat model on its own.",
            )
        if architecture is None:
            return SupportVerdict(
                False, "Not verified against Matricxon — architecture unknown for an admin-added model."
            )
        if self._capabilities is None:
            return SupportVerdict(
                False, "Could not reach Matricxon to check its supported architectures/quantizations right now."
            )
        if architecture not in self._capabilities.get("supported_architectures", []):
            return SupportVerdict(False, f'Matricxon doesn\'t implement the "{architecture}" architecture yet.')
        if not quantizations:
            return SupportVerdict(
                False, "This file's quantization isn't a recognized GGUF type — Matricxon support can't be verified."
            )
        supported_quantizations = self._capabilities.get("supported_quantizations", [])
        unsupported = [q for q in quantizations if q not in supported_quantizations]
        if unsupported:
            return SupportVerdict(
                False, f"Matricxon doesn't support the {', '.join(unsupported)} quantization this file uses."
            )
        return SupportVerdict(True)
