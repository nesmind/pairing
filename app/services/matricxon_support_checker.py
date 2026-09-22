"""
Whether Matricxon (the sibling from-scratch GGUF runtime, ../matricxon — see app.services.matricxon_client) can
actually run a given model — computed live against Matricxon's own real, current support surface instead of a
hand-maintained flag that silently drifts out of sync the moment Matricxon's own registries change (see
../matricxon/ROADMAP.md's "Expose supported architectures/quantizations" entry for the full history).

One `MatricxonSupportChecker` is built once per catalog build (`await MatricxonSupportChecker.load()`, one real
network round trip to Matricxon — or none at all if it isn't the active engine, see `load`'s own docstring) and
its `.verdict_for(...)` reused for every entry in that catalog — the same "fetch once, reuse for every model"
contract the three catalog builders (app.services.model_catalog_service.ChatModelCatalogBuilder,
embedding_model_catalog_service.EmbeddingModelCatalogService,
app.services.extended_model_catalog_service.ExtendedModelCatalog) already kept as separate functions before
this class existed — consolidated here since all three were duplicating the exact same fetch-then-check
sequence.
"""

from dataclasses import dataclass

from app.services import engine_service, matricxon_client
from app.services.matricxon_client import MatricxonError


@dataclass(frozen=True)
class SupportVerdict:
    """`.supported` mirrors CatalogEntry.matricxon_supported; `.reason` mirrors
    CatalogEntry.matricxon_unsupported_reason (always None when supported)."""

    supported: bool
    reason: str | None = None


class MatricxonSupportChecker:
    def __init__(self, capabilities: dict | None, installed_info: dict[str, dict] | None) -> None:
        self._capabilities = capabilities
        self._installed_info = installed_info or {}

    @classmethod
    async def load(cls) -> "MatricxonSupportChecker":
        """Skips both real Matricxon calls entirely (an "unreachable" checker, capabilities=None) unless
        Matricxon is the currently active engine — asking about it regardless used to mean every catalog load
        made a real network round trip to a server nobody was actually using, worth nothing while Ollama serves
        every request (confirmed a real, unwanted cost, not just theoretical)."""
        if engine_service.current_engine() != "matricxon":
            return cls(None, None)
        return cls(await cls._capabilities_or_none(), await cls._installed_info_or_none())

    @staticmethod
    async def _capabilities_or_none() -> dict | None:
        """matricxon_client.get_capabilities()'s own response, or None if Matricxon couldn't be reached —
        best-effort: a caller still needs to render the rest of the catalog even if Matricxon itself happens to
        be down right now; verdict_for below treats None the same as "not installed," not a hard failure."""
        try:
            return await matricxon_client.get_capabilities()
        except MatricxonError:
            return None

    @staticmethod
    async def _installed_info_or_none() -> dict[str, dict] | None:
        """{tag: {"capabilities": [...], "estimated_ram_gb": float}} for every model Matricxon itself currently
        has installed (GET /api/tags on Matricxon specifically, not app.services.inference_client's
        active-engine-dispatched list_models). Feeds Matricxon's own real per-tag RAM estimate (estimated_ram_gb
        below) — its always-dequantize-to-bf16 real requirement runs meaningfully higher than this app's own
        static, Ollama-shaped min_ram_gb estimate (confirmed: a real Ministral-3B file needs ~7.5GB on
        Matricxon, not the ~3GB that estimate implies)."""
        try:
            installed = await matricxon_client.list_models()
        except MatricxonError:
            return None
        return {
            m["name"]: {"capabilities": m.get("capabilities", []), "estimated_ram_gb": m.get("estimated_ram_gb")}
            for m in installed
        }

    def estimated_ram_gb(self, tag: str) -> float | None:
        """Matricxon's own real per-tag RAM estimate — None if this tag isn't confirmed installed there (an
        un-pulled tag's real requirement can't be computed without its real GGUF file to read tensor shapes
        from, so this is never a guess)."""
        return self._installed_info.get(tag, {}).get("estimated_ram_gb")

    def verdict_for(
        self,
        architecture: str | None,
        quantizations: list[str] | None,
        *,
        is_projector: bool = False,
    ) -> SupportVerdict:
        """Whether Matricxon implements `architecture`/`quantizations` — the base text-chat requirement, the
        same thing regardless of whether the entry also happens to be a vision model (see CatalogEntry.vision's
        own docstring: matricxon_supported here is deliberately just this architecture+quantization check, not
        a claim about whether a vision entry's image support specifically is confirmed paired — confirmed live
        (2026-09-21) that gating this field on vision-pairing too made an installed, demonstrably-working chat
        model (moondream2, text half confirmed running) show a plain "Not supported" badge, which reads as "this
        doesn't work at all" and is wrong for a model whose text chat clearly does).

        `is_projector` (a vision-projector/mmproj sidecar, never a standalone chat model — see
        app.services.huggingface_client's own is_projector docstring) always wins: architecture/quantization
        support means nothing for a file that was never meant to run on its own. `architecture=None` (an
        admin-added entry whose real GGUF header couldn't be probed — see
        app.services.extended_model_catalog_enrichment.HuggingFaceModelProbe) gets its own honest "unknown"
        reason rather than a misleading "doesn't implement architecture 'None'"."""
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
