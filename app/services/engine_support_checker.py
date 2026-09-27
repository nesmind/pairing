"""
Whether a given engine can actually run a specific model — the generic shape every
engine's own support-checking follows, extracted from
app.services.matricxon_support_checker.MatricxonSupportChecker (Matricxon is still
the only engine with real, checkable support data; see that module for the concrete
subclass). Ollama has no equivalent check at all — it "runs every catalog entry
regardless" — so AlwaysSupportedChecker below is the trivial default every other
engine (including a future serverless one, unless it turns out to need its own real
check) gets for free.

One checker instance is built once per catalog build (see EngineSupportSet.load
below) and its verdict_for/estimated_ram_gb reused for every entry in that catalog —
the same "fetch once, reuse for every model" contract
app.services.model_catalog_service.ChatModelCatalogBuilder and its sibling catalog
builders already kept before this class existed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from app.schemas import EngineName

if TYPE_CHECKING:
    from app.schemas.model_catalog import EngineModelSupport

# Ollama's own rule-of-thumb per-model RAM multiplier (it exposes no better number of
# its own) — the single copy of a formula that used to be duplicated three times
# (app.services.model_catalog_service, app.services.embedding_model_catalog_service,
# and, as a hardcoded literal to dodge a circular import, app.services.
# extended_model_catalog_service._min_ram_gb).
OLLAMA_RAM_ESTIMATE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class SupportVerdict:
    """`.supported`/`.reason` mirror the engine's own entry in
    CatalogEntry.engine_support (see app/schemas/model_catalog.py's
    EngineModelSupport) — `.reason` is always None when supported."""

    supported: bool
    reason: str | None = None


class EngineSupportChecker(ABC):
    """One per engine. `load()` does whatever real I/O this engine's own support
    check needs (Matricxon: a GET /api/health round trip); `unloaded()` is the
    zero-I/O instance used for every engine that ISN'T currently active — asking a
    non-active engine (possibly a paid remote API) about its support surface on
    every catalog load would be a real, unwanted cost for no benefit, since nothing
    is routing traffic there right now anyway (see EngineSupportSet.load, which
    enforces this rule for every engine uniformly)."""

    @classmethod
    @abstractmethod
    async def load(cls, installed_models: list[dict] | None = None) -> Self:
        """`installed_models` — the caller's own already-fetched list_models()
        result, if it has one, to skip a redundant round trip for the same data
        (see MatricxonSupportChecker.load's own docstring for the real, confirmed
        latency bug this avoids)."""

    @classmethod
    def unloaded(cls) -> Self:
        """The engine isn't active right now — no real check was made. Default
        implementation assumes a no-arg constructor; override if a subclass's
        __init__ needs more."""
        return cls()

    def verdict_for(
        self,
        architecture: str | None,
        quantizations: list[str] | None,
        *,
        is_projector: bool = False,
    ) -> SupportVerdict:
        """Default: everything is supported (Ollama's own behavior — it runs every
        catalog entry regardless, with no architecture/quantization registry of its
        own to check against)."""
        return SupportVerdict(True)

    def estimated_ram_gb(
        self,
        tag: str | None,
        *,
        download_gb: float | None = None,
        fallback_min_ram_gb: float = 0.0,
    ) -> float | None:
        """This engine's own real per-model RAM estimate for `tag`, or None when
        unknown/not applicable. Default: unknown — a subclass overrides this only
        when it actually has a way to estimate (Ollama: download_gb * a fixed
        multiplier; Matricxon: a real per-tag figure read off the installed GGUF's
        own tensor shapes)."""
        return None


class AlwaysSupportedChecker(EngineSupportChecker):
    """The trivial default for any engine with no real support-checking of its own
    (InferenceEngine.support_checker's own default) — no I/O, no state."""

    @classmethod
    async def load(cls, installed_models: list[dict] | None = None) -> "AlwaysSupportedChecker":
        return cls()


class OllamaSupportChecker(AlwaysSupportedChecker):
    """Still "always supported" (Ollama has no support registry to check against),
    but knows how to estimate RAM from a model's download size — the one thing
    Ollama's own API can't report directly."""

    def estimated_ram_gb(
        self,
        tag: str | None,
        *,
        download_gb: float | None = None,
        fallback_min_ram_gb: float = 0.0,
    ) -> float | None:
        return round(download_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if download_gb else fallback_min_ram_gb


class EngineSupportSet:
    """Every registered engine's own checker, loaded once per catalog build (see
    ChatModelCatalogBuilder.build and its sibling catalog builders) — the active
    engine's checker actually does its real check; every other engine gets its
    zero-I/O `unloaded()` instance, so a future paid/remote engine's checker is never
    queried while it isn't the one serving traffic (the same rule
    MatricxonSupportChecker.load already enforced for itself alone, now shared by
    every engine uniformly)."""

    def __init__(self, checkers: dict[EngineName, EngineSupportChecker]) -> None:
        self._checkers = checkers

    @classmethod
    async def load(cls, installed_models: list[dict] | None = None) -> "EngineSupportSet":
        # Imported here, not at module level: app.services.engines.registry itself
        # imports app.services.engine_support_checker (for the EngineSupportChecker/
        # AlwaysSupportedChecker names InferenceEngine.support_checker is typed
        # against) — importing the registry back from here at module load time would
        # be circular.
        import asyncio

        from app.services import engine_service
        from app.services.engines.registry import registry

        active = engine_service.current_engine()

        async def _load_one(engine) -> tuple[EngineName, EngineSupportChecker]:
            checker_cls = engine.support_checker
            if engine.name != active:
                return engine.name, checker_cls.unloaded()
            return engine.name, await checker_cls.load(installed_models)

        pairs = await asyncio.gather(*(_load_one(engine) for engine in registry.all()))
        return cls(dict(pairs))

    def engine_names(self) -> list[EngineName]:
        return list(self._checkers)

    def checker_for(self, engine_name: EngineName) -> EngineSupportChecker:
        """Direct access to one engine's own checker — for a caller that needs finer control than support_for's
        "verdict_for + estimated_ram_gb, bundled" convenience gives (see
        app.services.installed_projector_catalog.InstalledProjectorCatalog.entries, which wants a real RAM
        estimate but a fixed, always-True supported verdict regardless of what verdict_for(is_projector=True)
        would say — see that call site's own comment)."""
        return self._checkers[engine_name]

    def support_for(
        self,
        *,
        tag: str | None,
        architecture: str | None,
        quantizations: list[str] | None,
        is_projector: bool = False,
        download_gb: float | None = None,
        min_ram_gb: float = 0.0,
    ) -> dict[EngineName, "EngineModelSupport"]:
        """One EngineModelSupport per registered engine for this one model — the
        shape CatalogEntry.engine_support/EmbeddingCatalogEntry.engine_support store
        directly (see app/schemas/model_catalog.py)."""
        from app.schemas.model_catalog import EngineModelSupport

        result = {}
        for engine_name, checker in self._checkers.items():
            verdict = checker.verdict_for(architecture, quantizations, is_projector=is_projector)
            ram = checker.estimated_ram_gb(tag, download_gb=download_gb, fallback_min_ram_gb=min_ram_gb)
            result[engine_name] = EngineModelSupport(supported=verdict.supported, reason=verdict.reason, min_ram_gb=ram)
        return result
