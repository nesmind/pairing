"""Whether an engine can run a specific model: the generic shape of matricxon_support_checker.MatricxonSupportChecker
(the only engine with real checkable data). Ollama "runs every catalog entry", so AlwaysSupportedChecker is
the trivial default for any other engine, including future serverless ones.

One checker is built per catalog build (EngineSupportSet.load) and its verdict_for/estimated_ram_gb reused
for every entry ("fetch once", as ChatModelCatalogBuilder and sibling builders already did).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from app.schemas import EngineName

if TYPE_CHECKING:
    from app.schemas.model_catalog import EngineModelSupport

# Ollama's rule-of-thumb per-model RAM multiplier (it exposes no better number); the single copy of a formula
# once duplicated in model_catalog_service, embedding_model_catalog_service and, as a literal to avoid a
# circular import, extended_model_catalog_service._min_ram_gb.
OLLAMA_RAM_ESTIMATE_MULTIPLIER = 1.25


@dataclass(frozen=True)
class SupportVerdict:
    """`.supported`/`.reason` mirror the engine's entry in CatalogEntry.engine_support (EngineModelSupport);
    `.reason` is None when supported.
    """

    supported: bool
    reason: str | None = None


class EngineSupportChecker(ABC):
    """One per engine. `load` does the engine's real I/O (Matricxon: a GET /api/health round trip);
    `unloaded` is the zero-I/O instance for inactive engines, so a non-active (possibly paid remote) engine
    isn't queried on every catalog load (enforced by EngineSupportSet.load).
    """

    @classmethod
    @abstractmethod
    async def load(cls, installed_models: list[dict] | None = None) -> Self:
        """`installed_models`: the caller's already-fetched list_models result, to skip a redundant round trip
        (a latency bug, see MatricxonSupportChecker.load).
        """

    @classmethod
    def unloaded(cls) -> Self:
        """For an inactive engine: no real check made. Assumes a no-arg constructor; override if __init__ needs more."""
        return cls()

    def verdict_for(
        self,
        architecture: str | None,
        quantizations: list[str] | None,
        *,
        is_projector: bool = False,
    ) -> SupportVerdict:
        """Default: everything supported (Ollama runs every entry; no architecture/quantization registry)."""
        return SupportVerdict(True)

    def estimated_ram_gb(
        self,
        tag: str | None,
        *,
        download_gb: float | None = None,
        fallback_min_ram_gb: float = 0.0,
    ) -> float | None:
        """This engine's per-model RAM estimate for `tag`, or None if unknown. Default: unknown; subclasses override
        (Ollama: download_gb * a multiplier; Matricxon: from the installed GGUF's tensor shapes).
        """
        return None


class AlwaysSupportedChecker(EngineSupportChecker):
    """The default for engines with no support checking (InferenceEngine.support_checker): no I/O, no state."""

    @classmethod
    async def load(cls, installed_models: list[dict] | None = None) -> "AlwaysSupportedChecker":
        return cls()


class OllamaSupportChecker(AlwaysSupportedChecker):
    """Still "always supported", but estimates RAM from download size, which Ollama's API can't report."""

    def estimated_ram_gb(
        self,
        tag: str | None,
        *,
        download_gb: float | None = None,
        fallback_min_ram_gb: float = 0.0,
    ) -> float | None:
        return round(download_gb * OLLAMA_RAM_ESTIMATE_MULTIPLIER, 1) if download_gb else fallback_min_ram_gb


class EngineSupportSet:
    """Every registered engine's checker, loaded once per catalog build (ChatModelCatalogBuilder.build and
    siblings). The active engine does its real check; others get a zero-I/O `unloaded` instance, so a
    paid/remote engine isn't queried while inactive (the rule MatricxonSupportChecker.load enforced alone).
    """

    def __init__(self, checkers: dict[EngineName, EngineSupportChecker]) -> None:
        self._checkers = checkers

    @classmethod
    async def load(cls, installed_models: list[dict] | None = None) -> "EngineSupportSet":
        # Imported here to avoid a cycle: engines.registry imports engine_support_checker (for the checker names
        # InferenceEngine.support_checker uses).
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
        """One engine's checker, for finer control than support_for (e.g. InstalledProjectorCatalog.entries wants a
        real RAM estimate but an always-True verdict regardless of verdict_for(is_projector=True)).
        """
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
        """One EngineModelSupport per registered engine for this model; stored directly in
        CatalogEntry/EmbeddingCatalogEntry.engine_support (schemas/model_catalog.py).
        """
        from app.schemas.model_catalog import EngineModelSupport

        result = {}
        for engine_name, checker in self._checkers.items():
            verdict = checker.verdict_for(architecture, quantizations, is_projector=is_projector)
            ram = checker.estimated_ram_gb(tag, download_gb=download_gb, fallback_min_ram_gb=min_ram_gb)
            result[engine_name] = EngineModelSupport(supported=verdict.supported, reason=verdict.reason, min_ram_gb=ram)
        return result
