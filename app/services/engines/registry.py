"""The fixed set of InferenceEngine instances, replacing inference_client's two-way ternary. Listed
explicitly (not decorator self-registration) so they're visible in one place and independent of import
order; a third engine is one adapter file plus one line here.
"""

from app.schemas import EngineName
from app.services.engines.base import EngineCapabilityError, InferenceEngine
from app.services.engines.cloudflare_engine import CloudflareEngine
from app.services.engines.digitalocean_engine import DigitalOceanEngine
from app.services.engines.matricxon_engine import MatricxonEngine
from app.services.engines.ollama_engine import OllamaEngine
from app.services.engines.runpod_engine import RunPodEngine


class EngineRegistry:
    def __init__(self, engines: list[InferenceEngine]) -> None:
        self._by_name: dict[EngineName, InferenceEngine] = {}
        for engine in engines:
            if engine.name in self._by_name:
                raise ValueError(f"Duplicate engine name registered: {engine.name!r}")
            self._by_name[engine.name] = engine

    def get(self, name: EngineName) -> InferenceEngine:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"No engine registered under {name!r} — registered: {sorted(self._by_name)}") from None

    def active(self) -> InferenceEngine:
        # Imported here: engine_service doesn't import this module, so no cycle, but local so this module (imported
        # by engine_support_checker.py) needn't track engine_service's imports.
        from app.services import engine_service

        return self.get(engine_service.current_engine())

    def all(self) -> list[InferenceEngine]:
        return list(self._by_name.values())

    def names(self) -> list[EngineName]:
        return list(self._by_name)

    def all_error_types(self) -> tuple[type[Exception], ...]:
        """The union of every engine's error types plus EngineCapabilityError (from base.py's "unsupported" stubs);
        inference_client catches it, replacing `except (OllamaError, MatricxonError)`.
        """
        types: set[type[Exception]] = {EngineCapabilityError}
        for engine in self._by_name.values():
            types.update(engine.error_types)
        return tuple(types)


registry = EngineRegistry([OllamaEngine(), MatricxonEngine(), RunPodEngine(), CloudflareEngine(), DigitalOceanEngine()])
