"""
The fixed set of InferenceEngine instances this app knows about — app.services.
inference_client's replacement for its old two-way "ollama" vs "matricxon" ternary.
Engines are listed explicitly here (not self-registering via a decorator on each
adapter module) so "which engines exist" is visible in one place and never depends on
import order — adding a third engine in stage 2 is one new adapter file plus one line
below, nothing else in this module or in inference_client.py changes.
"""

from app.schemas import EngineName
from app.services.engines.base import EngineCapabilityError, InferenceEngine
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
        # Imported here, not at module level: app.services.engine_service doesn't
        # import this module, so there's no real cycle — kept local anyway so this
        # module (imported by engine_support_checker.py) never has to care about
        # engine_service's own import surface.
        from app.services import engine_service

        return self.get(engine_service.current_engine())

    def all(self) -> list[InferenceEngine]:
        return list(self._by_name.values())

    def names(self) -> list[EngineName]:
        return list(self._by_name)

    def all_error_types(self) -> tuple[type[Exception], ...]:
        """The union of every registered engine's own error type(s), plus
        EngineCapabilityError (raised by an engine's own default "this isn't
        supported" stubs — see base.py) — app.services.inference_client catches this
        whole tuple, exactly replacing its old `except (OllamaError,
        MatricxonError)`."""
        types: set[type[Exception]] = {EngineCapabilityError}
        for engine in self._by_name.values():
            types.update(engine.error_types)
        return tuple(types)


registry = EngineRegistry([OllamaEngine(), MatricxonEngine(), RunPodEngine()])
