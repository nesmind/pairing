"""The interface every inference engine (Ollama, Matricxon, later a serverless GPU provider; buffer.md
"Future features > Serverless GPU providers") implements, so inference_client and generic call sites
(routers/health.py, engine_support_service) never hardcode a two-way choice. A third engine is one adapter
registered in engines.registry.

An ABC, not a Protocol: some operations (embeddings, pull/delete, stop_model) don't apply to every engine.
An ABC lets them default to EngineCapabilityError (or a no-op for stop_model, always best-effort cleanup)
instead of forcing every adapter to implement everything, and a broken adapter fails at import
(registry builds engines eagerly), not on the first request.

Each adapter forwards to its ollama_client/matricxon_client module by module reference, looked up on each
call, never bound at construction: tests (tests/test_inference_client.py etc.) monkeypatch those modules' functions.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import ClassVar

from app.schemas import EngineName
from app.services.engine_support_checker import AlwaysSupportedChecker, EngineSupportChecker


class EngineCapabilityError(Exception):
    """Raised when a caller asks the active engine for something its EngineCapabilities says it can't (e.g.
    inference_client.embed with capabilities.embeddings False): a typed refusal instead of an AttributeError or silent
    no-op.
    """


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can do; read by inference_client (whether to attempt a call) and Settings (to hide UI, e.g.
    the RAG embedding picker when `embeddings` is False).
    """

    # Whether embed does anything but raise EngineCapabilityError.
    embeddings: bool
    # Whether pull_model_stream/delete_model do anything but raise EngineCapabilityError (a serverless provider
    # likely manages models remotely).
    model_management: bool
    # Whether stop_model does more than a no-op.
    stop_model: bool
    # Whether this app manages a local process for the engine (ollama_process.py/matricxon_process.py and admin
    # routers); False for remote-only/serverless engines (no local/remote mode, install or start/stop).
    local_process: bool
    # Whether calls go through server_pool.HostPool (hosts, failover, cooldown); False for a single fixed
    # endpoint or a provider with its own load balancing.
    host_pool: bool
    # Whether there is a real EngineSupportChecker, i.e. "can this model run here" is checkable rather than "assume yes"
    # (AlwaysSupportedChecker).
    support_checking: bool
    # Whether get_format_support returns real data (architectures/quantizations) rather than None.
    format_introspection: bool


class InferenceEngine(ABC):
    """One instance per engine, held by engines.registry.registry, never per request. Every ClassVar is a fixed
    fact; adapters are stateless (the pool/process modules hold live state).
    """

    name: ClassVar[EngineName]
    display_name: ClassVar[str]
    capabilities: ClassVar[EngineCapabilities]
    # Every exception type the engine's client can raise; inference_client wraps the union across engines
    # (registry.EngineRegistry.all_error_types) into its engine-agnostic InferenceError. Never a shared base
    # class across engines (see ollama_client.OllamaError), so catching one engine's error can't catch another's.
    error_types: ClassVar[tuple[type[Exception], ...]]
    support_checker: ClassVar[type[EngineSupportChecker]] = AlwaysSupportedChecker

    @abstractmethod
    async def list_models(self) -> list[dict]: ...

    @abstractmethod
    def chat_stream(self, model: str, messages: list[dict], params: dict) -> AsyncGenerator[str, None]: ...

    @abstractmethod
    async def chat_once(self, model: str, messages: list[dict], params: dict | None = None) -> str: ...

    @abstractmethod
    async def check_health(self) -> dict[str, bool]:
        """Reachability of each currently-effective host (server_pool.HostPool.check_hosts); powers GET /health."""

    async def embed(self, text: str, model: str) -> list[float]:
        raise EngineCapabilityError(f"{self.display_name} does not support embeddings.")

    async def stop_model(  # noqa: B027 - deliberate no-op default, not abstract
        self, model: str, request_id: str | None = None
    ) -> None:
        """Best-effort cleanup, a no-op by default as every caller already treats it (see ollama_client.stop_model:
        "cleanup fan-out", never load-bearing)."""

    def pull_model_stream(self, tag: str) -> AsyncGenerator[dict, None]:
        raise EngineCapabilityError(f"{self.display_name} does not support pulling models through this app.")

    async def delete_model(self, tag: str) -> None:
        raise EngineCapabilityError(f"{self.display_name} does not support deleting models through this app.")

    async def get_format_support(self) -> dict | None:
        """Supported architectures/quantizations (matricxon_client.get_capabilities), or None without introspection
        (EngineCapabilities.format_introspection).
        """
        return None

    async def is_ready(self) -> bool:
        """Whether the engine can be activated now. Distinct from EngineCapabilities: a Connector-backed engine
        (services.connectors) isn't ready until an admin saved valid config and enabled it (RunPodEngine overrides).
        Default True: Ollama/Matricxon need no secret to be selected.
        """
        return True

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        """Validates a candidate Connector config by trying the real service; called by the Connectors page's "Test
        connection" (routers/connectors.py) before the engine may become ready (see is_ready). Default: no real
        test (Ollama/Matricxon aren't Connector-backed).
        """
        return True, "No connectivity test defined for this engine."
