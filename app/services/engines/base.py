"""
The formal interface every inference engine (Ollama, Matricxon, and eventually a
serverless GPU provider — see buffer.md's "Future features > Serverless GPU
providers") implements, so app.services.inference_client (the only caller most of
the app ever goes through) and the few call sites that legitimately need to treat
"whichever engine is active" generically (app/routers/health.py,
app.services.engine_support_service) never hardcode a two-way "ollama" vs
"matricxon" choice again — adding a third engine is one new adapter class registered
in app.services.engines.registry, not a new branch here or anywhere that imports
this module.

An ABC, not a Protocol: some operations (embeddings, model pull/delete, stopping a
loaded model) don't apply to every engine — a serverless provider may have no
concept of "stop this model," and its billing model might not support raw embedding
calls at all. An ABC lets those default to a clear EngineCapabilityError (or a
harmless no-op for stop_model, since every existing caller already treats stopping a
model as best-effort cleanup) instead of forcing every adapter to implement
everything. It also means a broken/incomplete adapter fails at import time
(app.services.engines.registry builds every registered engine eagerly), not on the
first real request.

Each concrete engine (see ollama_engine.py/matricxon_engine.py) is a thin adapter
that forwards to its existing app.services.ollama_client/matricxon_client module —
by module reference, looked up again on every call, never bound to a local name at
construction time, since the existing test suite (tests/test_inference_client.py and
friends) monkeypatches those modules' own functions directly and must keep working
unchanged.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import ClassVar

from app.schemas import EngineName
from app.services.engine_support_checker import AlwaysSupportedChecker, EngineSupportChecker


class EngineCapabilityError(Exception):
    """Raised when a caller asks the active engine to do something its own
    EngineCapabilities says it can't (e.g. app.services.inference_client.embed while
    the active engine's capabilities.embeddings is False) — a clear, typed refusal
    instead of a confusing AttributeError or a silent no-op."""


@dataclass(frozen=True)
class EngineCapabilities:
    """What an engine can actually do — read by app.services.inference_client (to
    decide whether to even attempt a call) and by Settings (to grey out/hide UI that
    assumes a capability the active engine doesn't have, e.g. the RAG embedding model
    picker when `embeddings` is False)."""

    # Whether embed() does anything but raise EngineCapabilityError.
    embeddings: bool
    # Whether pull_model_stream()/delete_model() do anything but raise
    # EngineCapabilityError — a serverless provider likely manages its own models
    # remotely, with nothing for this app to pull/delete.
    model_management: bool
    # Whether stop_model() does anything but a harmless no-op.
    stop_model: bool
    # Whether this engine has a local process this app manages (see
    # app.services.ollama_process.py/matricxon_process.py and their own admin
    # routers) — False for anything remote-only/serverless, which has no
    # local/remote mode, no install flow, and no start/stop.
    local_process: bool
    # Whether this engine routes calls through app.services.server_pool.HostPool
    # (multiple hosts, failover, cooldown) — False for a single fixed remote
    # endpoint or a provider with its own load balancing.
    host_pool: bool
    # Whether this engine has a real (non-trivial) EngineSupportChecker — i.e.
    # whether "can this specific model actually run here" is a meaningful, checkable
    # question, as opposed to "assume yes" (see AlwaysSupportedChecker).
    support_checking: bool
    # Whether get_format_support() returns real data (supported architectures/
    # quantizations) rather than always None.
    format_introspection: bool


class InferenceEngine(ABC):
    """One instance per engine, held by app.services.engines.registry.registry —
    never constructed per-request. Every ClassVar below is a fixed fact about the
    engine, not per-instance state (an engine adapter is stateless; the pool/process
    modules it forwards to hold whatever live state actually exists)."""

    name: ClassVar[EngineName]
    display_name: ClassVar[str]
    capabilities: ClassVar[EngineCapabilities]
    # Every exception type this engine's own client module can raise for a failed
    # call — app.services.inference_client wraps these (unioned across every
    # registered engine, see registry.EngineRegistry.all_error_types) into its own
    # engine-agnostic InferenceError. Deliberately never a shared base class across
    # engines (see app.services.ollama_client.OllamaError's own docstring on why) —
    # a caller reaching for one engine's error type can never accidentally also
    # catch another's.
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
        """Reachability of every one of this engine's currently-effective hosts (see
        app.services.server_pool.HostPool.check_hosts) — powers GET /health."""

    async def embed(self, text: str, model: str) -> list[float]:
        raise EngineCapabilityError(f"{self.display_name} does not support embeddings.")

    async def stop_model(  # noqa: B027 - deliberate no-op default, not abstract
        self, model: str, request_id: str | None = None
    ) -> None:
        """Best-effort cleanup, not a real operation for every engine — defaults to
        a no-op rather than raising, matching how every existing caller already
        treats this (see app.services.ollama_client.stop_model's own docstring:
        "cleanup fan-out," never load-bearing)."""

    def pull_model_stream(self, tag: str) -> AsyncGenerator[dict, None]:
        raise EngineCapabilityError(f"{self.display_name} does not support pulling models through this app.")

    async def delete_model(self, tag: str) -> None:
        raise EngineCapabilityError(f"{self.display_name} does not support deleting models through this app.")

    async def get_format_support(self) -> dict | None:
        """Real supported-architectures/quantizations data (see
        app.services.matricxon_client.get_capabilities) — None when this engine has
        no such introspection (see EngineCapabilities.format_introspection)."""
        return None

    async def is_ready(self) -> bool:
        """Whether this engine can actually be activated right now — distinct from
        EngineCapabilities (what an engine can do at all): a Connector-backed engine
        (see app.services.connectors) isn't ready until an admin has saved valid
        config and enabled it (see app.services.engines.runpod_engine.RunPodEngine's
        own override). Default True: Ollama/Matricxon need no admin-supplied secret
        to be selected — their own local/remote server config is independent of
        whether they're the *active* engine."""
        return True

    async def test_connection(self, config: dict[str, str]) -> tuple[bool, str]:
        """Validates a candidate Connector config (see app.services.connectors) by actually attempting to reach
        the real service — called from the Connectors page's "Test connection" action
        (app/routers/connectors.py), before that config is allowed to make this engine ready (see is_ready).
        Default: no real connectivity test exists for this engine (Ollama/Matricxon aren't Connector-backed, so
        nothing calls this for them in practice)."""
        return True, "No connectivity test defined for this engine."
