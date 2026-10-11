"""Routes an incoming HTTP request across this machine's local pAIring instances (instance_service.py) when
"proxy mode" (Settings > System) is "local": a "fewest in-flight, health-aware" selector, a port of
ollama_pool.py to instance indices. instance_proxy.py forwards using pick_instance.

Primary-only: a sibling may be a *target* but never chooses. The candidate set changes live with the
instance count, so candidates are recomputed per call.
"""

import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.services import instance_process
from app.services.settings_service import DEFAULT_PROXY_MODE

# Same cooldown as ollama_pool.py: long enough not to retry a still-broken instance at once, short enough
# that a recovered one isn't idle for minutes.
_COOLDOWN_SECONDS = 30.0


class _InstanceState:
    __slots__ = ("in_flight", "cooldown_until")

    def __init__(self) -> None:
        self.in_flight = 0
        self.cooldown_until = 0.0


# Grown lazily via setdefault, never shrunk while a track_request may hold an entry (it would corrupt
# in_flight). A stale entry for a scaled-down index is harmless.
_pool: dict[int, _InstanceState] = {}


def _state(index: int) -> _InstanceState:
    return _pool.setdefault(index, _InstanceState())


def _candidate_indices() -> set[int]:
    """Index 0 (the primary, answering now) is always a candidate; the rest are tracked siblings confirmed alive
    (instance_process.read_tracking/is_alive: cheap, no network, as instance_service's reconcile).
    """
    tracked = instance_process.read_tracking()
    alive = {index for index, info in tracked.items() if instance_process.is_alive(info["pid"])}
    return {0} | alive


def pick_instance(exclude: frozenset[int] = frozenset()) -> int:
    """The healthy candidate with the fewest in-flight requests (like ollama_pool.pick_host; if all are excluded
    or cooling down, the least-recently-failed). Difference: ties break *randomly*. App traffic is mostly fast
    sequential requests with 0 in flight everywhere, and plain min (ollama_pool) always picked index 0
    (100% of 20 live requests hit the primary), so nothing balanced under light load. Ollama's long, varied
    requests don't have this problem, so ollama_pool is unchanged.
    """
    now = time.monotonic()
    candidates = _candidate_indices()
    healthy = [i for i in candidates if i not in exclude and _state(i).cooldown_until <= now]
    if healthy:
        fewest = min(_state(i).in_flight for i in healthy)
        return random.choice([i for i in healthy if _state(i).in_flight == fewest])
    fallback = [i for i in candidates if i not in exclude] or list(candidates)
    soonest = min(_state(i).cooldown_until for i in fallback)
    return random.choice([i for i in fallback if _state(i).cooldown_until == soonest])


def mark_failure(index: int) -> None:
    _state(index).cooldown_until = time.monotonic() + _COOLDOWN_SECONDS


def mark_recovered(index: int) -> None:
    _state(index).cooldown_until = 0.0


@asynccontextmanager
async def track_request(index: int) -> AsyncGenerator[None, None]:
    """Brackets one request's lifetime against `index`'s in-flight count, held for the caller's whole block (a
    proxied chat SSE stream's full duration). Same contract as ollama_pool.track_request.
    """
    state = _state(index)
    state.in_flight += 1
    try:
        yield
    finally:
        state.in_flight -= 1


# Proxy-mode write-through cache
# Populated at primary startup (app/main.py on_startup) and updated on admin save (routers/proxy_admin.py),
# avoiding a DB read per request. Siblings never use it; only the primary routes.
_proxy_mode = DEFAULT_PROXY_MODE


def get_cached_proxy_mode() -> str:
    return _proxy_mode


def set_cached_proxy_mode(mode: str) -> None:
    global _proxy_mode
    _proxy_mode = mode
