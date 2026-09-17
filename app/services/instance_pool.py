"""
Routes an incoming HTTP request across this machine's local pAIring
instances (see app/services/instance_service.py for how those get
spawned) when "proxy mode" (Settings > System) is set to "local" — the
"fewest in-flight requests, health-aware" pool selector, a direct port
of app/services/ollama_pool.py's design to instance indices instead of
Ollama host URLs. See app/services/instance_proxy.py for the ASGI
middleware that actually forwards a request using pick_instance's
choice.

Primary-process-only: a sibling never imports this for routing
decisions (it may still end up as a *target* another instance forwards
to, but it never chooses). Unlike Ollama's own remote host list, the
candidate set isn't fixed for the process's lifetime — instance count
can change live from
Settings — so candidates are recomputed on every call rather than built
once at import time.
"""

import random
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from app.services import instance_process
from app.services.settings_service import DEFAULT_PROXY_MODE

# Same cooldown window as ollama_pool.py, same reasoning: long enough
# that one failed request doesn't get immediately retried against the
# same still-broken instance, short enough that a recovered instance
# doesn't sit unused for minutes.
_COOLDOWN_SECONDS = 30.0


class _InstanceState:
    __slots__ = ("in_flight", "cooldown_until")

    def __init__(self) -> None:
        self.in_flight = 0
        self.cooldown_until = 0.0


# Lazily grown via setdefault — never shrunk while a track_request could
# still hold a reference to an entry, which would corrupt its in_flight
# count. A stale entry for an index that's since scaled down just sits
# unused; harmless.
_pool: dict[int, _InstanceState] = {}


def _state(index: int) -> _InstanceState:
    return _pool.setdefault(index, _InstanceState())


def _candidate_indices() -> set[int]:
    """Index 0 (the primary itself, answering this very call) is always
    a candidate; the rest come from whichever siblings are currently
    tracked and confirmed alive — see instance_process.read_tracking/
    is_alive, the same cheap synchronous (no network) checks
    instance_service's own reconcile loop uses."""
    tracked = instance_process.read_tracking()
    alive = {index for index, info in tracked.items() if instance_process.is_alive(info["pid"])}
    return {0} | alive


def pick_instance(exclude: frozenset[int] = frozenset()) -> int:
    """The healthy candidate with the fewest in-flight requests right
    now — mirrors ollama_pool.pick_host's shape (falls back to the
    least-recently-failed candidate rather than raising if every
    candidate is excluded or cooling down at once), with one deliberate
    difference: ties are broken *randomly*, not by always favoring
    whichever candidate happens to sort first. app-instance traffic is
    dominated by fast, mostly-sequential requests (page loads, small API
    calls) where every candidate's in-flight count is 0 far more often
    than not — with Python's plain min() (what ollama_pool uses), a tie
    always resolves to the same candidate (index 0, confirmed live:
    100% of a real 20-request sequential/concurrent test all landed on
    the primary), which never actually balances anything under normal
    light load. Ollama's own traffic doesn't have this problem in
    practice (requests are long and vary enough that real in-flight
    differences dominate), so ollama_pool is left as-is."""
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
    """Brackets one request's lifetime against `index`'s in-flight count
    — held for the caller's entire block, which for a proxied chat SSE
    stream is its whole duration, not just until headers are sent. Same
    contract as ollama_pool.track_request."""
    state = _state(index)
    state.in_flight += 1
    try:
        yield
    finally:
        state.in_flight -= 1


# --- Proxy-mode write-through cache -------------------------------------
# Populated once at primary startup (app/main.py's on_startup) and
# updated immediately whenever the admin saves a change (see
# app/routers/proxy_admin.py) — avoids a DB round trip on every single
# incoming request. A sibling process never reads or writes this; only
# the primary ever makes routing decisions.
_proxy_mode = DEFAULT_PROXY_MODE


def get_cached_proxy_mode() -> str:
    return _proxy_mode


def set_cached_proxy_mode(mode: str) -> None:
    global _proxy_mode
    _proxy_mode = mode
