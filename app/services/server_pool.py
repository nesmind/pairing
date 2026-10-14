"""Generic "fewest in-flight requests, health-aware" host pool: the selection/cooldown/failover algorithm of
ollama_pool.py, extracted when comfyui_pool.py needed the same for another backend. No engine or ComfyUI
specifics here; the thin wrappers keep their public names (pick_host/mark_failure/...), so call sites are
unchanged.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TypeVar

import httpx

T = TypeVar("T")


async def ping_host(host: str, timeout: httpx.Timeout, path: str = "/") -> bool:
    """One-off reachability probe: a plain `GET`, success = 200. Used by HostPool.check_hosts (all configured hosts)
    and admin endpoints checking a candidate URL typed into Settings > External servers before saving
    (ollama_pool.check_host / comfyui_pool.check_host).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{host}{path}")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


@dataclass(frozen=True)
class RouteInfo:
    """What the hosts report about themselves (cluster_routing.NodeLoads), for one request: `warm` have its model
    loaded, `active` counts each host's replies from every client, `capacity` is its compute relative to the
    biggest host (1.0), and `no_room` lack the free memory to load the model."""

    warm: frozenset[str] = frozenset()
    active: dict[str, int] = field(default_factory=dict)
    capacity: dict[str, float] = field(default_factory=dict)
    no_room: frozenset[str] = frozenset()


class _HostState:
    __slots__ = ("in_flight", "cooldown_until")

    def __init__(self) -> None:
        self.in_flight = 0
        self.cooldown_until = 0.0


class HostPool:
    """One independent pool; each engine and ComfyUI own one, sharing no state. `cooldown_seconds`: how long a
    just-failed host sits out of rotation (mark_failure).
    """

    def __init__(self, cooldown_seconds: float = 30.0) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._hosts: list[str] = []
        self._states: dict[str, _HostState] = {}

    def set_hosts(self, hosts: list[str]) -> None:
        """Replaces the live host list; safe any time (after an admin saves hosts or switches mode), even while requests
        run against a host being removed. _HostState entries are only added, never pruned, so a request holding a
        removed host's state gets no KeyError (as instance_pool's dict never shrinks).
        """
        self._hosts = list(hosts)
        for host in self._hosts:
            self._states.setdefault(host, _HostState())

    def get_hosts(self) -> list[str]:
        return list(self._hosts)

    def load(self, host: str, info: RouteInfo | None = None) -> float:
        """How busy `host` is, in requests on the biggest host: this pool's own in-flight count or the host's own
        count (every client), whichever is higher, divided by its relative capacity."""
        busy = self._states[host].in_flight
        if info is not None:
            busy = max(busy, info.active.get(host, 0))
            return busy / max(info.capacity.get(host, 1.0), 0.05)
        return busy

    def healthy_hosts(self) -> list[str]:
        now = time.monotonic()
        return [h for h in self._hosts if self._states[h].cooldown_until <= now]

    def pick_host(
        self,
        exclude: frozenset[str] = frozenset(),
        prefer: str | None = None,
        slack: int = 2,
        info: RouteInfo | None = None,
        warm_slack: int = 1,
    ) -> str:
        """The healthy host with the fewest in-flight requests; "healthy" = outside its cooldown and not in `exclude`
        (hosts already tried this request; stream_with_failover/call_with_failover). If all are excluded or cooling
        down, falls back to the least-recently-failed rather than raising: a stale host beats refusing. IndexError if
        no host was ever configured (set_hosts never called, or empty): a real misconfiguration.
        """
        if not self._hosts:
            raise IndexError("HostPool has no hosts configured.")
        now = time.monotonic()
        healthy = [h for h in self._hosts if h not in exclude and self._states[h].cooldown_until <= now]
        if healthy:
            info = info or RouteInfo()
            # A host without room for the model only if nothing else is left.
            healthy = [h for h in healthy if h not in info.no_room or h in info.warm] or healthy

            def busy(h: str) -> float:
                return self.load(h, info)

            best = min(healthy, key=busy)
            # Affinity: the host already holding this request's cached state wins unless clearly busier than the best
            # (more than `slack` extra requests in flight); then load wins.
            if prefer in healthy and busy(prefer) <= busy(best) + slack:
                return prefer
            # Then a host that already has the model loaded, but only up to `warm_slack` extra requests: past that the
            # least busy host gets it, so load keeps spreading and cold hosts join in.
            warm_healthy = [h for h in healthy if h in info.warm]
            if warm_healthy:
                best_warm = min(warm_healthy, key=busy)
                if busy(best_warm) <= busy(best) + warm_slack:
                    return best_warm
            return best
        fallback = [h for h in self._hosts if h not in exclude] or list(self._hosts)
        return min(fallback, key=lambda h: self._states[h].cooldown_until)

    def mark_failure(self, host: str) -> None:
        self._states[host].cooldown_until = time.monotonic() + self._cooldown_seconds

    def mark_recovered(self, host: str) -> None:
        """Clears any cooldown on `host` when a request to it succeeds, instead of waiting out the window."""
        self._states[host].cooldown_until = 0.0

    @asynccontextmanager
    async def track_request(self, host: str):
        """Brackets one request's lifetime against `host`'s in-flight count, held for the caller's whole block (a
        streamed reply's *entire* duration), so "fewest in-flight" means fewest still being generated, not just started.
        """
        self._states[host].in_flight += 1
        try:
            yield
        finally:
            self._states[host].in_flight -= 1

    async def stream_with_failover(
        self,
        make_stream: Callable[[str], AsyncGenerator[str, None]],
        prefer: str | None = None,
        info: RouteInfo | None = None,
    ) -> AsyncGenerator[str, None]:
        """Runs `make_stream(host)` on the chosen host, retrying on another if `httpx.HTTPError` occurs *before* any
        chunk was yielded (safe: nothing downstream has seen anything). After one chunk has gone out a failure
        propagates: switching hosts mid-reply would duplicate or truncate it. Gives up once every host was tried once.
        """
        tried: set[str] = set()
        while True:
            host = self.pick_host(exclude=frozenset(tried), prefer=prefer, info=info)
            tried.add(host)
            first_chunk_sent = False
            async with self.track_request(host):
                try:
                    async for chunk in make_stream(host):
                        first_chunk_sent = True
                        yield chunk
                except httpx.HTTPError:
                    self.mark_failure(host)
                    if first_chunk_sent or tried >= set(self._hosts):
                        raise
                    continue
                else:
                    self.mark_recovered(host)
                    return

    async def call_with_failover(self, make_call: Callable[[str], Awaitable[T]]) -> T:
        """Same retry/cooldown contract as stream_with_failover for a single request/response call (comfyui_client): no
        partial output exists, so retrying on another host is always safe.
        """
        tried: set[str] = set()
        while True:
            host = self.pick_host(exclude=frozenset(tried))
            tried.add(host)
            async with self.track_request(host):
                try:
                    result = await make_call(host)
                except httpx.HTTPError:
                    self.mark_failure(host)
                    if tried >= set(self._hosts):
                        raise
                    continue
                else:
                    self.mark_recovered(host)
                    return result

    async def check_hosts(self, timeout: httpx.Timeout, path: str = "/") -> dict[str, bool]:
        """Pings every configured host and reports which are reachable: a fresh probe each call that deliberately
        doesn't read or update pick_host's cooldown state. A health poller (routers/health.py via
        ollama_pool.check_hosts) needs this instance's current reality, not possibly stale routing history.
        """
        hosts = self.get_hosts()
        results = await asyncio.gather(*(ping_host(host, timeout, path) for host in hosts))
        return dict(zip(hosts, results, strict=True))
