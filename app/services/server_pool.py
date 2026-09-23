"""
Generic "fewest in-flight requests, health-aware" host pool — the exact
selection/cooldown/failover algorithm app/services/ollama_pool.py always
used, extracted into a reusable class once app/services/comfyui_pool.py
needed the identical logic for a second, unrelated backend. Neither
ML engine nor ComfyUI specifics live here; both thin wrapper modules keep
their own public function names (pick_host/mark_failure/... ) so no
existing call site anywhere else in the app needs to change its own
call shape — only how each pool's host list gets populated does.
"""

import asyncio
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import TypeVar

import httpx

T = TypeVar("T")


async def ping_host(host: str, timeout: httpx.Timeout, path: str = "/") -> bool:
    """One-off reachability probe against `host` — a plain `GET`,
    success meaning a 200 response, nothing more. Used both by
    HostPool.check_hosts (every currently configured host) and directly
    by admin endpoints checking a single candidate URL an admin just
    typed into Settings > External servers, before it's ever saved to
    the pool (see app.services.ollama_pool.check_host/
    comfyui_pool.check_host)."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{host}{path}")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


class _HostState:
    __slots__ = ("in_flight", "cooldown_until")

    def __init__(self) -> None:
        self.in_flight = 0
        self.cooldown_until = 0.0


class HostPool:
    """One independent pool — each ML engine and ComfyUI own their own
    instance, never sharing state. `cooldown_seconds` is how long a host
    that just failed sits out of rotation before being considered again
    (see mark_failure)."""

    def __init__(self, cooldown_seconds: float = 30.0) -> None:
        self._cooldown_seconds = cooldown_seconds
        self._hosts: list[str] = []
        self._states: dict[str, _HostState] = {}

    def set_hosts(self, hosts: list[str]) -> None:
        """Replaces the live host list — safe to call at any time (e.g.
        right after an admin saves a new remote host list, or switches
        mode), including while requests are in flight against a host
        being removed: existing _HostState entries are never pruned, only
        added to, so a request already holding a reference to a since-
        removed host's state doesn't hit a KeyError (same reasoning
        app.services.instance_pool's own pool never shrinks its
        tracking dict while a request could still reference an entry)."""
        self._hosts = list(hosts)
        for host in self._hosts:
            self._states.setdefault(host, _HostState())

    def get_hosts(self) -> list[str]:
        return list(self._hosts)

    def pick_host(self, exclude: frozenset[str] = frozenset()) -> str:
        """The healthy host with the fewest in-flight requests right
        now — "healthy" meaning outside its post-failure cooldown window
        and not in `exclude` (hosts a caller already tried this request
        — see stream_with_failover/call_with_failover). If every host is
        excluded or cooling down at once, falls back to the least-
        recently-failed one anyway rather than raising — a stale host is
        still worth one more try over refusing to serve the request at
        all. Raises IndexError if no host has ever been configured
        (set_hosts([...]) never called, or called with an empty list) —
        a real misconfiguration, not something to paper over."""
        if not self._hosts:
            raise IndexError("HostPool has no hosts configured.")
        now = time.monotonic()
        healthy = [h for h in self._hosts if h not in exclude and self._states[h].cooldown_until <= now]
        if healthy:
            return min(healthy, key=lambda h: self._states[h].in_flight)
        fallback = [h for h in self._hosts if h not in exclude] or list(self._hosts)
        return min(fallback, key=lambda h: self._states[h].cooldown_until)

    def mark_failure(self, host: str) -> None:
        self._states[host].cooldown_until = time.monotonic() + self._cooldown_seconds

    def mark_recovered(self, host: str) -> None:
        """Clears any cooldown on `host` the moment a request to it
        actually succeeds, rather than waiting out the rest of an
        already-irrelevant cooldown window."""
        self._states[host].cooldown_until = 0.0

    @asynccontextmanager
    async def track_request(self, host: str):
        """Brackets one request's lifetime against `host`'s in-flight
        count — held for as long as the caller stays in this block,
        which for a streamed reply is its *entire* duration, not just
        the initial connect. That's what makes "fewest in-flight" mean
        "fewest requests still being generated," not just "fewest just
        started."""
        self._states[host].in_flight += 1
        try:
            yield
        finally:
            self._states[host].in_flight -= 1

    async def stream_with_failover(
        self, make_stream: Callable[[str], AsyncGenerator[str, None]]
    ) -> AsyncGenerator[str, None]:
        """Runs `make_stream(host)` against the pool's chosen host,
        retrying on a different host if `httpx.HTTPError` happens
        *before* any chunk was yielded — safe exactly then, since
        nothing downstream (an SSE response a browser is reading) has
        seen anything yet. Once even one chunk has gone out, a failure
        propagates instead of retrying: switching hosts mid-reply would
        duplicate or truncate it, not fix it. Gives up once every
        configured host has been tried once."""
        tried: set[str] = set()
        while True:
            host = self.pick_host(exclude=frozenset(tried))
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
        """Same retry/cooldown contract as stream_with_failover, for a
        plain single-shot request/response call instead of a stream
        (see app.services.comfyui_client, which has no long-lived
        generator to worry about truncating mid-flight — a failed call
        here has produced no partial output at all, so it's always safe
        to retry on a different host)."""
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
        """Pings every configured host directly and reports which are
        currently reachable — a fresh, independent probe every call,
        deliberately not reading or updating pick_host's own cooldown
        state (see app/routers/health.py's use of this for the ML engine, via
        app.services.ollama_pool.check_hosts, for why: a health poller
        needs this instance's *current* reality, not routing history
        that might be stale)."""
        hosts = self.get_hosts()
        results = await asyncio.gather(*(ping_host(host, timeout, path) for host in hosts))
        return dict(zip(hosts, results, strict=True))
