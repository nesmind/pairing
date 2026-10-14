"""Model-aware cluster routing: HostPool prefers nodes with the model loaded only up to a load limit, counts what
each node reports (every client, its threads, its free memory), NodeLoads reads each node once per TTL, and WarmUp
loads the model on an idle node only when every node that has it is busy."""

import asyncio

import httpx
import pytest

from app.services.cluster_routing import NodeLoads, WarmUp
from app.services.server_pool import HostPool, RouteInfo

A, B, C = "http://a:8420", "http://b:8420", "http://c:8420"


def _pool(*loads: int) -> HostPool:
    pool = HostPool()
    pool.set_hosts([A, B, C][: len(loads)])
    for host, n in zip([A, B, C], loads, strict=False):
        pool._states[host].in_flight = n
    return pool


def _warm(*hosts: str, **kw) -> RouteInfo:
    return RouteInfo(warm=frozenset(hosts), **kw)


class TestPickHost:
    def test_a_warm_node_wins_up_to_one_extra_request(self) -> None:
        assert _pool(0, 1).pick_host(info=_warm(B)) == B

    def test_past_the_limit_the_least_busy_cold_node_wins(self) -> None:
        assert _pool(0, 2).pick_host(info=_warm(B)) == A

    def test_the_chat_holder_still_comes_first(self) -> None:
        assert _pool(1, 0, 0).pick_host(prefer=A, info=_warm(B)) == A

    def test_without_info_it_is_least_busy(self) -> None:
        assert _pool(2, 1, 3).pick_host() == B

    def test_counts_replies_from_other_clients(self) -> None:
        # This instance sent nothing to A, but A reports 3 replies (another pAIring instance): B wins.
        assert _pool(0, 1).pick_host(info=RouteInfo(active={A: 3})) == B

    def test_a_smaller_node_counts_each_request_more(self) -> None:
        # One request on a half-size node weighs like two on the big one.
        info = RouteInfo(capacity={A: 1.0, B: 0.5})
        assert _pool(1, 1).pick_host(info=info) == A

    def test_a_node_without_room_for_the_model_is_skipped(self) -> None:
        info = _warm(C, no_room=frozenset({A}))
        assert _pool(0, 1, 3).pick_host(info=info) == B
        assert _pool(0).pick_host(info=RouteInfo(no_room=frozenset({A}))) == A  # nothing else left


class _Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _client_factory(handler, calls: list[str]):
    def factory(**kwargs) -> httpx.AsyncClient:
        def counted(request: httpx.Request) -> httpx.Response:
            calls.append(f"{request.url.host}{request.url.path}")
            return handler(request)

        return httpx.AsyncClient(transport=httpx.MockTransport(counted), **kwargs)

    return factory


GB = 1 << 30


def _node(models: dict[str, int], active: int, threads: int, free: int) -> dict:
    return {
        "active": active,
        "models": [{"name": n, "size_bytes": s, "active": 0} for n, s in models.items()],
        "threads": threads,
        "free_memory_bytes": free,
        "node_kind": "compute",
    }


@pytest.mark.asyncio
async def test_node_loads_build_the_route_info() -> None:
    nodes = {
        "a": _node({"llama:3b": 2 * GB}, active=2, threads=8, free=6 * GB),
        "b": _node({}, active=0, threads=4, free=1 * GB),  # too little room for a 2 GB model
    }
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "c":  # an older node: no /api/node/load
            if request.url.path == "/api/ps":
                return httpx.Response(200, json={"models": [{"name": "llama:3b", "size": 2 * GB}]})
            return httpx.Response(404)
        return httpx.Response(200, json=nodes[request.url.host])

    info = await NodeLoads(_client_factory(handler, calls)).route_info([A, B, C], "llama:3b")

    assert info.warm == {A, C}
    assert info.active == {A: 2, B: 0, C: 0}
    assert info.capacity == {A: 1.0, B: 0.5}  # C reports no threads: counted as full size
    assert info.no_room == {B}


@pytest.mark.asyncio
async def test_node_loads_read_once_per_ttl_note_a_used_model_and_skip_one_host() -> None:
    calls: list[str] = []
    clock = _Clock()
    handler = lambda r: httpx.Response(200, json=_node({}, 0, 4, 8 * GB))  # noqa: E731
    loads = NodeLoads(_client_factory(handler, calls), ttl=2.0, clock=clock)
    assert (await loads.route_info([A, B], "m")).warm == frozenset()
    await loads.route_info([A, B], "m")
    assert len(calls) == 2  # cached
    loads.note(B, "m")
    assert (await loads.route_info([A, B], "m")).warm == {B}
    clock.now += 3
    await loads.route_info([A, B], "m")
    assert len(calls) == 4
    assert await loads.route_info([A], "m") == RouteInfo()


class TestWarmUp:
    def _warm_up(self, clock: _Clock | None = None) -> tuple[WarmUp, list[tuple[str, str]]]:
        loads: list[tuple[str, str]] = []

        async def load(host: str, model: str) -> None:
            loads.append((host, model))

        return WarmUp(load, cooldown=600.0, clock=clock or _Clock()), loads

    def test_only_when_every_node_with_the_model_is_busy(self) -> None:
        warm_up, _ = self._warm_up()
        assert warm_up.pick(_pool(0, 0), "m", _warm(A)) is None  # A has room
        assert warm_up.pick(_pool(1, 0), "m", _warm(A)) == B
        assert warm_up.pick(_pool(0, 0), "m", _warm(A, active={A: 1})) == B  # busy from another client
        assert warm_up.pick(_pool(1, 1), "m", _warm(A)) is None  # B is not idle
        assert warm_up.pick(_pool(1, 0), "m", _warm(A, no_room=frozenset({B}))) is None  # B lacks memory
        assert warm_up.pick(_pool(0, 0), "m", RouteInfo()) is None  # nobody has it: the chat loads it

    @pytest.mark.asyncio
    async def test_loads_in_the_background_and_waits_out_the_cooldown(self) -> None:
        clock = _Clock()
        warm_up, loads = self._warm_up(clock)
        pool = _pool(1, 0)
        assert warm_up.start_if_needed(pool, "m", _warm(A)) == B
        await asyncio.gather(*warm_up._tasks)
        assert loads == [(B, "m")] and pool.load(B) == 0
        assert warm_up.start_if_needed(pool, "m", _warm(A)) is None  # cooldown
        clock.now += 601
        assert warm_up.start_if_needed(pool, "m", _warm(A)) == B
        await asyncio.gather(*warm_up._tasks)
