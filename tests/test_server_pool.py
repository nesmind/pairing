"""Unit tests for app/services/server_pool.py's generic HostPool — the
"fewest in-flight, health-aware" selection/cooldown/failover algorithm
shared by app/services/ollama_pool.py and app/services/comfyui_pool.py.
Exercised directly against a fresh HostPool instance here so this logic
is only ever tested once, not duplicated per wrapper module."""

import httpx
import pytest

from app.services.server_pool import HostPool, ping_host


@pytest.fixture
def pool():
    p = HostPool(cooldown_seconds=30.0)
    p.set_hosts(["http://h1:1", "http://h2:1", "http://h3:1"])
    return p


def test_pick_host_with_a_single_host_always_returns_it():
    p = HostPool()
    p.set_hosts(["http://only:1"])
    assert p.pick_host() == "http://only:1"


def test_pick_host_raises_if_no_hosts_configured():
    p = HostPool()
    with pytest.raises(IndexError):
        p.pick_host()


def test_pick_host_prefers_fewest_in_flight(pool):
    h1, h2, h3 = pool.get_hosts()
    pool._states[h1].in_flight = 3
    pool._states[h2].in_flight = 0
    pool._states[h3].in_flight = 1
    assert pool.pick_host() == h2


def test_pick_host_excludes_given_hosts(pool):
    h1, h2, h3 = pool.get_hosts()
    pool._states[h1].in_flight = 0  # would otherwise win
    assert pool.pick_host(exclude=frozenset({h1})) in (h2, h3)


def test_mark_failure_puts_a_host_in_cooldown_and_pick_host_avoids_it(pool):
    h1 = pool.get_hosts()[0]
    pool.mark_failure(h1)
    for _ in range(10):
        assert pool.pick_host() != h1


def test_mark_recovered_clears_cooldown_immediately(pool):
    h1 = pool.get_hosts()[0]
    pool.mark_failure(h1)
    assert pool.pick_host() != h1
    pool.mark_recovered(h1)
    assert pool.pick_host() == h1  # first in list, tied in-flight, no longer cooling


def test_pick_host_falls_back_to_something_when_every_host_is_cooling(pool):
    hosts = pool.get_hosts()
    for h in hosts:
        pool.mark_failure(h)
    assert pool.pick_host() in hosts


def test_set_hosts_does_not_reset_state_of_hosts_still_present(pool):
    h1 = pool.get_hosts()[0]
    pool.mark_failure(h1)
    pool.set_hosts(pool.get_hosts())  # re-saving the same list
    assert pool.pick_host() != h1  # cooldown survived


@pytest.mark.asyncio
async def test_track_request_increments_and_decrements_in_flight(pool):
    host = pool.get_hosts()[0]
    assert pool._states[host].in_flight == 0
    async with pool.track_request(host):
        assert pool._states[host].in_flight == 1
    assert pool._states[host].in_flight == 0


@pytest.mark.asyncio
async def test_track_request_decrements_even_on_exception(pool):
    host = pool.get_hosts()[0]
    with pytest.raises(ValueError):
        async with pool.track_request(host):
            raise ValueError("boom")
    assert pool._states[host].in_flight == 0


@pytest.mark.asyncio
async def test_stream_with_failover_succeeds_on_a_healthy_host(pool):
    async def make_stream(_host):
        yield "a"
        yield "b"

    result = [c async for c in pool.stream_with_failover(make_stream)]
    assert result == ["a", "b"]


@pytest.mark.asyncio
async def test_stream_with_failover_retries_a_different_host_before_any_chunk(pool):
    hosts = pool.get_hosts()
    attempted: list[str] = []

    async def make_stream(host):
        attempted.append(host)
        if host == hosts[0]:
            raise httpx.ConnectError("first host down")
        yield "ok from a working host"

    result = [c async for c in pool.stream_with_failover(make_stream)]

    assert result == ["ok from a working host"]
    assert hosts[0] in attempted
    assert len(attempted) == 2
    assert len(set(attempted)) == 2
    assert pool._states[hosts[0]].cooldown_until > 0


@pytest.mark.asyncio
async def test_stream_with_failover_does_not_retry_once_a_chunk_was_already_sent(pool):
    attempted: list[str] = []

    async def make_stream(host):
        attempted.append(host)
        yield "first chunk reaches the caller"
        raise httpx.ConnectError("dies mid-stream")

    with pytest.raises(httpx.ConnectError):
        _ = [c async for c in pool.stream_with_failover(make_stream)]

    assert len(attempted) == 1


@pytest.mark.asyncio
async def test_stream_with_failover_tries_every_host_exactly_once_before_giving_up(pool):
    hosts = pool.get_hosts()
    attempted: list[str] = []

    async def always_fails(host):
        attempted.append(host)
        raise httpx.ConnectError("down")
        yield  # pragma: no cover - never reached; makes this an async generator

    with pytest.raises(httpx.ConnectError):
        _ = [c async for c in pool.stream_with_failover(always_fails)]

    assert sorted(attempted) == sorted(hosts)


@pytest.mark.asyncio
async def test_stream_with_failover_marks_the_successful_host_recovered(pool, monkeypatch):
    recovered: list[str] = []
    monkeypatch.setattr(pool, "mark_recovered", recovered.append)

    async def make_stream(_host):
        yield "ok"

    _ = [c async for c in pool.stream_with_failover(make_stream)]
    assert len(recovered) == 1


@pytest.mark.asyncio
async def test_call_with_failover_succeeds_on_a_healthy_host(pool):
    async def make_call(host):
        return f"result from {host}"

    result = await pool.call_with_failover(make_call)
    assert result.startswith("result from")


@pytest.mark.asyncio
async def test_call_with_failover_retries_a_different_host_on_failure(pool):
    hosts = pool.get_hosts()
    attempted: list[str] = []

    async def make_call(host):
        attempted.append(host)
        if host == hosts[0]:
            raise httpx.ConnectError("down")
        return "ok"

    result = await pool.call_with_failover(make_call)
    assert result == "ok"
    assert len(set(attempted)) == 2
    assert pool._states[hosts[0]].cooldown_until > 0


@pytest.mark.asyncio
async def test_call_with_failover_raises_once_every_host_fails(pool):
    hosts = pool.get_hosts()
    attempted: list[str] = []

    async def always_fails(host):
        attempted.append(host)
        raise httpx.ConnectError("down")

    with pytest.raises(httpx.ConnectError):
        await pool.call_with_failover(always_fails)

    assert sorted(attempted) == sorted(hosts)


@pytest.mark.asyncio
async def test_call_with_failover_marks_the_successful_host_recovered(pool, monkeypatch):
    recovered: list[str] = []
    monkeypatch.setattr(pool, "mark_recovered", recovered.append)

    async def make_call(_host):
        return "ok"

    await pool.call_with_failover(make_call)
    assert len(recovered) == 1


class _FakeHealthResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _FakeHealthClient:
    """Stands in for httpx.AsyncClient in check_hosts's own probe: only
    the hosts in `up` respond 200, everything else raises a connection
    error — no real network involved."""

    def __init__(self, up: set[str]):
        self._up = up

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url: str):
        host = url.removesuffix("/api/tags")
        if host in self._up:
            return _FakeHealthResponse(200)
        raise httpx.ConnectError("down")


@pytest.mark.asyncio
async def test_check_hosts_reports_per_host_reachability(pool, monkeypatch):
    hosts = pool.get_hosts()
    up = {hosts[0], hosts[2]}
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeHealthClient(up))

    result = await pool.check_hosts(httpx.Timeout(1.0), path="/api/tags")

    assert result == {hosts[0]: True, hosts[1]: False, hosts[2]: True}


@pytest.mark.asyncio
async def test_ping_host_reports_a_reachable_host_as_healthy(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeHealthClient({"http://up:1"}))
    assert await ping_host("http://up:1", httpx.Timeout(1.0), path="/api/tags") is True


@pytest.mark.asyncio
async def test_ping_host_reports_an_unreachable_host_as_unhealthy(monkeypatch):
    # An arbitrary candidate URL that was never added to any pool —
    # exactly the "check before you save" case check-host endpoints use.
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeHealthClient(set()))
    assert await ping_host("http://down:1", httpx.Timeout(1.0), path="/api/tags") is False
