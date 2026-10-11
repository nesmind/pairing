"""Cluster routing: a chat goes to the node holding its cached conversation; if moved, the holder spills it to shared
disk first."""

import httpx
import pytest

from app.services import matricxon_client, matricxon_pool
from app.services.server_pool import HostPool

A, B = "http://a:8420", "http://b:8420"
REAL_CLIENT = httpx.AsyncClient  # captured once: monkeypatching replaces httpx.AsyncClient


def _serve(monkeypatch, module, handler) -> None:
    """Routes `module`'s httpx clients to `handler`."""
    transport = handler if isinstance(handler, httpx.MockTransport) else httpx.MockTransport(handler)
    monkeypatch.setattr(module.httpx, "AsyncClient", lambda **kw: REAL_CLIENT(transport=transport, **kw))


def _pool(in_flight: dict[str, int] | None = None) -> HostPool:
    pool = HostPool(cooldown_seconds=30.0)
    pool.set_hosts([A, B])
    for host, count in (in_flight or {}).items():
        pool._states[host].in_flight = count
    return pool


def test_pick_host_prefers_the_holder_when_it_is_not_much_busier():
    assert _pool().pick_host(prefer=B) == B
    assert _pool({B: 2}).pick_host(prefer=B) == B
    assert _pool({B: 3}).pick_host(prefer=B) == A


def test_pick_host_ignores_a_holder_that_is_excluded_cooling_down_or_unknown():
    pool = _pool()
    assert pool.pick_host(exclude=frozenset({B}), prefer=B) == A
    pool.mark_failure(B)
    assert pool.pick_host(prefer=B) == A
    assert _pool().pick_host(prefer="http://other:1") in (A, B)
    assert _pool({A: 1}).pick_host() == B


def _answers(holders: set[str], online: set[str] | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        if online is not None and host not in online:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"ram": host in holders})

    return httpx.MockTransport(handler)


@pytest.fixture
def two_hosts(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "_pool", _pool())


async def _find(monkeypatch, transport: httpx.MockTransport, tag: str | None = "chat-1") -> str | None:
    _serve(monkeypatch, matricxon_pool, transport)
    return await matricxon_pool.find_holder(tag)


@pytest.mark.asyncio
async def test_find_holder_returns_the_host_with_the_chat_in_ram(monkeypatch, two_hosts):
    assert await _find(monkeypatch, _answers({B})) == B
    assert await _find(monkeypatch, _answers(set())) is None


@pytest.mark.asyncio
async def test_find_holder_survives_a_dead_or_old_node_and_skips_single_hosts(monkeypatch, two_hosts):
    assert await _find(monkeypatch, _answers({A, B}, online={B})) == B
    assert await _find(monkeypatch, _answers({B}), tag=None) is None
    monkeypatch.setattr(matricxon_pool, "_pool", HostPool(cooldown_seconds=30.0))
    matricxon_pool._pool.set_hosts([A])
    assert await _find(monkeypatch, _answers({A})) is None


@pytest.mark.asyncio
async def test_spill_chat_posts_the_tag_and_reports_how_many_moved(monkeypatch):
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path, request.content))
        return httpx.Response(200, json={"spilled": 1})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        matricxon_pool.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    assert await matricxon_pool.spill_chat(A, "chat-1") == 1
    assert seen == [("POST", "/api/cache/spill", b'{"tag":"chat-1"}')]


@pytest.mark.asyncio
async def test_spill_chat_never_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    real = httpx.AsyncClient
    monkeypatch.setattr(
        matricxon_pool.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    assert await matricxon_pool.spill_chat(A, "chat-1") == 0


PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,
    "num_predict": 64,
    "conversation_id": "chat-1",
}


@pytest.mark.asyncio
async def test_chat_stream_prefers_the_holder_and_spills_it_only_when_the_chat_moves(monkeypatch):
    spilled: list[tuple[str, str]] = []

    async def holder(_tag):
        return B

    async def spill(host, tag):
        spilled.append((host, tag))
        return 1

    chosen: dict = {}

    async def fake_stream(make_stream, prefer=None):
        chosen["prefer"] = prefer
        async for chunk in make_stream(chosen["host"]):
            yield chunk

    monkeypatch.setattr(matricxon_pool, "find_holder", holder)
    monkeypatch.setattr(matricxon_pool, "spill_chat", spill)
    monkeypatch.setattr(matricxon_pool, "stream_with_failover", fake_stream)
    _serve(
        monkeypatch,
        matricxon_client,
        lambda request: httpx.Response(200, content=b'{"message": {"content": "hi"}, "done": false}\n{"done": true}\n'),
    )

    async def run(host: str) -> list[str]:
        chosen["host"] = host
        messages = [{"role": "user", "content": "hi"}]
        return [c async for c in matricxon_client.chat_stream("m", messages, dict(PARAMS))]

    assert await run(B) == ["hi"] and chosen["prefer"] == B and spilled == []
    assert await run(A) == ["hi"] and spilled == [(B, "chat-1")]  # moved: holder spilled it first
