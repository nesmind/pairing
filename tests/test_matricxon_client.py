"""Unit tests for app/services/matricxon_client.py — mirrors tests/test_ollama_client.py's wiring coverage
(confirms embed/chat_stream/stop_model actually route through app.services.matricxon_pool rather than a fixed
host), minus the OpenTelemetry span assertions: matricxon_client deliberately has no span instrumentation yet
(see that module's own docstring)."""

import httpx
import pytest

from app.services import matricxon_client, matricxon_pool


@pytest.fixture(autouse=True)
def _reset_capabilities_cache():
    """get_capabilities' own module-level cache (see its own docstring) must never leak between tests — a
    plain global would otherwise let one test's fake response satisfy a later test's own real call, silently
    hiding whatever that later test meant to check."""
    matricxon_client._capabilities_cache = None
    yield
    matricxon_client._capabilities_cache = None


class _FakeResponse:
    def __init__(self, json_body: dict):
        self._json_body = json_body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_body


class _FakeAsyncClient:
    def __init__(self, calls: list[str], json_body: dict):
        self._calls = calls
        self._json_body = json_body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url: str, json: dict):
        self._calls.append(url)
        return _FakeResponse(self._json_body)

    async def get(self, url: str):
        self._calls.append(url)
        return _FakeResponse(self._json_body)


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_list_models_uses_whatever_host_the_pool_picks(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"models": [{"name": "m"}]}))

    result = await matricxon_client.list_models()

    assert result == [{"name": "m"}]
    assert calls == ["http://pool-picked:8420/api/tags"]


@pytest.mark.asyncio
async def test_list_models_raises_matricxon_error_on_http_failure(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://x/api/tags"))

    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())

    with pytest.raises(matricxon_client.MatricxonError):
        await matricxon_client.list_models()


@pytest.mark.asyncio
async def test_get_capabilities_uses_whatever_host_the_pool_picks(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"supported_architectures": ["mistral3"]})
    )

    result = await matricxon_client.get_capabilities()

    assert result == {"supported_architectures": ["mistral3"]}
    assert calls == ["http://pool-picked:8420/api/health"]


@pytest.mark.asyncio
async def test_get_capabilities_raises_matricxon_error_on_http_failure(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://x/api/health"))

    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())

    with pytest.raises(matricxon_client.MatricxonError):
        await matricxon_client.get_capabilities()


@pytest.mark.asyncio
async def test_get_capabilities_reuses_the_cached_result_within_the_ttl(monkeypatch):
    """The real point of this cache (confirmed live, 2026-09-21): repeated catalog builds close together in
    time were each making their own fresh GET /api/health round trip — a second call within the TTL must reuse
    the first's result instead of hitting Matricxon again."""
    calls: list[str] = []
    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"supported_architectures": ["mistral3"]})
    )

    first = await matricxon_client.get_capabilities()
    second = await matricxon_client.get_capabilities()

    assert first == second == {"supported_architectures": ["mistral3"]}
    assert calls == ["http://pool-picked:8420/api/health"]  # only one real call, not two


@pytest.mark.asyncio
async def test_get_capabilities_refetches_once_the_cache_expires(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"supported_architectures": ["mistral3"]})
    )

    fake_now = [1000.0]
    monkeypatch.setattr(matricxon_client.time, "monotonic", lambda: fake_now[0])

    await matricxon_client.get_capabilities()
    fake_now[0] += matricxon_client._CAPABILITIES_CACHE_TTL_SECONDS + 1
    await matricxon_client.get_capabilities()

    assert calls == ["http://pool-picked:8420/api/health", "http://pool-picked:8420/api/health"]


@pytest.mark.asyncio
async def test_embed_uses_whatever_host_the_pool_picks(monkeypatch):
    calls: list[str] = []
    tracked_hosts: list[str] = []

    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"embedding": [0.1, 0.2]}))

    class _RecordingTrack:
        def __init__(self, host):
            tracked_hosts.append(host)

        async def __aenter__(self):
            return None

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(matricxon_pool, "track_request", _RecordingTrack)
    monkeypatch.setattr(matricxon_pool, "mark_recovered", lambda _host: None)

    result = await matricxon_client.embed("hello world", "nomic-embed-text:latest")

    assert result == [0.1, 0.2]
    assert calls == ["http://pool-picked:8420/api/embeddings"]
    assert tracked_hosts == ["http://pool-picked:8420"]


@pytest.mark.asyncio
async def test_embed_raises_matricxon_error_on_http_failure(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://x/api/embeddings"))

    monkeypatch.setattr(matricxon_pool, "pick_host", lambda: "http://pool-picked:8420")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())
    monkeypatch.setattr(matricxon_pool, "track_request", lambda _host: _NullCtx())
    monkeypatch.setattr(matricxon_pool, "mark_failure", lambda _host: None)

    with pytest.raises(matricxon_client.MatricxonError):
        await matricxon_client.embed("hello world", "nomic-embed-text:latest")


def _chat_params():
    return {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,
        "num_predict": 128,
        "seed": -1,
    }


@pytest.mark.asyncio
async def test_chat_stream_routes_the_actual_http_call_through_stream_with_failover(monkeypatch):
    seen_hosts: list[str] = []

    async def fake_stream_with_failover(make_stream):
        async for chunk in make_stream("http://chosen-by-pool:8420"):
            yield chunk

    monkeypatch.setattr(matricxon_pool, "stream_with_failover", fake_stream_with_failover)

    class _FakeStreamCtx:
        is_error = False

        def __init__(self, url):
            seen_hosts.append(url)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield '{"message": {"content": "hi"}, "done": false}'
            yield '{"done": true}'

    class _FakeStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url, json):
            return _FakeStreamCtx(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeStreamClient())

    result = [c async for c in matricxon_client.chat_stream("fake-model", [], _chat_params())]

    assert result == ["hi"]
    assert seen_hosts == ["http://chosen-by-pool:8420/api/chat"]


@pytest.mark.asyncio
async def test_chat_stream_surfaces_matricxons_own_error_body_not_the_generic_http_status_text(monkeypatch):
    """The real bug this covers: httpx.HTTPStatusError's own str() never includes the response body, only
    status/url — a 503 from Matricxon's own InsufficientMemoryError carries the actual actionable reason (e.g.
    "not enough memory to load ...: need ~12.9GB, only 10.9GB available") in a {"error": "..."} JSON body, which
    must reach the admin instead of being silently dropped for a generic "503 Service Unavailable"."""

    async def fake_stream_with_failover(make_stream):
        async for chunk in make_stream("http://chosen-by-pool:8420"):
            yield chunk

    monkeypatch.setattr(matricxon_pool, "stream_with_failover", fake_stream_with_failover)

    class _FakeErrorStreamCtx:
        is_error = True
        status_code = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def aread(self):
            return b'{"error": "not enough memory to load \'x\': need ~12.9GB, only 10.9GB available."}'

        async def aiter_lines(self):
            return
            yield  # pragma: no cover - never reached, keeps this an async generator

    class _FakeStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url, json):
            return _FakeErrorStreamCtx()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeStreamClient())

    with pytest.raises(matricxon_client.MatricxonError, match="not enough memory to load 'x'"):
        async for _ in matricxon_client.chat_stream("fake-model", [], _chat_params()):
            pass


@pytest.mark.asyncio
async def test_chat_stream_raises_matricxon_error_on_http_failure(monkeypatch):
    async def fake_stream_with_failover(make_stream):
        async for _ in make_stream("http://chosen-by-pool:8420"):
            pass
        raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://x/api/chat"))
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(matricxon_pool, "stream_with_failover", fake_stream_with_failover)

    with pytest.raises(matricxon_client.MatricxonError):
        async for _ in matricxon_client.chat_stream("fake-model", [], _chat_params()):
            pass


@pytest.mark.asyncio
async def test_chat_once_concatenates_every_chunk(monkeypatch):
    async def fake_chat_stream(_model, _messages, _params):
        yield "hel"
        yield "lo"

    monkeypatch.setattr(matricxon_client, "chat_stream", fake_chat_stream)
    assert await matricxon_client.chat_once("fake-model", []) == "hello"


class _RecordingPostClient:
    def __init__(self, calls: list[tuple[str, dict]], fail_hosts: frozenset[str] = frozenset()):
        self._calls = calls
        self._fail_hosts = fail_hosts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def post(self, url: str, json: dict):
        self._calls.append((url, json))
        if any(url.startswith(host) for host in self._fail_hosts):
            raise httpx.ConnectError("boom", request=httpx.Request("POST", url))
        return None


@pytest.mark.asyncio
async def test_stop_model_sends_keep_alive_zero_to_every_configured_host(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://host-a:8420", "http://host-b:8420"])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _RecordingPostClient(calls))

    await matricxon_client.stop_model("mistral3:latest")

    assert calls == [
        ("http://host-a:8420/api/chat", {"model": "mistral3:latest", "messages": [], "keep_alive": 0}),
        ("http://host-b:8420/api/chat", {"model": "mistral3:latest", "messages": [], "keep_alive": 0}),
    ]


@pytest.mark.asyncio
async def test_stop_model_swallows_a_failure_on_one_host_and_still_tries_the_rest(monkeypatch):
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        matricxon_pool, "get_effective_hosts", lambda: ["http://unreachable:8420", "http://host-b:8420"]
    )
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _RecordingPostClient(calls, fail_hosts=frozenset(["http://unreachable"]))
    )

    await matricxon_client.stop_model("mistral3:latest")  # must not raise

    assert [url for url, _ in calls] == ["http://unreachable:8420/api/chat", "http://host-b:8420/api/chat"]
