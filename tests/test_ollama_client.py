"""Integration-ish tests for app/services/ollama_client.py's embed and
chat_stream: confirms both are actually wired through the pool selector
(app.services.ollama_pool) rather than a single hardcoded host — the
pool's own routing/failover/cooldown logic is tested directly in
tests/test_ollama_pool.py; this file only checks the wiring, since a
refactor mistake here would silently drop pool routing back to a single
host without any test noticing."""

import asyncio

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from app.services import ollama_client, ollama_pool, ollama_telemetry


@pytest.fixture
def span_exporter(monkeypatch):
    """Wires ollama_telemetry's tracer at a real (in-memory, synchronous)
    SDK provider instead of the no-op default, so these tests can assert
    on what chat_stream/embed actually recorded — see
    app/services/ollama_telemetry.py's own docstring on why tests must
    never touch the global opentelemetry.trace registry to get this."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(ollama_telemetry, "_tracer_provider", provider)
    return exporter


class _FakeResponse:
    def __init__(self, json_body: dict):
        self._json_body = json_body

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._json_body


class _FakeAsyncClient:
    """Stands in for httpx.AsyncClient: records the URL each call was
    made against instead of hitting a real network."""

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


@pytest.mark.asyncio
async def test_embed_uses_whatever_host_the_pool_picks(monkeypatch):
    calls: list[str] = []
    tracked_hosts: list[str] = []

    monkeypatch.setattr(ollama_pool, "pick_host", lambda: "http://pool-picked:11434")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, {"embedding": [0.1, 0.2]}))

    class _RecordingTrack:
        def __init__(self, host):
            tracked_hosts.append(host)

        async def __aenter__(self):
            return None

        async def __aexit__(self, *_exc):
            return False

    monkeypatch.setattr(ollama_pool, "track_request", _RecordingTrack)
    monkeypatch.setattr(ollama_pool, "mark_recovered", lambda _host: None)

    result = await ollama_client.embed("hello world", "nomic-embed-text:latest")

    assert result == [0.1, 0.2]
    assert calls == ["http://pool-picked:11434/api/embeddings"]
    assert tracked_hosts == ["http://pool-picked:11434"]


@pytest.mark.asyncio
async def test_embed_records_a_successful_span(monkeypatch, span_exporter):
    monkeypatch.setattr(ollama_pool, "pick_host", lambda: "http://pool-picked:11434")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient([], {"embedding": [0.1]}))
    monkeypatch.setattr(ollama_pool, "track_request", lambda _host: _NullCtx())
    monkeypatch.setattr(ollama_pool, "mark_recovered", lambda _host: None)

    await ollama_client.embed("hello world", "nomic-embed-text:latest")

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "ollama.embed"
    assert span.status.status_code != StatusCode.ERROR
    assert span.attributes["server.address"] == "http://pool-picked:11434"
    assert span.attributes["gen_ai.request.model"] == "nomic-embed-text:latest"
    assert span.attributes["ollama.attempt_count"] == 1


class _NullCtx:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_embed_records_an_error_span_after_every_attempt_fails(monkeypatch, span_exporter):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def post(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://x/api/embeddings"))

    monkeypatch.setattr(ollama_pool, "pick_host", lambda: "http://pool-picked:11434")
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())
    monkeypatch.setattr(ollama_pool, "track_request", lambda _host: _NullCtx())
    monkeypatch.setattr(ollama_pool, "mark_failure", lambda _host: None)

    with pytest.raises(ollama_client.OllamaError):
        await ollama_client.embed("hello world", "nomic-embed-text:latest")

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "ollama.embed"
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["ollama.attempt_count"] == ollama_client._EMBED_MAX_ATTEMPTS


@pytest.mark.asyncio
async def test_chat_stream_routes_the_actual_http_call_through_stream_with_failover(monkeypatch):
    """chat_stream must hand its per-host request logic to
    ollama_pool.stream_with_failover rather than posting to a fixed host
    itself — verified here by making the *fake* stream_with_failover
    call back into chat_stream's own per-host function with a host of
    its choosing, and checking that host actually shows up in the
    (fake) HTTP call chat_stream makes."""
    seen_hosts: list[str] = []

    async def fake_stream_with_failover(make_stream):
        async for chunk in make_stream("http://chosen-by-pool:11434"):
            yield chunk

    async def fake_model_supports_thinking(_model: str) -> bool:
        return False

    monkeypatch.setattr(ollama_pool, "stream_with_failover", fake_stream_with_failover)
    monkeypatch.setattr(ollama_client, "model_supports_thinking", fake_model_supports_thinking)

    class _FakeStreamCtx:
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

    params = {
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,
        "num_predict": 128,
        "seed": -1,
    }
    result = [c async for c in ollama_client.chat_stream("fake-model", [], params)]

    assert result == ["hi"]
    assert seen_hosts == ["http://chosen-by-pool:11434/api/chat"]


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


class _FakeDoneStreamCtx:
    """A fake NDJSON stream ending with a real Ollama-shaped `done:true` line — so span attribute extraction
    (gen_ai.usage.*, ollama.*_duration_ns) has real data to pull from."""

    def __init__(self, url):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def raise_for_status(self):
        pass

    async def aiter_lines(self):
        yield '{"message": {"content": "hi"}, "done": false}'
        yield (
            '{"done": true, "total_duration": 5000000, "load_duration": 1000000, '
            '"prompt_eval_count": 7, "prompt_eval_duration": 2000000, "eval_count": 3, "eval_duration": 2000000}'
        )


@pytest.mark.asyncio
async def test_chat_stream_records_a_successful_span_with_ollama_timing(monkeypatch, span_exporter):
    async def fake_stream_with_failover(make_stream):
        async for chunk in make_stream("http://chosen-by-pool:11434"):
            yield chunk

    async def fake_model_supports_thinking(_model):
        return False

    monkeypatch.setattr(ollama_pool, "stream_with_failover", fake_stream_with_failover)
    monkeypatch.setattr(ollama_client, "model_supports_thinking", fake_model_supports_thinking)

    class _FakeStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url, json):
            return _FakeDoneStreamCtx(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeStreamClient())

    result = [c async for c in ollama_client.chat_stream("fake-model", [], _chat_params())]

    assert result == ["hi"]
    (span,) = span_exporter.get_finished_spans()
    assert span.name == "ollama.chat"
    assert span.status.status_code != StatusCode.ERROR
    assert span.attributes["server.address"] == "http://chosen-by-pool:11434"
    assert span.attributes["gen_ai.usage.input_tokens"] == 7
    assert span.attributes["gen_ai.usage.output_tokens"] == 3
    assert span.attributes["ollama.total_duration_ns"] == 5000000
    assert span.attributes["ollama.attempt_count"] == 1


@pytest.mark.asyncio
async def test_chat_stream_records_an_error_span_on_http_failure(monkeypatch, span_exporter):
    async def fake_stream_with_failover(make_stream):
        # Every host tried, all failing — matches stream_with_failover's own "gives up" contract.
        async for _ in make_stream("http://chosen-by-pool:11434"):
            pass
        raise httpx.ConnectError("boom", request=httpx.Request("POST", "http://x/api/chat"))
        yield  # pragma: no cover - makes this an async generator

    async def fake_model_supports_thinking(_model):
        return False

    monkeypatch.setattr(ollama_pool, "stream_with_failover", fake_stream_with_failover)
    monkeypatch.setattr(ollama_client, "model_supports_thinking", fake_model_supports_thinking)

    with pytest.raises(ollama_client.OllamaError):
        async for _ in ollama_client.chat_stream("fake-model", [], _chat_params()):
            pass

    (span,) = span_exporter.get_finished_spans()
    assert span.name == "ollama.chat"
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_chat_stream_does_not_mark_span_error_on_task_cancellation(monkeypatch, span_exporter):
    """The real cancellation path (see reply_generation_service.py:180 — a detached task's own `async for
    chunk in chat_stream(...)` cancelled via task.cancel()) throws CancelledError at the exact point that
    task is suspended, in the *same* context — not a separate gen.aclose() call from a different context,
    which is a distinct (and, with OTel's contextvars-based span tracking, less well-behaved) scenario. This
    must not read as an "error" on the dashboard — see ollama_client.py's own comment on why
    record_exception/set_status_on_exception are off."""

    chunk_yielded = asyncio.Event()

    async def fake_stream_with_failover(make_stream):
        async for chunk in make_stream("http://chosen-by-pool:11434"):
            yield chunk
            chunk_yielded.set()
            await asyncio.sleep(999)  # never resumes on its own — only via cancellation

    async def fake_model_supports_thinking(_model):
        return False

    monkeypatch.setattr(ollama_pool, "stream_with_failover", fake_stream_with_failover)
    monkeypatch.setattr(ollama_client, "model_supports_thinking", fake_model_supports_thinking)

    class _FakeStreamClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url, json):
            return _FakeDoneStreamCtx(url)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeStreamClient())

    async def _consume():
        async for _ in ollama_client.chat_stream("fake-model", [], _chat_params()):
            pass

    task = asyncio.create_task(_consume())
    await chunk_yielded.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (span,) = span_exporter.get_finished_spans()
    assert span.status.status_code != StatusCode.ERROR


class _RecordingPostClient:
    """Stands in for httpx.AsyncClient for stop_model's plain POST calls
    — records (url, json) pairs, optionally raising on a configured host
    to test that a failure on one host doesn't stop the rest."""

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
    monkeypatch.setattr(ollama_pool, "get_effective_hosts", lambda: ["http://host-a:11434", "http://host-b:11434"])
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _RecordingPostClient(calls))

    await ollama_client.stop_model("llava:latest")

    assert calls == [
        ("http://host-a:11434/api/chat", {"model": "llava:latest", "messages": [], "keep_alive": 0}),
        ("http://host-b:11434/api/chat", {"model": "llava:latest", "messages": [], "keep_alive": 0}),
    ]


@pytest.mark.asyncio
async def test_stop_model_swallows_a_failure_on_one_host_and_still_tries_the_rest(monkeypatch):
    """A cleanup nicety, not something that should ever raise and mask
    the real termination reason a caller is already handling — see
    stop_model's own docstring."""
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(ollama_pool, "get_effective_hosts", lambda: ["http://unreachable:11434", "http://host-b:11434"])
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _RecordingPostClient(calls, fail_hosts=frozenset(["http://unreachable"]))
    )

    await ollama_client.stop_model("llava:latest")  # must not raise

    assert [url for url, _ in calls] == ["http://unreachable:11434/api/chat", "http://host-b:11434/api/chat"]
