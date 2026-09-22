"""Unit tests for app/services/matricxon_admin.py (the pull/delete service, not the router of the same name in
app/routers/) — mirrors the HTTP-wiring style of tests/test_ollama_client.py, confirming pull/delete target
matricxon_pool's primary host and surface a real error as MatricxonError."""

import httpx
import pytest

from app.services import matricxon_admin, matricxon_pool
from app.services.matricxon_direct_puller import MatricxonDirectPuller


@pytest.mark.asyncio
async def test_pull_model_stream_uses_the_direct_puller_when_matricxon_is_local(monkeypatch):
    """See app.services.matricxon_direct_puller's own module docstring: only usable (and only used) when
    pAIring and Matricxon share a filesystem, i.e. matricxon_pool's effective hosts are exactly
    [LOCAL_MATRICXON_HOST] — every other test in this file uses a host list that doesn't match this, so they
    keep exercising the old proxy-through-/api/pull path unchanged (confirmed by them still passing)."""
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: [matricxon_pool.LOCAL_MATRICXON_HOST])

    async def fake_pull_stream(self, tag):
        yield {"status": "resolving manifest"}
        yield {"status": "success"}

    monkeypatch.setattr(MatricxonDirectPuller, "pull_stream", fake_pull_stream)

    result = [p async for p in matricxon_admin.pull_model_stream("hf.co/some/repo:tag")]

    assert result == [{"status": "resolving manifest"}, {"status": "success"}]


@pytest.mark.asyncio
async def test_pull_model_stream_targets_the_primary_host_and_yields_progress(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://primary:8420", "http://other:8420"])
    captured = {}

    class _FakeStreamCtx:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield '{"status": "pulling"}'
            yield '{"status": "success"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, method, url, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return _FakeStreamCtx()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())

    result = [p async for p in matricxon_admin.pull_model_stream("hf.co/some/repo:tag")]

    assert result == [{"status": "pulling"}, {"status": "success"}]
    assert captured["url"] == "http://primary:8420/api/pull"
    assert captured["json"] == {"model": "hf.co/some/repo:tag", "stream": True}


@pytest.mark.asyncio
async def test_pull_model_stream_raises_matricxon_error_on_an_error_frame(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://primary:8420"])

    class _FakeStreamCtx:
        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield '{"error": "unknown model"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url, json):
            return _FakeStreamCtx()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())

    with pytest.raises(matricxon_admin.MatricxonError):
        async for _ in matricxon_admin.pull_model_stream("hf.co/some/repo:tag"):
            pass


@pytest.mark.asyncio
async def test_delete_model_targets_the_primary_host(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://primary:8420"])
    captured = {}

    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, method, url, json):
            captured["method"] = method
            captured["url"] = url
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeClient())

    await matricxon_admin.delete_model("hf.co/some/repo:tag")

    assert captured["method"] == "DELETE"
    assert captured["url"] == "http://primary:8420/api/delete"
    assert captured["json"] == {"model": "hf.co/some/repo:tag"}


@pytest.mark.asyncio
async def test_delete_model_raises_matricxon_error_on_http_failure(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://primary:8420"])

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def request(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("DELETE", "http://x/api/delete"))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())

    with pytest.raises(matricxon_admin.MatricxonError):
        await matricxon_admin.delete_model("hf.co/some/repo:tag")
