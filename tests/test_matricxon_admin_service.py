"""Tests for services/matricxon_admin.py (the pull/delete service, not the router in app/routers/): the HTTP-wiring
style of tests/test_ollama_client.py, confirming pull/delete target matricxon_pool's primary host and surface
real errors as MatricxonError.
"""

import httpx
import pytest

from app.services import matricxon_admin, matricxon_pool
from app.services.matricxon_direct_puller import MatricxonDirectPuller


@pytest.fixture(autouse=True)
def _no_consensus_by_default(monkeypatch):
    """Most tests are a plain non-cluster Matricxon: no lease, the primary host does everything."""

    async def standalone(_lease):
        return False, None

    monkeypatch.setattr(matricxon_pool, "lease_holder", standalone)


@pytest.mark.asyncio
async def test_in_a_cluster_models_are_changed_by_the_lease_owner_and_refused_without_one(monkeypatch):
    monkeypatch.setattr(matricxon_pool, "get_effective_hosts", lambda: ["http://primary:8420", "http://owner:8420"])
    answers = {"holder": (True, "http://owner:8420")}

    async def lease_holder(_lease):
        return answers["holder"]

    monkeypatch.setattr(matricxon_pool, "lease_holder", lease_holder)
    assert await matricxon_admin._model_host() == "http://owner:8420"
    answers["holder"] = (True, None)
    with pytest.raises(matricxon_admin.MatricxonError, match="no quorum"):
        await matricxon_admin._model_host()
    answers["holder"] = (False, None)
    assert await matricxon_admin._model_host() == "http://primary:8420"
    with pytest.raises(matricxon_admin.MatricxonError, match="no quorum"):
        answers["holder"] = (True, None)
        await matricxon_admin.delete_model("some-model")


@pytest.mark.asyncio
async def test_pull_model_stream_uses_the_direct_puller_when_matricxon_is_local(monkeypatch):
    """See matricxon_direct_puller: used only when pAIring and Matricxon share a filesystem, i.e. the effective hosts
    are exactly [LOCAL_MATRICXON_HOST]. Other tests in this file use another host list and keep the
    proxy-through-/api/pull path.
    """
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
