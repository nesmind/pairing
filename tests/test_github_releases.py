"""Unit tests for app/services/github_releases.py — the shared GitHub-tags fetcher behind Settings > External
servers' version-picker datalist (see app/routers/ollama_admin.py|comfyui_admin.py|matricxon_admin.py's shared
available-versions endpoint, the only real callers). No real network access happens here — every httpx.AsyncClient
call is monkeypatched."""

import httpx
import pytest

from app.services import github_releases


@pytest.fixture(autouse=True)
def _clear_cache():
    github_releases._cache.clear()
    yield
    github_releases._cache.clear()


class _FakeResponse:
    def __init__(self, json_body, status_code=200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("boom", request=httpx.Request("GET", "http://x"), response=self)

    def json(self):
        return self._json_body


class _FakeAsyncClient:
    def __init__(self, responder):
        self._responder = responder
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url, params=None, headers=None):
        self.calls += 1
        return self._responder(url, params, headers)


@pytest.mark.asyncio
async def test_list_tags_returns_empty_for_a_blank_repo():
    assert await github_releases.list_tags("") == []
    assert await github_releases.list_tags(None) == []


@pytest.mark.asyncio
async def test_list_tags_returns_names_newest_first_as_github_sends_them(monkeypatch):
    fake_client = _FakeAsyncClient(
        lambda *_a: _FakeResponse([{"name": "v0.2"}, {"name": "v0.1"}, {"name": "v0.1-rc1"}])
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: fake_client)

    result = await github_releases.list_tags("me/repo")

    assert result == ["v0.2", "v0.1", "v0.1-rc1"]


@pytest.mark.asyncio
async def test_list_tags_uses_the_cache_on_a_second_call_within_the_ttl(monkeypatch):
    fake_client = _FakeAsyncClient(lambda *_a: _FakeResponse([{"name": "v1"}]))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: fake_client)

    first = await github_releases.list_tags("me/repo")
    second = await github_releases.list_tags("me/repo")

    assert first == second == ["v1"]
    assert fake_client.calls == 1  # the second call was served from cache, no real request


@pytest.mark.asyncio
async def test_list_tags_refetches_once_the_cache_entry_has_expired(monkeypatch):
    fake_client = _FakeAsyncClient(lambda *_a: _FakeResponse([{"name": "v1"}]))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: fake_client)
    await github_releases.list_tags("me/repo")

    # Backdate the cache entry past _CACHE_TTL_SECONDS instead of sleeping for real.
    fetched_at, tags = github_releases._cache["me/repo"]
    github_releases._cache["me/repo"] = (fetched_at - github_releases._CACHE_TTL_SECONDS - 1, tags)

    await github_releases.list_tags("me/repo")

    assert fake_client.calls == 2


@pytest.mark.asyncio
async def test_list_tags_different_repos_are_cached_independently(monkeypatch):
    def responder(url, *_a):
        return _FakeResponse([{"name": "repo-a-tag"}]) if "repo-a" in url else _FakeResponse([{"name": "repo-b-tag"}])

    fake_client = _FakeAsyncClient(responder)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: fake_client)

    a = await github_releases.list_tags("me/repo-a")
    b = await github_releases.list_tags("me/repo-b")

    assert a == ["repo-a-tag"]
    assert b == ["repo-b-tag"]


@pytest.mark.asyncio
async def test_list_tags_returns_empty_on_a_network_failure_with_no_prior_cache(monkeypatch):
    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())

    assert await github_releases.list_tags("me/repo") == []


@pytest.mark.asyncio
async def test_list_tags_returns_empty_for_a_404_repo(monkeypatch):
    """A typo'd or private repo — GitHub's own 404 for it must degrade the same way any other failure does, not
    raise and break the whole Settings page."""
    fake_client = _FakeAsyncClient(lambda *_a: _FakeResponse({"message": "Not Found"}, status_code=404))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: fake_client)

    assert await github_releases.list_tags("me/typo-repo") == []


@pytest.mark.asyncio
async def test_list_tags_falls_back_to_stale_cache_on_a_later_failure(monkeypatch):
    """A transient GitHub outage after a previous successful fetch should still show the admin *something*
    (last-known-good) rather than silently emptying an already-populated datalist."""
    good_client = _FakeAsyncClient(lambda *_a: _FakeResponse([{"name": "v1"}]))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: good_client)
    await github_releases.list_tags("me/repo")
    fetched_at, tags = github_releases._cache["me/repo"]
    github_releases._cache["me/repo"] = (fetched_at - github_releases._CACHE_TTL_SECONDS - 1, tags)

    class _FailingClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        async def get(self, *_a, **_kw):
            raise httpx.ConnectError("boom", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FailingClient())

    assert await github_releases.list_tags("me/repo") == ["v1"]
