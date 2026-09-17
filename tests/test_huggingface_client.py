"""Unit tests for app/services/huggingface_client.py. httpx.AsyncClient is replaced with a fake that records the
call and returns a canned response instead of hitting the real network — same "stand in for AsyncClient" pattern
tests/test_ollama_client.py already uses."""

import httpx
import pytest

from app.services import huggingface_client as hf


class _FakeResponse:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class _FakeAsyncClient:
    def __init__(self, calls: list, response: _FakeResponse | None = None, raise_exc: Exception | None = None):
        self._calls = calls
        self._response = response
        self._raise_exc = raise_exc

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def get(self, url: str, params: dict):
        self._calls.append((url, params))
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def _patch_client(monkeypatch, calls, response=None, raise_exc=None):
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(calls, response=response, raise_exc=raise_exc)
    )


@pytest.mark.asyncio
async def test_search_models_maps_real_search_response_fields(monkeypatch):
    calls: list = []
    body = [
        {"id": "Qwen/Qwen2.5-14B-Instruct-GGUF", "downloads": 12345, "likes": 42, "gated": False},
        {"id": "org/gated-repo", "downloads": 1, "likes": 0, "gated": "manual"},
    ]
    _patch_client(monkeypatch, calls, response=_FakeResponse(200, body))

    results = await hf.search_models("qwen2.5", proxy_url=None)

    assert results == [
        {"repo_id": "Qwen/Qwen2.5-14B-Instruct-GGUF", "downloads": 12345, "likes": 42, "gated": False, "license": None},
        {"repo_id": "org/gated-repo", "downloads": 1, "likes": 0, "gated": True, "license": None},
    ]
    url, params = calls[0]
    assert url == hf.HF_API_BASE
    assert params["search"] == "qwen2.5"
    assert params["filter"] == "gguf"


@pytest.mark.asyncio
async def test_search_models_returns_empty_list_for_no_matches(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, response=_FakeResponse(200, []))

    assert await hf.search_models("zzz-no-such-model-zzz", proxy_url=None) == []


@pytest.mark.asyncio
async def test_search_models_raises_on_a_network_failure(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, raise_exc=httpx.ConnectError("no route"))

    with pytest.raises(hf.HuggingFaceLookupError, match="Could not reach Hugging Face"):
        await hf.search_models("qwen2.5", proxy_url=None)


@pytest.mark.asyncio
async def test_search_models_raises_on_a_non_200_status(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, response=_FakeResponse(500, {}))

    with pytest.raises(hf.HuggingFaceLookupError, match="unexpected error"):
        await hf.search_models("qwen2.5", proxy_url=None)


@pytest.mark.asyncio
async def test_get_repo_files_maps_real_repo_response_fields(monkeypatch):
    calls: list = []
    body = {
        "id": "Qwen/Qwen2.5-14B-Instruct-GGUF",
        "gated": False,
        "cardData": {"license": "apache-2.0"},
        "gguf": {"total": 14770033664, "architecture": "qwen2", "context_length": 32768},
        "siblings": [
            {"rfilename": "README.md"},
            {"rfilename": "qwen2.5-14b-instruct-q4_k_m.gguf", "size": 9_500_000_000},
            {"rfilename": "qwen2.5-14b-instruct-q8_0-00001-of-00002.gguf", "size": 8_000_000_000},
            {"rfilename": "qwen2.5-14b-instruct-fp16.gguf", "size": 0},
        ],
    }
    _patch_client(monkeypatch, calls, response=_FakeResponse(200, body))

    repo = await hf.get_repo_files("Qwen/Qwen2.5-14B-Instruct-GGUF", proxy_url=None)

    assert repo["repo_id"] == "Qwen/Qwen2.5-14B-Instruct-GGUF"
    assert repo["family"] == "qwen2"
    assert repo["parameter_size"] == "14.8B"
    assert repo["context_length"] == 32768
    assert repo["license"] == "apache-2.0"
    # README.md (not gguf), the sharded "-of-" file, and the zero-size file are all excluded
    assert repo["files"] == [{"filename": "qwen2.5-14b-instruct-q4_k_m.gguf", "download_gb": 9.5}]
    url, params = calls[0]
    assert url == f"{hf.HF_API_BASE}/Qwen/Qwen2.5-14B-Instruct-GGUF"
    assert params == {"blobs": "true"}


@pytest.mark.asyncio
async def test_get_repo_files_prefers_license_name_over_license(monkeypatch):
    calls: list = []
    body = {
        "gated": False,
        "cardData": {"license": "other", "license_name": "qwen-research"},
        "gguf": {},
        "siblings": [],
    }
    _patch_client(monkeypatch, calls, response=_FakeResponse(200, body))

    repo = await hf.get_repo_files("org/repo", proxy_url=None)

    assert repo["license"] == "qwen-research"
    assert repo["family"] is None
    assert repo["parameter_size"] is None
    assert repo["files"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 404])
async def test_get_repo_files_treats_401_and_404_as_not_found(monkeypatch, status_code):
    """Hugging Face's own API returns 401, not 404, for an unauthenticated request to a nonexistent repo
    (confirmed live) — both must produce the same "not found" error."""
    calls: list = []
    _patch_client(monkeypatch, calls, response=_FakeResponse(status_code, {"error": "Invalid username or password."}))

    with pytest.raises(hf.HuggingFaceLookupError, match="was not found on Hugging Face"):
        await hf.get_repo_files("org/nope", proxy_url=None)


@pytest.mark.asyncio
async def test_get_repo_files_raises_on_a_network_failure(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, raise_exc=httpx.ConnectTimeout("timed out"))

    with pytest.raises(hf.HuggingFaceLookupError, match="Could not reach Hugging Face"):
        await hf.get_repo_files("org/repo", proxy_url=None)


@pytest.mark.asyncio
async def test_get_repo_files_raises_on_an_unexpected_status(monkeypatch):
    calls: list = []
    _patch_client(monkeypatch, calls, response=_FakeResponse(503, {}))

    with pytest.raises(hf.HuggingFaceLookupError, match="unexpected error"):
        await hf.get_repo_files("org/repo", proxy_url=None)


def test_is_single_file_gguf_excludes_sharded_files():
    assert hf._is_single_file_gguf("model.gguf") is True
    assert hf._is_single_file_gguf("model-00001-of-00002.gguf") is False
    assert hf._is_single_file_gguf("README.md") is False


@pytest.mark.parametrize(
    "total,expected",
    [
        (None, None),
        (0, None),
        (500_000_000, "500M"),
        (1_200_000_000, "1.2B"),
        (14_770_033_664, "14.8B"),
    ],
)
def test_format_param_count(total, expected):
    assert hf._format_param_count(total) == expected
