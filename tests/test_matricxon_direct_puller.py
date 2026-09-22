"""Unit tests for app/services/matricxon_direct_puller.py. Every network-touching dependency
(HuggingFaceCatalogSearch.repo_files, gguf_probe.probe_metadata, httpx.AsyncClient,
MatricxonDirectPuller._get_config, matricxon_process.resolve_project_dir) is monkeypatched, so no real
Hugging Face, Matricxon, or database call happens here."""

import asyncio
import json

import httpx
import pytest

from app.schemas import MatricxonServerConfig
from app.services import gguf_probe, matricxon_process
from app.services.huggingface_client import HuggingFaceCatalogSearch
from app.services.matricxon_client import MatricxonError
from app.services.matricxon_direct_puller import MatricxonDirectPuller

_TAG = "hf.co/org/repo:model-q4_k_m"
_REPO_FILES = {
    "repo_id": "org/repo",
    "context_length": 4096,
    "files": [
        {
            "filename": "model-q4_k_m.gguf",
            "download_gb": 0.0,
            "is_projector": False,
            "sha256": None,
            "size_bytes": 12,
        }
    ],
}
_CONTENT = b"fake gguf bb"  # 12 bytes, matches _REPO_FILES's size_bytes


async def _async_return(value):
    return value


class _FakeStreamResponse:
    def __init__(self, content: bytes):
        self._content = content

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        for i in range(0, len(self._content), chunk_size):
            yield self._content[i : i + chunk_size]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


class _FakePostResponse:
    def __init__(self, capabilities: list[str]):
        self._capabilities = capabilities

    def raise_for_status(self):
        pass

    def json(self):
        return {"capabilities": self._capabilities}


class _FakeAsyncClient:
    def __init__(self, content: bytes, capabilities: list[str]):
        self._content = content
        self._capabilities = capabilities

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, _url):
        return _FakeStreamResponse(self._content)

    async def post(self, _url, json):  # noqa: A002 - matches httpx's own kwarg name
        return _FakePostResponse(self._capabilities)


def _patch_config(monkeypatch, **overrides) -> MatricxonServerConfig:
    """Stands in for the real DB-backed MatricxonServerConfig fetch (MatricxonDirectPuller._get_config
    opens its own AsyncSessionLocal — see that method's own docstring) — no real database involved here."""
    config = MatricxonServerConfig(**overrides)
    monkeypatch.setattr(MatricxonDirectPuller, "_get_config", staticmethod(lambda: _async_return(config)))
    return config


def _patch_success(monkeypatch, tmp_path, content: bytes = _CONTENT, capabilities: list[str] | None = None):
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(_REPO_FILES))
    )
    _patch_config(monkeypatch)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _project_dir=None: tmp_path)
    monkeypatch.setattr(
        gguf_probe,
        "probe_metadata",
        lambda repo_id, filename, proxy_url: _async_return(
            {"architecture": "mistral3", "name": "model", "parameter_count": 3_000_000_000}
        ),
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(content, capabilities or ["completion"]))


@pytest.mark.asyncio
async def test_pull_stream_writes_the_gguf_and_a_matching_sidecar(monkeypatch, tmp_path):
    _patch_success(monkeypatch, tmp_path)

    progress = [p async for p in MatricxonDirectPuller().pull_stream(_TAG)]

    assert progress[0] == {"status": "resolving manifest"}
    assert progress[-1] == {"status": "success"}

    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert gguf_path.read_bytes() == _CONTENT
    sidecar = json.loads(gguf_path.with_suffix(".gguf.json").read_text())
    assert sidecar == {
        "tag": _TAG,
        "path": str(gguf_path),
        "architecture": "mistral3",
        "capabilities": ["completion"],
        "size_bytes": 12,
        "family": "mistral3",
        "parameter_size": "3.0B",
        "context_length": 4096,
    }


@pytest.mark.asyncio
async def test_pull_stream_leaves_no_partial_file_behind_on_success(monkeypatch, tmp_path):
    _patch_success(monkeypatch, tmp_path)

    async for _ in MatricxonDirectPuller().pull_stream(_TAG):
        pass

    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert not gguf_path.with_suffix(".gguf.partial").exists()


@pytest.mark.asyncio
async def test_pull_stream_uses_a_configured_models_path_directly(monkeypatch, tmp_path):
    """When models_path is set, it's used as-is (no /data/models suffix appended) and
    matricxon_process.resolve_project_dir is never even consulted — this is the actual fix for the
    field's own bug: models silently landing wherever the old hardcoded guess pointed."""
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(_REPO_FILES))
    )
    monkeypatch.setattr(
        gguf_probe,
        "probe_metadata",
        lambda repo_id, filename, proxy_url: _async_return(
            {"architecture": "mistral3", "name": "model", "parameter_count": 3_000_000_000}
        ),
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_CONTENT, ["completion"]))
    _patch_config(monkeypatch, models_path=str(tmp_path / "custom-models"))

    def fail_if_called(_project_dir=None):
        raise AssertionError("resolve_project_dir must not be consulted when models_path is set")

    monkeypatch.setattr(matricxon_process, "resolve_project_dir", fail_if_called)

    async for _ in MatricxonDirectPuller().pull_stream(_TAG):
        pass

    gguf_path = tmp_path / "custom-models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert gguf_path.read_bytes() == _CONTENT


@pytest.mark.asyncio
async def test_pull_stream_honors_a_configured_project_dir_override(monkeypatch, tmp_path):
    """The other real bug this closes: _resolve_dest used to always call the no-override
    auto_detect_project_dir(), silently ignoring whatever project_dir the admin had already saved —
    now it's threaded through to matricxon_process.resolve_project_dir instead."""
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(_REPO_FILES))
    )
    monkeypatch.setattr(
        gguf_probe,
        "probe_metadata",
        lambda repo_id, filename, proxy_url: _async_return(
            {"architecture": "mistral3", "name": "model", "parameter_count": 3_000_000_000}
        ),
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_CONTENT, ["completion"]))
    _patch_config(monkeypatch, project_dir="/configured/checkout")

    received = []

    def fake_resolve(project_dir=None):
        received.append(project_dir)
        return tmp_path

    monkeypatch.setattr(matricxon_process, "resolve_project_dir", fake_resolve)

    async for _ in MatricxonDirectPuller().pull_stream(_TAG):
        pass

    assert received == ["/configured/checkout"]


@pytest.mark.asyncio
async def test_pull_stream_raises_and_cleans_up_on_a_sha256_mismatch(monkeypatch, tmp_path):
    repo_with_sha = {**_REPO_FILES, "files": [{**_REPO_FILES["files"][0], "sha256": "0" * 64}]}
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(lambda repo_id, proxy_url: _async_return(repo_with_sha)),
    )
    _patch_config(monkeypatch)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _project_dir=None: tmp_path)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _FakeAsyncClient(_CONTENT, ["completion"]))

    with pytest.raises(MatricxonError, match="sha256"):
        async for _ in MatricxonDirectPuller().pull_stream(_TAG):
            pass

    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert not gguf_path.exists()
    assert not gguf_path.with_suffix(".gguf.partial").exists()


def test_match_finds_a_short_human_friendly_suffix_by_substring():
    """Confirmed live (2026-09-21): a real curated default_models.json tag's suffix ("Q8_0") is much shorter
    than the real file's own stem ("all-MiniLM-L6-v2.Q8_0") — an exact-stem check alone would miss it."""
    files = [{"filename": "all-MiniLM-L6-v2.Q8_0.gguf"}, {"filename": "all-MiniLM-L6-v2.Q4_K_M.gguf"}]
    assert MatricxonDirectPuller._match("Q8_0", files)["filename"] == "all-MiniLM-L6-v2.Q8_0.gguf"


def test_match_prefers_an_exact_stem_match_over_a_substring_one():
    files = [{"filename": "model-q4_k_m.gguf"}, {"filename": "model-q4_k_m-extra.gguf"}]
    assert MatricxonDirectPuller._match("model-q4_k_m", files)["filename"] == "model-q4_k_m.gguf"


def test_match_raises_when_nothing_matches():
    with pytest.raises(MatricxonError, match="No .gguf file matching"):
        MatricxonDirectPuller._match("nope", [{"filename": "model.gguf"}])


def test_match_raises_when_the_suffix_is_ambiguous():
    files = [{"filename": "model-q4_k_m.gguf"}, {"filename": "model-q4_k_m-v2.gguf"}]
    with pytest.raises(MatricxonError, match="Ambiguous"):
        MatricxonDirectPuller._match("q4_k_m", files)


class TestDescribe:
    """Real bug found live (2026-09-22): a transient httpx failure stringified to an empty message,
    surfacing as "download of ... failed: " with nothing after the colon — no clue what actually failed."""

    def test_uses_the_exceptions_own_message_when_it_has_one(self):
        assert MatricxonDirectPuller._describe(ValueError("boom")) == "boom"

    def test_falls_back_to_the_type_name_when_the_message_is_empty(self):
        assert MatricxonDirectPuller._describe(ValueError()) == "ValueError"

    def test_falls_back_to_the_type_name_and_url_for_an_empty_httpx_error_with_a_request(self):
        request = httpx.Request("GET", "https://huggingface.co/org/repo/resolve/main/model.gguf")
        exc = httpx.ConnectError("", request=request)
        assert MatricxonDirectPuller._describe(exc) == (
            "ConnectError for https://huggingface.co/org/repo/resolve/main/model.gguf"
        )


@pytest.mark.asyncio
async def test_pull_stream_raises_when_the_file_is_no_longer_on_hugging_face(monkeypatch, tmp_path):
    monkeypatch.setattr(
        HuggingFaceCatalogSearch,
        "repo_files",
        staticmethod(lambda repo_id, proxy_url: _async_return({**_REPO_FILES, "files": []})),
    )
    _patch_config(monkeypatch)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _project_dir=None: tmp_path)

    with pytest.raises(MatricxonError, match="No .gguf file matching"):
        async for _ in MatricxonDirectPuller().pull_stream(_TAG):
            pass


@pytest.mark.asyncio
async def test_pull_stream_raises_when_matricxons_local_install_cannot_be_found(monkeypatch, tmp_path):
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(_REPO_FILES))
    )
    _patch_config(monkeypatch)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _project_dir=None: None)

    with pytest.raises(MatricxonError, match="couldn't be found"):
        async for _ in MatricxonDirectPuller().pull_stream(_TAG):
            pass


@pytest.mark.parametrize("bad_tag", ["not-an-hf-tag", "hf.co/org-repo-no-colon"])
@pytest.mark.asyncio
async def test_pull_stream_raises_for_a_non_hf_tag(bad_tag):
    with pytest.raises(MatricxonError):
        async for _ in MatricxonDirectPuller().pull_stream(bad_tag):
            pass


@pytest.mark.asyncio
async def test_pull_stream_skips_the_download_when_the_file_already_exists(monkeypatch, tmp_path):
    """The real case this guards against: a concurrent pull of the exact same tag already finished while this
    one was waiting for the coordinator's lock (see MatricxonDirectPuller's own module docstring on the real
    bug — an orphaned, still-running pull racing a retry — this and the lock together prevent)."""
    _patch_success(monkeypatch, tmp_path)
    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    gguf_path.parent.mkdir(parents=True)
    gguf_path.write_bytes(_CONTENT)

    _patch_success(monkeypatch, tmp_path)

    class _NoDownloadClient(_FakeAsyncClient):
        def stream(self, _method, _url):
            raise AssertionError("the download should have been skipped — the file already exists")

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _NoDownloadClient(_CONTENT, ["completion"]))

    progress = [p async for p in MatricxonDirectPuller().pull_stream(_TAG)]

    assert {"status": "already downloaded", "completed": 12, "total": 12} in progress
    assert progress[-1] == {"status": "success"}


@pytest.mark.asyncio
async def test_pull_stream_serializes_two_concurrent_pulls_of_the_same_tag(monkeypatch, tmp_path):
    """Two real concurrent MatricxonDirectPuller().pull_stream(_TAG) calls (not just one retried after the
    other) must not both write the same .partial file — the second one waits for the coordinator's lock, then
    finds the file already there and skips straight to "already downloaded" instead of racing the first."""
    _patch_success(monkeypatch, tmp_path)

    async def _drain(tag):
        return [p async for p in MatricxonDirectPuller().pull_stream(tag)]

    results = await asyncio.gather(_drain(_TAG), _drain(_TAG))

    statuses = [{p["status"] for p in result} for result in results]
    # Exactly one of the two actually downloaded; the other found it already there.
    assert any("already downloaded" in s for s in statuses)
    assert all("success" in s for s in statuses)
    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert gguf_path.read_bytes() == _CONTENT


@pytest.mark.asyncio
async def test_pull_stream_cleans_up_the_partial_file_on_cancellation(monkeypatch, tmp_path):
    """A client disconnecting mid-download (see this module's own docstring on the real bug this — and a
    matching fix on Matricxon's own side — closes) must not leave a half-written .partial file behind."""
    monkeypatch.setattr(
        HuggingFaceCatalogSearch, "repo_files", staticmethod(lambda repo_id, proxy_url: _async_return(_REPO_FILES))
    )
    _patch_config(monkeypatch)
    monkeypatch.setattr(matricxon_process, "resolve_project_dir", lambda _project_dir=None: tmp_path)

    class _HangingStreamResponse(_FakeStreamResponse):
        async def aiter_bytes(self, chunk_size):
            yield self._content[:4]
            await asyncio.sleep(3600)  # never actually reached — the task is cancelled first

    class _HangingClient(_FakeAsyncClient):
        def stream(self, _method, _url):
            return _HangingStreamResponse(self._content)

    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kw: _HangingClient(_CONTENT, ["completion"]))

    async def _drain():
        async for _ in MatricxonDirectPuller().pull_stream(_TAG):
            pass

    task = asyncio.ensure_future(_drain())
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    gguf_path = tmp_path / "data" / "models" / "hf.co" / "org" / "repo" / "model-q4_k_m.gguf"
    assert not gguf_path.exists()
    assert not gguf_path.with_suffix(".gguf.partial").exists()
