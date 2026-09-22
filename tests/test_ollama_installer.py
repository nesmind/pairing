"""Unit tests for app/services/ollama_installer.py — every httpx call
and subprocess.run (tar extraction) is monkeypatched; no real network
access or multi-GB download happens in this test suite."""

import subprocess

import httpx
import pytest

from app.services import ollama_installer


def test_arch_name_maps_common_machine_strings(monkeypatch):
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")
    assert ollama_installer._arch_name() == "amd64"
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "aarch64")
    assert ollama_installer._arch_name() == "arm64"


def test_arch_name_raises_for_an_unknown_machine(monkeypatch):
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "sparc64")
    with pytest.raises(RuntimeError):
        ollama_installer._arch_name()


def test_asset_url_uses_the_given_repo_and_version(monkeypatch):
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")
    url = ollama_installer._asset_url("ollama/ollama", "v9.9.9")
    assert url == "https://github.com/ollama/ollama/releases/download/v9.9.9/ollama-linux-amd64.tar.zst"


def test_is_installed_delegates_to_ollama_process(monkeypatch):
    from app.services import ollama_process

    monkeypatch.setattr(ollama_process, "is_installed", lambda: True)
    assert ollama_installer.is_installed() is True


@pytest.mark.asyncio
async def test_install_stream_reports_error_on_an_unsupported_os(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer, "local_install_supported", lambda: False)
    monkeypatch.setattr(ollama_installer.platform, "system", lambda: "Darwin")

    events = [event async for event in ollama_installer.install_stream()]

    assert len(events) == 1
    assert "Darwin" in events[0]["error"]
    assert "Linux" in events[0]["error"]


@pytest.mark.asyncio
async def test_install_stream_reports_error_when_tar_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: None)

    events = [event async for event in ollama_installer.install_stream()]

    assert len(events) == 1
    assert "tar" in events[0]["error"]


@pytest.mark.asyncio
async def test_install_stream_reports_error_for_an_unknown_architecture(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "sparc64")

    events = [event async for event in ollama_installer.install_stream()]

    assert len(events) == 1
    assert "architecture" in events[0]["error"]


class _FakeStreamResponse:
    def __init__(self, chunks: list[bytes], total: int | None):
        self._chunks = chunks
        self.headers = {"content-length": str(total)} if total else {}

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        for chunk in self._chunks:
            yield chunk


class _FakeStreamClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    def stream(self, _method, _url):
        return self


class _FakeStreamContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *_exc):
        return False


@pytest.mark.asyncio
async def test_install_stream_downloads_extracts_and_reports_done(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    response = _FakeStreamResponse([b"a" * 500, b"b" * 500], total=1000)

    class _Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(response)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    def fake_run(argv, **_kwargs):
        # Simulate tar actually producing the expected binary layout.
        target_dir = tmp_path / "ollama"
        (target_dir / "bin").mkdir(parents=True, exist_ok=True)
        (target_dir / "bin" / "ollama").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    events = [event async for event in ollama_installer.install_stream()]

    assert events[-1] == {
        "done": True,
        "step": 2,
        "total_steps": 2,
        "binary_path": str(ollama_installer.binary_path()),
    }
    assert ollama_installer.binary_path().exists()
    assert not (tmp_path / "ollama" / "ollama.tar.zst").exists()  # archive cleaned up


@pytest.mark.asyncio
async def test_install_stream_passes_proxy_url_to_the_http_client(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    captured_kwargs = {}

    class _Client:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(_FakeStreamResponse([b"x"], total=None))

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
    )

    [event async for event in ollama_installer.install_stream(proxy_url="http://10.0.0.5:8080")]

    assert captured_kwargs["proxy"] == "http://10.0.0.5:8080"


@pytest.mark.asyncio
async def test_install_stream_passes_no_proxy_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    captured_kwargs = {}

    class _Client:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(_FakeStreamResponse([b"x"], total=None))

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")
    )

    [event async for event in ollama_installer.install_stream()]

    assert captured_kwargs["proxy"] is None


@pytest.mark.asyncio
async def test_install_stream_only_reports_progress_on_whole_percent_changes(monkeypatch, tmp_path):
    """A real, confirmed-live perf issue: yielding an SSE event on every
    1MB chunk of a multi-GB download (thousands of events) measurably
    slowed the download itself down — a slow-to-render browser creates
    real backpressure on this same coroutine's own network read loop.
    500 chunks here move the percentage by 0.1% each — reporting on every
    single one would yield 500 events; reporting only on whole-percent
    changes should yield roughly 50."""
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    chunks = [b"x" * 1000] * 500  # 500,000 of 1,000,000 total bytes
    response = _FakeStreamResponse(chunks, total=1_000_000)

    class _Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(response)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    def fake_run(argv, **_kwargs):
        target_dir = tmp_path / "ollama"
        (target_dir / "bin").mkdir(parents=True, exist_ok=True)
        (target_dir / "bin" / "ollama").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    events = [event async for event in ollama_installer.install_stream()]

    download_events = [e for e in events if e.get("step") == 1 and "completed" in e]
    assert 0 < len(download_events) <= 51
    # Every reported percentage is genuinely distinct — no duplicate
    # back-to-back events for the same whole percent.
    reported_pcts = [e["completed"] * 100 // e["total"] for e in download_events]
    assert reported_pcts == sorted(set(reported_pcts))


@pytest.mark.asyncio
async def test_install_stream_uses_a_given_repo_and_version_override(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    requested_urls = []

    class _Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, url):
            requested_urls.append(url)
            return _FakeStreamContext(_FakeStreamResponse([b"x"], total=None))

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    def fake_run(argv, **_kwargs):
        target_dir = tmp_path / "ollama"
        (target_dir / "bin").mkdir(parents=True, exist_ok=True)
        (target_dir / "bin" / "ollama").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    events = [event async for event in ollama_installer.install_stream(repo="me/ollama-fork", version="v1.2.3")]

    assert requested_urls == ["https://github.com/me/ollama-fork/releases/download/v1.2.3/ollama-linux-amd64.tar.zst"]
    assert "v1.2.3" in events[0]["status"]


@pytest.mark.asyncio
async def test_install_stream_reports_step_progress_through_download_and_extract(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    response = _FakeStreamResponse([b"a" * 500, b"b" * 500], total=1000)

    class _Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(response)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    def fake_run(argv, **_kwargs):
        target_dir = tmp_path / "ollama"
        (target_dir / "bin").mkdir(parents=True, exist_ok=True)
        (target_dir / "bin" / "ollama").write_text("#!/bin/sh\n")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    events = [event async for event in ollama_installer.install_stream()]

    download_events = [e for e in events if e.get("step") == 1]
    extract_events = [e for e in events if e.get("step") == 2 and not e.get("done")]
    assert all(e["total_steps"] == 2 and e["step_label"] == "Downloading" for e in download_events)
    assert all(e["total_steps"] == 2 and e["step_label"] == "Extracting" for e in extract_events)
    assert len(extract_events) == 1


@pytest.mark.asyncio
async def test_install_stream_reports_error_when_extraction_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    response = _FakeStreamResponse([b"x" * 10], total=None)

    class _Client:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            return _FakeStreamContext(response)

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    monkeypatch.setattr(
        subprocess, "run", lambda argv, **_kw: subprocess.CompletedProcess(argv, 1, stdout="", stderr="bad archive")
    )

    events = [event async for event in ollama_installer.install_stream()]

    assert "error" in events[-1]
    assert "bad archive" in events[-1]["error"]


@pytest.mark.asyncio
async def test_install_stream_reports_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(ollama_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(ollama_installer.shutil, "which", lambda _name: "/usr/bin/tar")
    monkeypatch.setattr(ollama_installer.platform, "machine", lambda: "x86_64")

    class _FailingClient:
        def __init__(self, **_kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc):
            return False

        def stream(self, _method, _url):
            raise httpx.ConnectError("refused", request=httpx.Request("GET", "http://x"))

    monkeypatch.setattr(httpx, "AsyncClient", _FailingClient)

    events = [event async for event in ollama_installer.install_stream()]

    assert "error" in events[-1]
    assert "Could not download" in events[-1]["error"]
