"""Unit tests for app/services/sdcpp_installer.py — GitHub is faked with httpx.MockTransport and a real in-memory
zip; nothing touches the network."""

import io
import zipfile

import httpx
import pytest

from app.services import sdcpp_installer

_REAL_CLIENT = httpx.AsyncClient


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _install_fake_github(monkeypatch, assets, archive: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/repos/"):
            return httpx.Response(200, json={"assets": assets})
        return httpx.Response(200, content=archive, headers={"content-length": str(len(archive))})

    monkeypatch.setattr(
        sdcpp_installer.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)
    )


@pytest.fixture(autouse=True)
def _linux_x86(tmp_path, monkeypatch):
    monkeypatch.setattr(sdcpp_installer, "EXTERNAL_DIR", tmp_path)
    monkeypatch.setattr(sdcpp_installer, "local_install_supported", lambda: True)
    monkeypatch.setattr(sdcpp_installer.platform, "machine", lambda: "x86_64")


_ASSETS = [
    {"name": "sd-master-abc-bin-Linux-Ubuntu-24.04-x86_64-vulkan.zip", "browser_download_url": "https://x/vulkan.zip"},
    {"name": "sd-master-abc-bin-win-cpu-x64.zip", "browser_download_url": "https://x/win.zip"},
    {"name": "sd-master-abc-bin-Linux-Ubuntu-24.04-x86_64.zip", "browser_download_url": "https://x/cpu.zip"},
]


async def _collect(**kw) -> list[dict]:
    return [event async for event in sdcpp_installer.install_stream(**kw)]


def test_is_installed_needs_the_marker_and_the_binary(tmp_path):
    assert sdcpp_installer.is_installed() is False
    (tmp_path / "stable-diffusion.cpp").mkdir()
    (tmp_path / "stable-diffusion.cpp" / "sd-server").write_text("")
    assert sdcpp_installer.is_installed() is False  # no marker yet — extraction may still be running
    (tmp_path / "stable-diffusion.cpp" / ".install_complete").write_text("")
    assert sdcpp_installer.is_installed() is True


@pytest.mark.asyncio
async def test_find_asset_picks_the_plain_cpu_linux_zip(monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS, b"")
    assert await sdcpp_installer._find_asset_url("o/r", "v1", None) == "https://x/cpu.zip"


@pytest.mark.asyncio
@pytest.mark.parametrize("build,expected", [("vulkan", "https://x/vulkan.zip"), ("cpu", "https://x/cpu.zip")])
async def test_find_asset_matches_the_requested_build(monkeypatch, build, expected):
    _install_fake_github(monkeypatch, _ASSETS, b"")
    assert await sdcpp_installer._find_asset_url("o/r", "v1", None, build) == expected


@pytest.mark.asyncio
async def test_find_asset_fails_when_the_requested_gpu_build_is_missing(monkeypatch):
    _install_fake_github(monkeypatch, [_ASSETS[1], _ASSETS[2]], b"")
    with pytest.raises(RuntimeError, match="vulkan build"):
        await sdcpp_installer._find_asset_url("o/r", "v1", None, "vulkan")


@pytest.mark.parametrize(
    "gpu,loader,expected", [(True, "libvulkan.so.1", "vulkan"), (True, None, "cpu"), (False, "libvulkan.so.1", "cpu")]
)
def test_auto_build_needs_both_a_gpu_and_the_vulkan_loader(monkeypatch, gpu, loader, expected):
    monkeypatch.setattr(sdcpp_installer, "_has_gpu", lambda: gpu)
    monkeypatch.setattr(sdcpp_installer.ctypes.util, "find_library", lambda _name: loader)
    assert sdcpp_installer.resolve_build("auto") == expected


def test_an_explicit_build_is_never_overridden_by_detection(monkeypatch):
    monkeypatch.setattr(sdcpp_installer, "_has_gpu", lambda: False)
    assert sdcpp_installer.resolve_build("rocm") == "rocm"


@pytest.mark.asyncio
async def test_install_stream_installs_the_chosen_build_and_reports_it(monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS, _zip_bytes({"sd-server": b"bin"}))
    events = await _collect(build="vulkan")
    assert events[-1]["build"] == "vulkan"
    assert "vulkan build" in events[0]["status"]


@pytest.mark.asyncio
async def test_find_asset_fails_when_there_is_no_linux_cpu_build(monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS[:2], b"")
    with pytest.raises(RuntimeError, match="no Linux x86_64 cpu build"):
        await sdcpp_installer._find_asset_url("o/r", "v1", None)


@pytest.mark.asyncio
async def test_find_asset_rejects_a_non_x86_cpu(monkeypatch):
    monkeypatch.setattr(sdcpp_installer.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="architecture"):
        await sdcpp_installer._find_asset_url("o/r", "v1", None)


@pytest.mark.asyncio
async def test_install_stream_blocks_on_an_unsupported_os(monkeypatch):
    monkeypatch.setattr(sdcpp_installer, "local_install_supported", lambda: False)
    events = await _collect()
    assert len(events) == 1 and "Linux" in events[0]["error"]


@pytest.mark.asyncio
async def test_install_stream_downloads_extracts_and_marks_installed(tmp_path, monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS, _zip_bytes({"sd-server": b"bin", "libggml.so": b"lib"}))

    events = await _collect()

    done = events[-1]
    assert done["done"] is True
    assert done["binary_path"] == str(tmp_path / "stable-diffusion.cpp" / "sd-server")
    assert any(e.get("completed") and e.get("unit") == "bytes" for e in events)
    assert sdcpp_installer.is_installed() is True
    assert (tmp_path / "stable-diffusion.cpp" / "sd-server").stat().st_mode & 0o111
    assert (tmp_path / "stable-diffusion.cpp" / "models").is_dir()
    assert not (tmp_path / "stable-diffusion.cpp" / "sd.zip").exists()


@pytest.mark.asyncio
async def test_reinstall_preserves_downloaded_models_and_replaces_old_files(tmp_path, monkeypatch):
    target = tmp_path / "stable-diffusion.cpp"
    (target / "models").mkdir(parents=True)
    (target / "models" / "keep.gguf").write_text("weights")
    (target / "stale.so").write_text("old")
    _install_fake_github(monkeypatch, _ASSETS, _zip_bytes({"sd-server": b"bin"}))

    assert (await _collect())[-1]["done"] is True

    assert (target / "models" / "keep.gguf").read_text() == "weights"
    assert not (target / "stale.so").exists()


@pytest.mark.asyncio
async def test_a_download_reinstall_keeps_a_source_builds_clone_and_build_tree(tmp_path, monkeypatch):
    target = tmp_path / "stable-diffusion.cpp"
    for kept in ("src", "build"):
        (target / kept).mkdir(parents=True)
        (target / kept / "file.txt").write_text(kept)
    _install_fake_github(monkeypatch, _ASSETS, _zip_bytes({"sd-server": b"bin"}))

    assert (await _collect())[-1]["done"] is True

    assert (target / "src" / "file.txt").read_text() == "src"
    assert (target / "build" / "file.txt").read_text() == "build"


@pytest.mark.asyncio
async def test_install_stream_reports_a_corrupt_archive(monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS, b"not a zip")
    events = await _collect()
    assert "extract" in events[-1]["error"].lower()
    assert sdcpp_installer.is_installed() is False


@pytest.mark.asyncio
async def test_install_stream_reports_an_archive_without_sd_server(monkeypatch):
    _install_fake_github(monkeypatch, _ASSETS, _zip_bytes({"readme.txt": b"hi"}))
    assert "sd-server" in (await _collect())[-1]["error"]


@pytest.mark.asyncio
async def test_install_stream_reports_a_download_failure(monkeypatch):
    def handler(_request):
        return httpx.Response(404)

    monkeypatch.setattr(
        sdcpp_installer.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)
    )
    assert "Could not download" in (await _collect())[-1]["error"]
