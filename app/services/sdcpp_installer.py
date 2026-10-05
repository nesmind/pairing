"""Installs stable-diffusion.cpp from GitHub — the "local" auto-install behind Settings > Image's Install button.
Downloads the pinned release's prebuilt Linux x86_64 CPU zip (`sd-server` + its shared libs, which include per-CPU
ggml backends down to SSE4.2 — no AVX2 requirement, unlike ComfyUI's PyTorch) and unzips it into EXTERNAL_DIR.

A reinstall preserves target_dir's own "models" subdirectory (same reasoning as comfyui_installer): downloaded
models are the admin's library, not disposable engine artifacts."""

import ctypes.util
import logging
import platform
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from app.config import EXTERNAL_DIR, SDCPP_GITHUB_REPO, SDCPP_PINNED_VERSION
from app.hardware import local_install_supported
from app.schemas import SdCppBuild

logger = logging.getLogger("llama_chat")

_TOTAL_STEPS = 2
_STEP_DOWNLOAD = 1
_STEP_EXTRACT = 2
_DOWNLOAD_TIMEOUT = httpx.Timeout(None, connect=10.0)
_API_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def install_dir() -> Path:
    return EXTERNAL_DIR / "stable-diffusion.cpp"


def binary_path() -> Path:
    return install_dir() / "sd-server"


def models_dir() -> Path:
    return install_dir() / "models"


def _marker_path() -> Path:
    return install_dir() / ".install_complete"


def is_installed() -> bool:
    """Marker-gated (written only after extraction) so a half-finished install isn't reported as installed."""
    return _marker_path().exists() and binary_path().exists()


def _has_gpu() -> bool:
    """Any GPU at all (NVIDIA, AMD or Intel): a DRM render node, or a working nvidia-smi."""
    if any(Path("/dev/dri").glob("renderD*")):
        return True
    try:
        subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.SubprocessError):
        return False
    return True


def resolve_build(build: SdCppBuild = "auto") -> str:
    """ "auto" -> "vulkan" when a GPU and the Vulkan loader are both present (the Vulkan build works on NVIDIA, AMD
    and Intel and still falls back to CPU if the device is unusable), else "cpu". An explicit choice is kept."""
    if build != "auto":
        return build
    return "vulkan" if _has_gpu() and ctypes.util.find_library("vulkan") else "cpu"


def _matches(name: str, build: str) -> bool:
    if "Linux" not in name or not name.endswith(".zip"):
        return False
    if build == "cpu":
        return name.endswith("x86_64.zip")  # the plain build — the others carry a -vulkan/-rocm suffix
    return build in name


async def _find_asset_url(repo: str, version: str, proxy_url: str | None, build: str = "cpu") -> str:
    """The Linux x86_64 zip of `build` ("cpu"/"vulkan"/"rocm") in release `version` (asset names embed the commit
    hash and distro version, so they're looked up from the release, not constructed). RuntimeError if none."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError(
            f"No prebuilt stable-diffusion.cpp release for this CPU architecture ({platform.machine()})."
        )
    async with httpx.AsyncClient(timeout=_API_TIMEOUT, proxy=proxy_url) as client:
        resp = await client.get(
            f"https://api.github.com/repos/{repo}/releases/tags/{version}",
            headers={"Accept": "application/vnd.github+json"},
        )
        resp.raise_for_status()
    for asset in resp.json().get("assets", []):
        if _matches(asset["name"], build):
            return asset["browser_download_url"]
    raise RuntimeError(f"Release {version} of {repo} has no Linux x86_64 {build} build.")


def _preserve_models_dir(target_dir: Path) -> Path | None:
    models = target_dir / "models"
    if not models.is_dir():
        return None
    preserved = Path(tempfile.mkdtemp(prefix="sdcpp-models-")) / "models"
    shutil.move(str(models), str(preserved))
    return preserved


def _restore_models_dir(preserved: Path | None, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if preserved is None:
        (target_dir / "models").mkdir(exist_ok=True)
        return
    shutil.move(str(preserved), str(target_dir / "models"))
    preserved.parent.rmdir()


async def install_stream(
    repo: str | None = None,
    version: str | None = None,
    proxy_url: str | None = None,
    build: SdCppBuild = "auto",
) -> AsyncIterator[dict]:
    """Same event shape as the other installers ({"status"...}, then {"done": True, ...} or {"error": ...})."""
    repo = repo or SDCPP_GITHUB_REPO
    version = version or SDCPP_PINNED_VERSION
    build = resolve_build(build)

    if not local_install_supported():
        yield {
            "error": (
                f"Auto-install isn't available on this OS ({platform.system()}) — only Linux. Install "
                'stable-diffusion.cpp yourself, then use "Enter its paths manually", or use Remote mode.'
            )
        }
        return

    download_meta = {"step": _STEP_DOWNLOAD, "total_steps": _TOTAL_STEPS, "step_label": "Downloading"}
    yield {**download_meta, "status": f"Downloading stable-diffusion.cpp {version} ({build} build)..."}
    target_dir = install_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "sd.zip"
    try:
        url = await _find_asset_url(repo, version, proxy_url, build)
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                completed = 0
                last_pct = -1  # report whole-percent changes only — see ollama_installer for why
                with open(archive_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        completed += len(chunk)
                        if total and completed * 100 // total != last_pct:
                            last_pct = completed * 100 // total
                            yield {
                                **download_meta,
                                "status": "Downloading",
                                "completed": completed,
                                "total": total,
                                "unit": "bytes",
                            }
    except (httpx.HTTPError, RuntimeError) as exc:
        archive_path.unlink(missing_ok=True)
        yield {"error": f"Could not download stable-diffusion.cpp: {exc}"}
        return

    extract_meta = {"step": _STEP_EXTRACT, "total_steps": _TOTAL_STEPS, "step_label": "Extracting"}
    yield {**extract_meta, "status": "Extracting..."}
    preserved = _preserve_models_dir(target_dir)
    try:
        for child in target_dir.iterdir():
            if child == archive_path:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(target_dir)
    except (zipfile.BadZipFile, OSError) as exc:
        yield {"error": f"Failed to extract the downloaded archive: {exc}"}
        return
    finally:
        archive_path.unlink(missing_ok=True)
        _restore_models_dir(preserved, target_dir)

    if not binary_path().exists():
        yield {"error": "Extraction finished but the sd-server binary wasn't found where expected."}
        return
    for name in ("sd-server", "sd-cli"):
        if (target_dir / name).exists():
            (target_dir / name).chmod(0o755)
    _marker_path().write_text("")
    logger.info("Installed stable-diffusion.cpp %s to %s", version, target_dir)
    yield {
        "done": True,
        "step": _TOTAL_STEPS,
        "total_steps": _TOTAL_STEPS,
        "binary_path": str(binary_path()),
        "build": build,
        "models_dir": str(models_dir()),
    }
