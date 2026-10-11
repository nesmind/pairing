"""Downloads and installs Ollama's official prebuilt release from GitHub: the "local" auto-install behind
Settings > External servers' Install. Never builds from source (no Go) and never "latest": always the pinned
app.config.OLLAMA_PINNED_VERSION, for predictable restarts and reinstalls.

Asset shape (checked against the real release): `ollama-linux-{arch}.tar.zst`, a Zstandard tarball with
`bin/ollama` plus `lib/ollama/*` (libraries loaded relative to the binary), not a flat binary. Extracted with
system `tar --zstd` (GNU tar's zstd filter) rather than a new Python dependency; Linux only (scripts/start.sh).
"""

import logging
import platform
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from app.config import EXTERNAL_DIR, OLLAMA_GITHUB_REPO, OLLAMA_PINNED_VERSION
from app.hardware import local_install_supported
from app.services import ollama_process

logger = logging.getLogger("llama_chat")

_DOWNLOAD_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded read: a multi-GB download

# Every event carries step/total_steps/step_label with its status text, rendered by Settings > External servers
# as a labeled "Step X of Y" panel (settings.js installServer) so admins see where a multi-minute install is.
_TOTAL_STEPS = 2
_STEP_DOWNLOAD = 1
_STEP_EXTRACT = 2


def install_dir() -> Path:
    return EXTERNAL_DIR / "ollama"


def binary_path() -> Path:
    return install_dir() / "bin" / "ollama"


def is_installed() -> bool:
    """True if a managed install (previous Install) or any other Ollama binary is reachable; search:
    ollama_process.is_installed
    (PATH, ~/bin, /usr/local/bin, /usr/bin, this module's managed path).
    """
    return ollama_process.is_installed()


def _arch_name() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "amd64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    raise RuntimeError(f"No known Ollama release asset for this machine's architecture ({machine}).")


def _asset_url(repo: str, version: str) -> str:
    return f"https://github.com/{repo}/releases/download/{version}/ollama-linux-{_arch_name()}.tar.zst"


async def install_stream(
    repo: str | None = None, version: str | None = None, proxy_url: str | None = None
) -> AsyncIterator[dict]:
    """Downloads the pinned release asset (streamed to disk, never buffered; multi-GB), extracts it into the managed
    directory and yields progress like ollama_admin.pull_model_stream ({"status":...}, optionally
    "completed"/"total", finally {"done": True} or {"error":...}), relayed by routers/ollama_admin.py.

    `repo`/`version` (OllamaServerConfig.install_repo/install_version via the install endpoint) override
    OLLAMA_GITHUB_REPO/OLLAMA_PINNED_VERSION; None uses the default. `proxy_url` (HttpProxyConfig,
    http_proxy_service) routes the download through the admin's proxy; None downloads directly.

    Needs `tar` with zstd on PATH (GNU tar >= 1.31, ~2019+); yields a clear {"error":...} if missing, as the
    toolchain isn't this app's to control.
    """
    repo = repo or OLLAMA_GITHUB_REPO
    version = version or OLLAMA_PINNED_VERSION

    if not local_install_supported():
        yield {
            "error": (
                f"Auto-install isn't available on this OS ({platform.system()}) — the release asset this "
                'downloads is Linux-only. Install Ollama yourself, then use "Enter its path manually" above, '
                "or switch this to Remote mode and point it at an Ollama server running elsewhere."
            )
        }
        return

    if shutil.which("tar") is None:
        yield {"error": "'tar' was not found on PATH — required to extract the downloaded archive."}
        return

    target_dir = install_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "ollama.tar.zst"

    try:
        url = _asset_url(repo, version)
    except RuntimeError as exc:
        yield {"error": str(exc)}
        return

    download_meta = {"step": _STEP_DOWNLOAD, "total_steps": _TOTAL_STEPS, "step_label": "Downloading"}
    yield {**download_meta, "status": f"Downloading Ollama {version}..."}
    try:
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True, proxy=proxy_url) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                completed = 0
                # Yielding per 1MB chunk (thousands for a multi-GB file) slowed the download itself: each yield is an
                # SSE
                # event the browser renders, and a slow render backpressures this coroutine's httpx read loop. It ran at
                # about
                # half of curl's throughput on the same URL. Reporting only when the whole-percent value changes cuts a
                # ~1.4GB download from ~1400 events to ~100, as informative to a human.
                last_reported_pct = -1
                with open(archive_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        completed += len(chunk)
                        if total:
                            pct = completed * 100 // total
                            if pct != last_reported_pct:
                                last_reported_pct = pct
                                yield {
                                    **download_meta,
                                    "status": "Downloading",
                                    "completed": completed,
                                    "total": total,
                                }
    except httpx.HTTPError as exc:
        yield {"error": f"Could not download Ollama: {exc}"}
        return

    extract_meta = {"step": _STEP_EXTRACT, "total_steps": _TOTAL_STEPS, "step_label": "Extracting"}
    yield {**extract_meta, "status": "Extracting..."}
    result = subprocess.run(
        ["tar", "--zstd", "-xf", str(archive_path), "-C", str(target_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    archive_path.unlink(missing_ok=True)
    if result.returncode != 0:
        yield {"error": f"Failed to extract the downloaded archive: {result.stderr.strip() or 'unknown tar error'}"}
        return

    if not binary_path().exists():
        yield {"error": "Extraction finished but the ollama binary wasn't found where expected."}
        return
    binary_path().chmod(0o755)

    logger.info("Installed Ollama %s to %s", version, target_dir)
    yield {"done": True, "step": _TOTAL_STEPS, "total_steps": _TOTAL_STEPS, "binary_path": str(binary_path())}
