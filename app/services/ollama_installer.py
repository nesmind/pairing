"""
Downloads and installs Ollama's official prebuilt release binary from
GitHub — the "local" mode auto-install behind Settings > External
servers' Install button. Never builds from source (no Go toolchain
needed) and never installs "latest" — always the exact pinned
app.config.OLLAMA_PINNED_VERSION, so behavior stays predictable across
restarts/reinstalls.

Real release asset shape (confirmed against the actual GitHub release,
not assumed): `ollama-linux-{arch}.tar.zst` — a Zstandard-compressed
tarball containing `bin/ollama` plus `lib/ollama/*` (backend libraries
the binary loads relative to its own path), not a single flat binary.
Extracted via the system `tar --zstd` (GNU tar's built-in zstd filter,
confirmed present) rather than a new Python dependency — this app
already only targets Linux deployments (see scripts/start.sh).
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

_DOWNLOAD_TIMEOUT = httpx.Timeout(None, connect=10.0)  # unbounded read — this is a real, multi-GB download

# Every event below carries step/total_steps/step_label alongside its
# status text — Settings > External servers renders these as a labeled
# "Step X of Y" progress panel (see app/static/js/settings.js's
# installServer), not just a bare spinner, so the admin can see exactly
# where a multi-minute install actually is.
_TOTAL_STEPS = 2
_STEP_DOWNLOAD = 1
_STEP_EXTRACT = 2


def install_dir() -> Path:
    return EXTERNAL_DIR / "ollama"


def binary_path() -> Path:
    return install_dir() / "bin" / "ollama"


def is_installed() -> bool:
    """True if either an already-managed install (from a previous
    Install click) or any other Ollama binary is reachable at all — see
    ollama_process.is_installed for the full search (PATH, ~/bin,
    /usr/local/bin, /usr/bin, and this module's own managed path)."""
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
    """Downloads the pinned release asset (streamed to disk — this is a
    multi-GB file, never buffered in memory), extracts it into this
    module's own managed directory, and yields progress events the same
    shape app.services.ollama_admin.pull_model_stream already does
    ({"status": ...}, optionally with "completed"/"total" byte counts,
    finally {"done": True} or {"error": ...}) — reused as-is by the SSE
    relay in app/routers/ollama_admin.py.

    `repo`/`version` let the caller (see app/routers/ollama_admin.py's
    install endpoint, which passes through OllamaServerConfig.install_repo/
    install_version) pin a different fork/release than the maintainer's
    own OLLAMA_GITHUB_REPO/OLLAMA_PINNED_VERSION default — None for
    either falls back to that default, same "admin override, else a
    sane built-in" pattern the rest of this config already uses.
    `proxy_url` (from Settings > System's HttpProxyConfig, see
    app.services.http_proxy_service) routes this download through an
    admin-configured HTTP proxy — None means download directly.

    Requires `tar` with zstd support on PATH (standard on any GNU
    tar >= 1.31, i.e. any Linux distro from ~2019 on) — yields a clear
    {"error": ...} rather than a raw crash if it's missing, since this
    machine's toolchain isn't something this app controls."""
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
                # Yielding on every 1MB chunk (a real, multi-GB file is
                # thousands of them) turned out to genuinely slow the
                # download down, not just the UI: each yield is an SSE
                # event the browser has to render (a DOM update to the
                # progress bar/log), and a slow-to-render browser creates
                # real TCP backpressure on this same coroutine's own
                # httpx read loop — confirmed live, this app's own
                # download ran at roughly half the raw throughput a
                # plain curl got against the identical URL. Reporting
                # only when the whole-percent value actually changes cuts
                # a ~1.4GB download's event count from ~1400 to ~100
                # while staying exactly as informative to a human eye.
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
    yield {"done": True, "step": _TOTAL_STEPS, "total_steps": _TOTAL_STEPS}
