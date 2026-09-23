"""
Installs Matricxon from GitHub — the "local" mode auto-install behind Settings > External servers' Install
button, the Matricxon-flavored twin of app.services.comfyui_installer (see that module's own docstring for the
full reasoning behind each step; unchanged here except where Matricxon's own project shape differs). `git
clone`s the pinned app.config.MATRICXON_PINNED_VERSION tag (never a moving branch — same predictability
reasoning as Ollama/ComfyUI's own pinned installs), then builds a `.venv` (not ComfyUI's `venv` — Matricxon's
own scripts/start.sh hardcodes `.venv/bin/uvicorn`, confirmed against that script directly) and `pip install`s
its runtime requirements.txt (not requirements-dev.txt — the README's own dev Setup instructions install dev
tooling too, but an admin running this as a service has no use for pytest/ruff).

Requires `git` and a working `python3 -m venv` on this machine, same as comfyui_installer.

Deliberately has no AVX2 gate the way comfyui_installer does — confirmed live (2026-09-21, a real Sandy Bridge
i7-2640M with no AVX2 at all — /proc/cpuinfo's own flags line has no "avx2" entry) that this doesn't apply to
Matricxon's own dependency set: current torch/numpy prebuilt wheels (torch 2.14, numpy 2.5 as resolved by
requirements.txt today) both do runtime CPU dispatch — numpy's own np.show_config() reports its OpenBLAS build
as "DYNAMIC_ARCH", and a real POST /api/chat against that exact machine's already-installed Matricxon completed
with a clean 200, not a crash. ComfyUI's own gate is left untouched since it depends on more/different
torch-adjacent packages (kornia and others) with no equivalent evidence either way.

Installs into app.config.EXTERNAL_DIR (the same "keep an admin-managed checkout separate from a dev's own
project files" convention Ollama/ComfyUI already use), then saves that path into
MatricxonServerConfig.project_dir once done — app.services.matricxon_process._find_project_dir already treats an
explicit project_dir as an override ahead of its own sibling-directory default, so this needs no dedicated
marker file or new detection logic (unlike comfyui_installer's _marker_path, which exists specifically because
ComfyUI has no such config-driven override to lean on).

A reinstall preserves target_dir's own "data" subdirectory (see _preserve_data_dir/_restore_data_dir) — the
already-downloaded models living there are pAIring's own managed library (pulled through its Model tab, tracked
the same way regardless of which engine happens to store the bytes), not disposable engine-install artifacts,
so wiping them out from under an admin just because the *engine software* got reinstalled would be wrong in the
same way deleting Ollama's own pulled models on an Ollama reinstall would be — Ollama never has this problem
only because its models already live in a wholly separate location this installer never touches.
"""

import asyncio
import logging
import os
import platform
import shutil
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import EXTERNAL_DIR, MATRICXON_GITHUB_REPO, MATRICXON_PINNED_VERSION
from app.hardware import local_install_supported
from app.services import matricxon_process
from app.services.pip_progress import PipProgressTracker

logger = logging.getLogger("llama_chat")

# See app.services.ollama_installer/comfyui_installer's identical constants — every event below carries
# step/total_steps/step_label so Settings > External servers can render a labeled "Step X of Y" progress panel.
_TOTAL_STEPS = 3
_STEP_CLONE = 1
_STEP_VENV = 2
_STEP_PIP = 3


def install_dir() -> Path:
    return EXTERNAL_DIR / "matricxon"


def _preserve_data_dir(target_dir: Path) -> Path | None:
    """Moves target_dir's own "data" subdirectory (Matricxon's real Settings.models_dir default,
    "./data/models" relative to its own process cwd — which scripts/start.sh sets to exactly this directory) out
    to a fresh temp directory, returning where it went (None if there was nothing to preserve). Called right
    before the reinstall below wipes target_dir outright — a real, confirmed-live bug without this: every
    already-downloaded model (potentially several GB, hours of pulling) was silently destroyed by a reinstall,
    unlike Ollama's own installer, whose models live in a completely separate location (BASE_DIR/models, see
    app.services.ollama_process._build_env's OLLAMA_MODELS) that nothing here ever touches."""
    data_dir = target_dir / "data"
    if not data_dir.is_dir():
        return None
    holding_dir = Path(tempfile.mkdtemp(prefix="matricxon-data-"))
    preserved = holding_dir / "data"
    shutil.move(str(data_dir), str(preserved))
    return preserved


def _restore_data_dir(preserved: Path | None, target_dir: Path) -> None:
    if preserved is None:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(preserved), str(target_dir / "data"))
    preserved.parent.rmdir()  # the now-empty tempfile.mkdtemp() holding directory


def is_installed() -> bool:
    return matricxon_process.is_installed()


async def _run_streamed(
    argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> AsyncIterator[str]:
    """See comfyui_installer._run_streamed's identical docstring — same real-time-progress and
    disconnect-kills-the-subprocess reasoning, unchanged here."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    try:
        async for raw_line in process.stdout:
            line = raw_line.decode(errors="replace").rstrip()
            if line:
                yield line
        returncode = await process.wait()
        if returncode != 0:
            raise RuntimeError(f"{argv[0]} exited with code {returncode}")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def _proxied_env(proxy_url: str | None) -> dict[str, str] | None:
    """See comfyui_installer._proxied_env's identical docstring."""
    if not proxy_url:
        return None
    return {
        **os.environ,
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
    }


async def install_stream(
    repo: str | None = None, version: str | None = None, proxy_url: str | None = None
) -> AsyncIterator[dict]:
    """Yields progress events the same shape app.services.ollama_installer/comfyui_installer's own install_stream
    already does ({"status": ...}, finally {"done": True, ...} or {"error": ...}) — reused as-is by the SSE relay
    in app/routers/matricxon_admin.py. Cleans up a partial clone on failure so a retry starts fresh.

    `repo`/`version` let the caller (MatricxonServerConfig.install_repo/install_version) pin a different
    fork/tag than MATRICXON_GITHUB_REPO/MATRICXON_PINNED_VERSION's own default — None for either falls back to
    that default. `proxy_url` routes the git clone and pip install through an admin-configured HTTP proxy, same
    as comfyui_installer."""
    repo = repo or MATRICXON_GITHUB_REPO
    version = version or MATRICXON_PINNED_VERSION
    proxied_env = _proxied_env(proxy_url)

    if not repo or not version:
        yield {
            "error": (
                "No Matricxon repo/version configured — set app.config.MATRICXON_GITHUB_REPO/"
                "MATRICXON_PINNED_VERSION, or pin a specific fork/tag below."
            )
        }
        return

    if not local_install_supported():
        yield {
            "error": (
                f"Auto-install isn't available on this OS ({platform.system()}) — this app's local process "
                "management (and its own deployment scripts) only target Linux. Install Matricxon yourself, "
                'then set "Project directory" above to that checkout, or switch this to Remote mode and point '
                "it at a Matricxon server running elsewhere."
            )
        }
        return

    if shutil.which("git") is None:
        yield {"error": "'git' was not found on PATH — required to download Matricxon."}
        return

    target_dir = install_dir()
    preserved_data_dir = _preserve_data_dir(target_dir)
    if target_dir.exists():
        shutil.rmtree(target_dir)

    clone_meta = {"step": _STEP_CLONE, "total_steps": _TOTAL_STEPS, "step_label": "Cloning"}
    yield {**clone_meta, "status": f"Cloning Matricxon {version}..."}
    try:
        repo_url = f"https://github.com/{repo}.git"
        argv = ["git", "clone", "--branch", version, "--depth", "1", repo_url, str(target_dir)]
        async for line in _run_streamed(argv, env=proxied_env):
            yield {**clone_meta, "status": line}
    except RuntimeError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        _restore_data_dir(preserved_data_dir, target_dir)
        yield {"error": f"Could not clone Matricxon: {exc}"}
        return

    # Restored right away, not held until the very end — venv creation/pip install below never touch "data" at
    # all, so there's no reason to leave already-downloaded models sitting in a temp directory (at real risk of
    # being stranded there) for the several more minutes those two steps can take.
    _restore_data_dir(preserved_data_dir, target_dir)

    venv_dir = target_dir / ".venv"
    venv_meta = {"step": _STEP_VENV, "total_steps": _TOTAL_STEPS, "step_label": "Creating a Python environment"}
    yield {**venv_meta, "status": "Creating a Python environment..."}
    try:
        async for line in _run_streamed([sys.executable, "-m", "venv", str(venv_dir)]):
            yield {**venv_meta, "status": line}
    except RuntimeError as exc:
        yield {"error": f"Could not create Matricxon's virtual environment: {exc}"}
        return

    # step_label (not just the one-off "status" log line below) carries the "this can take a while" caveat —
    # it's set from every event this step yields (see app/static/js/settings.js's installServer), so it stays
    # visible in the persistent step header for the step's whole duration, not just briefly in the scrolling
    # log. Matters because pip's own piped (non-TTY) output has no incremental byte-progress for a download the
    # way git clone's own --progress does (confirmed live: plain "Downloading torch-...whl (899.7 MB)" then
    # nothing until it's done) — this step's progress bar stays indeterminate the whole time with no percentage
    # to show, which reads as "stuck" without an explicit reassurance that multi-minute silence here is normal.
    pip_meta = {
        "step": _STEP_PIP,
        "total_steps": _TOTAL_STEPS,
        "step_label": "Installing dependencies — this can take several minutes (PyTorch is a large download)",
    }
    yield {**pip_meta, "status": "Installing dependencies (this can take a while)..."}
    venv_python = venv_dir / "bin" / "python"
    # Real per-download percentages instead of minutes of silence - see app.services.pip_progress.
    progress = PipProgressTracker("Installing dependencies")
    raw_progress = ["--progress-bar", "raw"] if await PipProgressTracker.supports_raw(venv_python) else []
    try:
        async for line in _run_streamed(
            [str(venv_python), "-m", "pip", "install", *raw_progress, "-r", str(target_dir / "requirements.txt")],
            env=proxied_env,
        ):
            yield {**pip_meta, **progress.event_for(line)}
    except RuntimeError as exc:
        yield {"error": f"Could not install Matricxon's dependencies: {exc}"}
        return

    logger.info("Installed Matricxon %s to %s", version, target_dir)
    yield {"done": True, "step": _TOTAL_STEPS, "total_steps": _TOTAL_STEPS, "project_dir": str(target_dir)}
