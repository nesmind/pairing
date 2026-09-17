"""
Installs ComfyUI from GitHub — the "local" mode auto-install behind
Settings > External servers' Install button. `git clone`s the exact
pinned app.config.COMFYUI_PINNED_VERSION tag (never "latest", for the
same predictability reason app.services.ollama_installer pins Ollama's
version), then builds a dedicated venv and `pip install`s ComfyUI's own
requirements.txt — there's no separate native "compile" step beyond
whatever pip itself does for any wheel that needs building.

Requires `git` and a working `python3 -m venv` on this machine — neither
is something this app can install for the admin, so both failures
surface as a clear {"error": ...} rather than a raw crash.
"""

import asyncio
import logging
import os
import platform
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import COMFYUI_GITHUB_REPO, COMFYUI_PINNED_VERSION, EXTERNAL_DIR
from app.hardware import get_gpu_vram_gb, has_avx2, local_install_supported

logger = logging.getLogger("llama_chat")

# See app.services.ollama_installer's identical constants — every event
# below carries step/total_steps/step_label so Settings > External
# servers can render a labeled "Step X of Y" progress panel instead of a
# bare spinner (see app/static/js/settings.js's installServer).
_TOTAL_STEPS = 3
_STEP_CLONE = 1
_STEP_VENV = 2
_STEP_PIP = 3


def install_dir() -> Path:
    return EXTERNAL_DIR / "ComfyUI"


def main_py_path() -> Path:
    return install_dir() / "main.py"


def python_path() -> Path:
    return install_dir() / "venv" / "bin" / "python"


def _marker_path() -> Path:
    return install_dir() / ".install_complete"


def is_installed() -> bool:
    """True once a previous Install click finished successfully — unlike
    Ollama, this module's managed path is the *only* place a "local"
    ComfyUI install is ever auto-detected from (an admin pointing at
    their own separately-installed ComfyUI instead just fills in the
    python_path/main_py_path fields directly — see
    app.schemas.comfyui_config.ComfyUIProcessConfig).

    Checks a marker file written only once install_stream's pip step
    finishes, not just main.py/venv existing — both of those already
    exist partway through a still-running install (git clone and venv
    creation both finish long before the pip install of requirements.txt
    does), so checking them alone would report "installed" while the
    install is still actively running — confirmed live: reloading
    Settings mid-install showed the finished-state params panel instead
    of the in-progress one."""
    return _marker_path().exists() and main_py_path().exists() and python_path().exists()


async def _run_streamed(
    argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> AsyncIterator[str]:
    """Runs `argv`, yielding its combined stdout/stderr line by line as
    it's produced (not buffered until exit) — both `git clone --progress`
    and `pip install` report real progress this way, and the caller
    relays each line straight into an SSE event. `env` defaults to the
    same "inherit this app's own process env" behavior as before this
    param existed — pass it (see _proxied_env below) only for a call that
    actually makes a network request.

    The `finally` below matters more than it looks: confirmed live that
    without it, a client disconnecting mid-install (a closed tab, or a
    reload) tears down *this* async generator (Python raises GeneratorExit
    at the suspended `yield`) but does nothing to the real OS subprocess
    it spawned — pip/git kept running fully orphaned, invisible to the
    app (no tracking, no way to know it finished, no marker file ever
    written since the coroutine that would write it is gone), for as
    long as it took to finish on its own. Killing it here instead makes
    a disconnect actually stop the install, matching what a plain
    process-name-based "start/stop" toggle already gives Ollama/ComfyUI
    once installed."""
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
    """None when no proxy is configured (the caller then leaves `env`
    unset, inheriting this app's own process env unchanged). Sets both
    the upper- and lower-case forms: git/curl and some pip backends only
    honor the lowercase convention, most Python libraries only the
    uppercase one — setting both removes any ambiguity about which the
    underlying tool happens to check."""
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
    """Yields progress events the same shape
    app.services.ollama_installer.install_stream already does
    ({"status": ...}, finally {"done": True, ...} or {"error": ...}) —
    reused as-is by the SSE relay in app/routers/comfyui_admin.py.
    Cleans up a partial clone on failure so a retry starts fresh rather
    than hitting "directory already exists".

    `repo`/`version` let the caller (see app/routers/comfyui_admin.py's
    install endpoint, which passes through ComfyUIProcessConfig.install_repo/
    install_version) pin a different fork/tag than the maintainer's own
    COMFYUI_GITHUB_REPO/COMFYUI_PINNED_VERSION default — None for either
    falls back to that default. `proxy_url` (from Settings > System's
    HttpProxyConfig, see app.services.http_proxy_service) routes the git
    clone and pip install through an admin-configured HTTP proxy — the
    venv-creation step in between doesn't need it, since it makes no
    network requests of its own."""
    repo = repo or COMFYUI_GITHUB_REPO
    version = version or COMFYUI_PINNED_VERSION
    proxied_env = _proxied_env(proxy_url)

    if not local_install_supported():
        yield {
            "error": (
                f"Auto-install isn't available on this OS ({platform.system()}) — this app's local process "
                "management (and its own deployment scripts) only target Linux. Install ComfyUI yourself, then "
                'use "Enter its paths manually" above, or switch this to Remote mode and point it at a ComfyUI '
                "server running elsewhere."
            )
        }
        return

    if not has_avx2():
        yield {
            "error": (
                "This machine's CPU doesn't support AVX2, an instruction set the prebuilt PyTorch/"
                "NumPy packages ComfyUI depends on require — installing anyway would download several "
                "GB and then crash immediately once ComfyUI actually starts (a hard 'Illegal "
                "instruction' crash, not something this app can work around or catch beforehand any "
                "other way). This needs a newer CPU — effectively any x86_64 machine from roughly the "
                "last 10 years."
            )
        }
        return

    if shutil.which("git") is None:
        yield {"error": "'git' was not found on PATH — required to download ComfyUI."}
        return

    target_dir = install_dir()
    if target_dir.exists():
        shutil.rmtree(target_dir)

    clone_meta = {"step": _STEP_CLONE, "total_steps": _TOTAL_STEPS, "step_label": "Cloning"}
    yield {**clone_meta, "status": f"Cloning ComfyUI {version}..."}
    try:
        repo_url = f"https://github.com/{repo}.git"
        argv = ["git", "clone", "--branch", version, "--depth", "1", repo_url, str(target_dir)]
        async for line in _run_streamed(argv, env=proxied_env):
            yield {**clone_meta, "status": line}
    except RuntimeError as exc:
        shutil.rmtree(target_dir, ignore_errors=True)
        yield {"error": f"Could not clone ComfyUI: {exc}"}
        return

    venv_meta = {"step": _STEP_VENV, "total_steps": _TOTAL_STEPS, "step_label": "Creating a Python environment"}
    yield {**venv_meta, "status": "Creating a Python environment..."}
    try:
        async for line in _run_streamed([sys.executable, "-m", "venv", str(target_dir / "venv")]):
            yield {**venv_meta, "status": line}
    except RuntimeError as exc:
        yield {"error": f"Could not create ComfyUI's virtual environment: {exc}"}
        return

    pip_meta = {"step": _STEP_PIP, "total_steps": _TOTAL_STEPS, "step_label": "Installing dependencies"}
    yield {**pip_meta, "status": "Installing dependencies (this can take a while)..."}
    try:
        async for line in _run_streamed(
            [str(python_path()), "-m", "pip", "install", "-r", str(target_dir / "requirements.txt")],
            env=proxied_env,
        ):
            yield {**pip_meta, "status": line}
    except RuntimeError as exc:
        yield {"error": f"Could not install ComfyUI's dependencies: {exc}"}
        return

    _marker_path().write_text("")
    logger.info("Installed ComfyUI %s to %s", version, target_dir)
    # ComfyUI's own requirements.txt pulls a CUDA-enabled PyTorch build
    # unconditionally — on a machine with no NVIDIA GPU/driver, that
    # crashes hard at startup trying to initialize CUDA (confirmed live:
    # "RuntimeError: Found no NVIDIA driver on your system") unless
    # launched with ComfyUI's own --cpu flag. Detected the same way
    # app.hardware already gates model recommendations elsewhere, and
    # suggested as the default extra_args so a fresh install actually
    # starts on the first Start click instead of crashing.
    extra_args = "--cpu" if get_gpu_vram_gb() == 0.0 else None
    yield {
        "done": True,
        "step": _TOTAL_STEPS,
        "total_steps": _TOTAL_STEPS,
        "python_path": str(python_path()),
        "main_py_path": str(main_py_path()),
        "extra_args": extra_args,
    }
