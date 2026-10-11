"""Installs ComfyUI from GitHub (the "local" auto-install behind Settings > External servers' Install). Clones
the pinned app.config.COMFYUI_PINNED_VERSION tag (never "latest", like ollama_installer), builds a venv and
`pip install`s ComfyUI's requirements.txt.

Needs `git` and `python3 -m venv`; failures surface as a clear {"error":...}.

A reinstall preserves target_dir's "models" subdirectory (_preserve_models_dir/_restore_models_dir):
checkpoints are pAIring's managed library, not disposable install artifacts (as in matricxon_installer's "data").
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

from app.config import COMFYUI_GITHUB_REPO, COMFYUI_PINNED_VERSION, EXTERNAL_DIR
from app.hardware import get_gpu_vram_gb, has_avx2, local_install_supported

logger = logging.getLogger("llama_chat")

# See ollama_installer's constants: every event carries step/total_steps/step_label for the labeled
# "Step X of Y" progress panel (static/js/settings.js installServer).
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


def _preserve_models_dir(target_dir: Path) -> Path | None:
    """Moves target_dir's "models" subdirectory (ComfyUI's checkpoints/loras/vae; no extra_model_paths.yaml) to a
    temp directory and returns it (None if nothing to preserve). Without this a reinstall destroyed every pulled
    checkpoint (see matricxon_installer._preserve_data_dir).
    """
    models_dir = target_dir / "models"
    if not models_dir.is_dir():
        return None
    holding_dir = Path(tempfile.mkdtemp(prefix="comfyui-models-"))
    preserved = holding_dir / "models"
    shutil.move(str(models_dir), str(preserved))
    return preserved


def _restore_models_dir(preserved: Path | None, target_dir: Path) -> None:
    if preserved is None:
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(preserved), str(target_dir / "models"))
    preserved.parent.rmdir()  # the now-empty mkdtemp holding directory


def _marker_path() -> Path:
    return install_dir() / ".install_complete"


def is_installed() -> bool:
    """True once a previous Install finished. This managed path is the only place a local ComfyUI is
    auto-detected (own installs fill python_path/main_py_path, see ComfyUIProcessConfig).

    Checks a marker written after the pip step: main.py and the venv exist long before pip finishes, so they
    would report "installed" mid-install (reloading Settings then showed the finished-state panel).
    """
    return _marker_path().exists() and main_py_path().exists() and python_path().exists()


async def _run_streamed(
    argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> AsyncIterator[str]:
    """Runs `argv`, yielding combined stdout/stderr line by line as produced (`git clone --progress` and
    `pip install` report real progress), which the caller relays as SSE. `env` defaults to this process's
    env; pass it (see _proxied_env) only for network calls.

    The `finally` matters: without it a client disconnect mid-install tore down this generator
    (GeneratorExit at the `yield`) but left the pip/git subprocess running orphaned and untracked, with no
    marker file. Killing it makes a disconnect actually stop the install.
    """
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
    """None when no proxy is configured (env stays inherited). Sets upper- and lower-case variables: git/curl
    and some pip backends read only lowercase, most Python libraries only uppercase.
    """
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
    """Yields progress events like ollama_installer.install_stream ({"status":...}, then {"done": True,...}
    or {"error":...}), relayed by routers/comfyui_admin.py. Cleans up a partial clone on failure so a retry
    starts fresh.

    `repo`/`version` (ComfyUIProcessConfig.install_repo/install_version) override COMFYUI_GITHUB_REPO/
    COMFYUI_PINNED_VERSION; None uses the default. `proxy_url` (HttpProxyConfig, http_proxy_service) routes the
    git clone and pip install; venv creation makes no network requests.
    """
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
    preserved_models_dir = _preserve_models_dir(target_dir)
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
        _restore_models_dir(preserved_models_dir, target_dir)
        yield {"error": f"Could not clone ComfyUI: {exc}"}
        return

    # Restored right away: venv creation and pip never touch "models", so no reason to hold checkpoints in a temp
    # directory meanwhile.
    _restore_models_dir(preserved_models_dir, target_dir)

    venv_meta = {"step": _STEP_VENV, "total_steps": _TOTAL_STEPS, "step_label": "Creating a Python environment"}
    yield {**venv_meta, "status": "Creating a Python environment..."}
    try:
        async for line in _run_streamed([sys.executable, "-m", "venv", str(target_dir / "venv")]):
            yield {**venv_meta, "status": line}
    except RuntimeError as exc:
        yield {"error": f"Could not create ComfyUI's virtual environment: {exc}"}
        return

    # step_label carries the "this can take a while" caveat: pip's piped output has no incremental byte
    # progress like git's --progress, so the bar stays indeterminate (see matricxon_installer pip_meta).
    pip_meta = {
        "step": _STEP_PIP,
        "total_steps": _TOTAL_STEPS,
        "step_label": "Installing dependencies — this can take several minutes (PyTorch is a large download)",
    }
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
    # ComfyUI's requirements.txt pulls a CUDA PyTorch build, which crashes at startup without an NVIDIA
    # driver ("Found no NVIDIA driver on your system") unless run with --cpu. Detected via app.hardware and
    # suggested as default extra_args so Start works first time.
    extra_args = "--cpu" if get_gpu_vram_gb() == 0.0 else None
    yield {
        "done": True,
        "step": _TOTAL_STEPS,
        "total_steps": _TOTAL_STEPS,
        "python_path": str(python_path()),
        "main_py_path": str(main_py_path()),
        "extra_args": extra_args,
    }
