"""Installs Matricxon from GitHub (the "local" auto-install behind Settings > Inference engines' Install), in
steps (each a "Step X of Y" event). `git clone`s app.config.MATRICXON_DEFAULT_VERSION (Matricxon's main branch,
its newest code, unlike Ollama's pinned tag) unless the admin chose a tag/branch, then builds a `.venv`
(Matricxon's scripts/start.sh hardcodes `.venv/bin/uvicorn`) and `pip install`s requirements.txt (not
requirements-dev.txt; a service needs no pytest/ruff).

Needs `git` and `python3 -m venv`.

No AVX2 gate: on a Sandy Bridge CPU without AVX2, Matricxon's dependencies are fine. Current torch 2.14 /
numpy 2.5 wheels dispatch CPU features at runtime (numpy's OpenBLAS is "DYNAMIC_ARCH"), and a POST /api/chat on
that machine returned 200.

Installs into app.config.EXTERNAL_DIR (admin checkouts kept apart from a dev's files, like Ollama),
then saves the path into MatricxonServerConfig.project_dir. matricxon_process._find_project_dir already
prefers an explicit project_dir over its sibling-directory default, so no marker file or new detection is
needed.

A reinstall preserves target_dir's "data" subdirectory (_preserve_data_dir/_restore_data_dir): the
downloaded models there are pAIring's managed library, not disposable install artifacts. Ollama avoids this
because its models live in a separate location the installer never touches.
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

from app.config import EXTERNAL_DIR, MATRICXON_DEFAULT_VERSION, MATRICXON_GITHUB_REPO
from app.hardware import local_install_supported
from app.services import matricxon_process
from app.services.pip_progress import PipProgressTracker

logger = logging.getLogger("llama_chat")

# As ollama_installer: every event carries step/total_steps/step_label for the "Step X of Y" panel.
_TOTAL_STEPS = 3
_STEP_CLONE = 1
_STEP_VENV = 2
_STEP_PIP = 3


def install_dir() -> Path:
    return EXTERNAL_DIR / "matricxon"


def _preserve_data_dir(target_dir: Path) -> Path | None:
    """Moves target_dir's "data" subdirectory (Matricxon's default models_dir "./data/models", relative to the cwd
    scripts/start.sh sets) to a temp directory and returns it (None if nothing to preserve). Called before
    the reinstall wipes target_dir: without it every downloaded model (several GB, hours of pulling) was
    destroyed, unlike Ollama (models in BASE_DIR/models, see ollama_process._build_env's OLLAMA_MODELS).
    """
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
    preserved.parent.rmdir()  # the now-empty mkdtemp holding directory


def is_installed() -> bool:
    return matricxon_process.is_installed()


async def _run_streamed(
    argv: list[str], cwd: Path | None = None, env: dict[str, str] | None = None
) -> AsyncIterator[str]:
    """Runs `argv`, yielding each output line as it comes (real-time progress); a disconnect kills the subprocess."""
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
    """The environment for git/pip to go through the admin's HTTP proxy, or None (inherit) without one."""
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
    """Yields progress events like ollama_installer ({"status":...}, then {"done": True,...}
    or {"error":...}), relayed by routers/matricxon_admin.py. Cleans up a partial clone on failure.

    `repo`/`version` (MatricxonServerConfig.install_repo/install_version) override MATRICXON_GITHUB_REPO/
    MATRICXON_DEFAULT_VERSION; None uses the default. `proxy_url` routes the git clone and pip install
    through the admin's HTTP proxy.
    """
    repo = repo or MATRICXON_GITHUB_REPO
    version = version or MATRICXON_DEFAULT_VERSION
    proxied_env = _proxied_env(proxy_url)

    if not repo or not version:
        yield {
            "error": (
                "No Matricxon repo/version configured — set app.config.MATRICXON_GITHUB_REPO/"
                "MATRICXON_DEFAULT_VERSION, or pin a specific fork/tag below."
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

    # Restored right away: venv creation and pip never touch "data", so don't leave downloaded models in a temp
    # directory (risking loss) for minutes.
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

    # step_label (not just the one-off "status" log line) carries the "this can take a while" caveat, since every
    # event sets it (settings.js installServer) and it stays in the step header. pip's piped output has no
    # incremental byte progress (just "Downloading torch-...whl (899.7 MB)", then silence until done), so the bar
    # stays indeterminate and multi-minute silence reads as "stuck" without reassurance.
    pip_meta = {
        "step": _STEP_PIP,
        "total_steps": _TOTAL_STEPS,
        "step_label": "Installing dependencies — this can take several minutes (PyTorch is a large download)",
    }
    yield {**pip_meta, "status": "Installing dependencies (this can take a while)..."}
    venv_python = venv_dir / "bin" / "python"
    # Real per-download percentages instead of silence (pip_progress).
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
