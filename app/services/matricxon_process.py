"""
Starts/stops the local Matricxon process via its own scripts/start.sh and scripts/stop.sh — the supervisor
behind Settings > External servers > Matricxon's "local" mode. Unlike Ollama (a single self-contained binary,
see app/services/ollama_process.py) or ComfyUI (a bare `python main.py` this app launches and tracks itself, see
app/services/comfyui_process.py), Matricxon has no binary and no invocation this app constructs by hand — it's a
sibling Python project with its own venv, whose own scripts/start.sh already writes and manages a PID file
(`run/matricxon.pid` inside the project directory) — read directly here rather than reinventing PID tracking the
way comfyui_process.py has to for a subprocess it spawns itself.

Only the primary process (app.config.IS_PRIMARY) may start/stop it — same reasoning as
app/services/comfyui_service.py's identical guard.
"""

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

from app.config import BASE_DIR, IS_PRIMARY
from app.schemas import MatricxonServerConfig, MatricxonServerStatus
from app.services.matricxon_pool import LOCAL_MATRICXON_HOST

logger = logging.getLogger("llama_chat")

# Matricxon isn't something this app installs itself (see app.services.matricxon_installer's own docstring — its
# GitHub install is a stub today) — the one default worth guessing is the sibling checkout this app's own
# development layout already uses.
_DEFAULT_PROJECT_DIR = BASE_DIR.parent / "matricxon"

_START_TIMEOUT_SECONDS = 30.0
_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# Same reasoning as ollama_process._lock / comfyui_service._lock: a live Start/Stop click racing a status poll
# (or another start/stop) must never interleave.
_lock = asyncio.Lock()


def _find_project_dir(project_dir: str | None = None) -> Path | None:
    """`project_dir` is the admin's own override (see Settings > External servers' "Already have Matricxon
    installed elsewhere?" link) — checked first and, if it looks like a real checkout (has scripts/start.sh),
    used exclusively. Falls back to the sibling-directory default otherwise, same "override, else a sane guess"
    pattern as ollama_process._find_binary."""
    candidates = [Path(project_dir).expanduser()] if project_dir else []
    candidates.append(_DEFAULT_PROJECT_DIR)
    for candidate in candidates:
        if (candidate / "scripts" / "start.sh").is_file():
            return candidate
    return None


def is_installed(project_dir: str | None = None) -> bool:
    return _find_project_dir(project_dir) is not None


def auto_detect_project_dir() -> str | None:
    """What _find_project_dir resolves to with no override at all — Settings > External servers shows this as
    the project_dir field's own placeholder, mirroring ollama_process.auto_detect_binary's identical purpose."""
    found = _find_project_dir()
    return str(found) if found else None


def resolve_project_dir(project_dir: str | None = None) -> Path | None:
    """Public wrapper around _find_project_dir for callers outside this module that need the
    actual resolved directory — honoring a saved override, unlike auto_detect_project_dir's
    "what would blank resolve to" behavior — see matricxon_direct_puller._resolve_dest."""
    return _find_project_dir(project_dir)


def auto_detect_models_dir() -> str | None:
    """What a blank models_path resolves to — Matricxon's own default models directory inside
    the auto-detected project checkout. None if no checkout can be found at all (mirrors
    auto_detect_project_dir's own None case)."""
    project_dir = _find_project_dir()
    return str(project_dir / "data" / "models") if project_dir else None


def _pid_file(project_dir: Path) -> Path:
    return project_dir / "run" / "matricxon.pid"


def _read_pid(project_dir: Path) -> int | None:
    try:
        return int(_pid_file(project_dir).read_text().strip())
    except (OSError, ValueError):
        return None


def _is_running(project_dir: Path) -> bool:
    pid = _read_pid(project_dir)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        # Real but not ours (e.g. owned by another user) — kill(pid, 0) still reached a live process.
        return True
    return True


def _build_env(config: MatricxonServerConfig, proxy_url: str | None = None) -> dict[str, str]:
    env = {**os.environ}
    if config.models_path is not None:
        env["MATRICXON_MODELS_DIR"] = config.models_path
    if config.max_loaded_models is not None:
        env["MATRICXON_MAX_LOADED_MODELS"] = str(config.max_loaded_models)
    if config.memory_safety_margin is not None:
        env["MATRICXON_MEMORY_SAFETY_MARGIN"] = str(config.memory_safety_margin)
    env["MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE"] = "true" if config.enable_quantized_native_compute else "false"
    env["MATRICXON_LOG_LEVEL"] = str(config.log_level)
    # Matricxon's own model-pull downloader (a plain httpx client) honors these the same standard way Ollama's Go
    # binary does — see ollama_process._build_env.
    if proxy_url:
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
    return env


def _run_script(project_dir: Path, script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(project_dir / "scripts" / script)],
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _start(project_dir: Path, config: MatricxonServerConfig, proxy_url: str | None = None) -> None:
    result = _run_script(project_dir, "start.sh", _build_env(config, proxy_url))
    if result.returncode != 0:
        raise RuntimeError(f"scripts/start.sh failed: {(result.stderr or result.stdout).strip()}")


def _stop(project_dir: Path) -> None:
    # scripts/stop.sh already does its own graceful-then-SIGKILL wait (see that script) — no extra polling loop
    # needed here the way ollama_process._stop/comfyui_process.terminate need for the raw signals they send.
    if not _is_running(project_dir):
        return
    result = _run_script(project_dir, "stop.sh", {**os.environ})
    if result.returncode != 0:
        logger.warning("Matricxon scripts/stop.sh exited %d: %s", result.returncode, result.stderr.strip())


async def _ping_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(f"{LOCAL_MATRICXON_HOST}/api/tags")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def _wait_until_healthy(timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await _ping_health():
            return True
        await asyncio.sleep(0.5)
    return False


async def get_status(project_dir_override: str | None = None) -> MatricxonServerStatus:
    project_dir = _find_project_dir(project_dir_override)
    if project_dir is None or not _is_running(project_dir):
        return MatricxonServerStatus(running=False, installed=project_dir is not None)
    return MatricxonServerStatus(running=True, installed=True, pid=_read_pid(project_dir), healthy=await _ping_health())


async def start(config: MatricxonServerConfig, proxy_url: str | None = None) -> MatricxonServerStatus:
    """Backs POST .../start. Raises RuntimeError if no checkout can be found or it doesn't come back up healthy
    in time, ValueError if called on anything but the primary instance — same contract as ollama_process.start."""
    if not IS_PRIMARY:
        raise ValueError("Matricxon can only be managed from the primary instance.")
    async with _lock:
        project_dir = _find_project_dir(config.project_dir)
        if project_dir is None:
            raise RuntimeError(
                "Could not find a Matricxon checkout (no scripts/start.sh at the configured or default project "
                "directory)."
            )
        if _is_running(project_dir):
            return await get_status(config.project_dir)
        logger.info("Starting Matricxon...")
        await asyncio.to_thread(_start, project_dir, config, proxy_url)
        if not await _wait_until_healthy(_START_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"Matricxon did not come up within {_START_TIMEOUT_SECONDS:.0f}s — check its own logs/matricxon.log."
            )
    return await get_status(config.project_dir)


async def stop(project_dir_override: str | None = None) -> MatricxonServerStatus:
    if not IS_PRIMARY:
        raise ValueError("Matricxon can only be managed from the primary instance.")
    async with _lock:
        project_dir = _find_project_dir(project_dir_override)
        if project_dir is not None:
            await asyncio.to_thread(_stop, project_dir)
    return await get_status(project_dir_override)


async def apply_local_config(config: MatricxonServerConfig, proxy_url: str | None = None) -> MatricxonServerStatus:
    """Backs a config Save while Matricxon is already running locally — like Ollama, Matricxon only reads its
    MATRICXON_* env vars at its own process startup, so this stops and restarts it with the new values. Callers
    (see app/routers/matricxon_admin.py) are expected to have already warned about the interruption."""
    if not IS_PRIMARY:
        raise ValueError("Matricxon can only be managed from the primary instance.")
    async with _lock:
        project_dir = _find_project_dir(config.project_dir)
        if project_dir is None:
            raise RuntimeError("Could not find a Matricxon checkout to restart.")
        await asyncio.to_thread(_stop, project_dir)
        logger.info("Starting Matricxon with the updated local configuration...")
        await asyncio.to_thread(_start, project_dir, config, proxy_url)
        if not await _wait_until_healthy(_START_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"Matricxon did not come back up within {_START_TIMEOUT_SECONDS:.0f}s — check its own "
                "logs/matricxon.log."
            )
    return await get_status(config.project_dir)
