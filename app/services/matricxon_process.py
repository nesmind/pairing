"""Starts/stops the local Matricxon via its scripts/start.sh and scripts/stop.sh: the supervisor behind
Settings > Inference engines > Matricxon "local" mode. Unlike Ollama (one binary, ollama_process.py) or
stable-diffusion.cpp (an `sd-server` this app tracks, sdcpp_process.py), Matricxon is a sibling Python project
with its own venv whose scripts/start.sh manages a PID file (`run/matricxon.pid` in the project), read here
directly instead of reinventing tracking.

Only the primary (app.config.IS_PRIMARY) may start/stop it, as sdcpp_service.py.
"""

import asyncio
import logging
import os
import re
import subprocess
import time
from pathlib import Path

import httpx

from app.config import BASE_DIR, IS_PRIMARY
from app.schemas import MatricxonServerConfig, MatricxonServerStatus
from app.services import env_file
from app.services.matricxon_pool import LOCAL_MATRICXON_HOST

logger = logging.getLogger("llama_chat")

# This app doesn't install Matricxon itself (matricxon_installer's GitHub install is a stub), so the one
# default worth guessing is the sibling checkout of this app's development layout.
_DEFAULT_PROJECT_DIR = BASE_DIR.parent / "matricxon"

_START_TIMEOUT_SECONDS = 30.0
_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# As ollama_process._lock / sdcpp_service._lock: a Start/Stop must not interleave with a status poll or another
# start/stop.
_lock = asyncio.Lock()


def _find_project_dir(project_dir: str | None = None) -> Path | None:
    """`project_dir` is the admin's override (Settings > Inference engines' "Already have Matricxon installed
    elsewhere?"): used exclusively if it looks like a checkout (has scripts/start.sh), else the sibling-directory
    default (like ollama_process._find_binary).
    """
    candidates = [Path(project_dir).expanduser()] if project_dir else []
    candidates.append(_DEFAULT_PROJECT_DIR)
    for candidate in candidates:
        if (candidate / "scripts" / "start.sh").is_file():
            return candidate
    return None


def is_installed(project_dir: str | None = None) -> bool:
    return _find_project_dir(project_dir) is not None


def auto_detect_project_dir() -> str | None:
    """What _find_project_dir resolves to with no override; the project_dir placeholder (like
    ollama_process.auto_detect_binary)."""
    found = _find_project_dir()
    return str(found) if found else None


def resolve_project_dir(project_dir: str | None = None) -> Path | None:
    """Public wrapper of _find_project_dir for outside callers needing the resolved directory, honoring a saved
    override (unlike auto_detect_project_dir); see matricxon_direct_puller._resolve_dest.
    """
    return _find_project_dir(project_dir)


def auto_detect_models_dir() -> str | None:
    """What a blank models_path resolves to: Matricxon's default models directory in the auto-detected checkout.
    None if no checkout is found (like auto_detect_project_dir).
    """
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
        # Real but not ours (e.g. another user's): kill(pid, 0) still reached a live process.
        return True
    return True


def _concurrency_env_values(config: MatricxonServerConfig) -> dict[str, str | None]:
    """The concurrent-user tuning fields as env values (None = unset, Matricxon's default)."""
    return {
        "MATRICXON_PROMPT_CACHE_SLOTS": _int_or_none(config.prompt_cache_slots),
        "MATRICXON_PROMPT_CACHE_BUDGET_MB": _int_or_none(config.prompt_cache_budget_mb),
        "MATRICXON_MAX_DECODE_BATCH": _int_or_none(config.max_decode_batch),
    }


def _int_or_none(value: int | None) -> str | None:
    return str(value) if value is not None else None


def _build_env(config: MatricxonServerConfig, proxy_url: str | None = None) -> dict[str, str]:
    env = {**os.environ}
    if config.models_path is not None:
        env["MATRICXON_MODELS_DIR"] = config.models_path
    if config.prompt_cache_path is not None:
        env["MATRICXON_PROMPT_CACHE_DIR"] = config.prompt_cache_path
    if config.advertise_url is not None:
        env["MATRICXON_ADVERTISE_URL"] = config.advertise_url
    if config.max_loaded_models is not None:
        env["MATRICXON_MAX_LOADED_MODELS"] = str(config.max_loaded_models)
    if config.keep_alive_minutes is not None:
        env["MATRICXON_DEFAULT_KEEP_ALIVE_SECONDS"] = str(config.keep_alive_minutes * 60)
    for variable, value in _concurrency_env_values(config).items():
        if value is not None:
            env[variable] = value
    if config.memory_safety_margin is not None:
        env["MATRICXON_MEMORY_SAFETY_MARGIN"] = str(config.memory_safety_margin)
    if config.torch_threads is not None:
        env["MATRICXON_TORCH_THREADS"] = str(config.torch_threads)
    env["MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE"] = "true" if config.enable_quantized_native_compute else "false"
    env["MATRICXON_GEMV_BACKEND"] = config.gemv_backend
    env["MATRICXON_DEVICE"] = config.device
    env["MATRICXON_GPU_WEIGHT_MODE"] = config.gpu_weight_mode
    env["MATRICXON_LOG_LEVEL"] = str(config.log_level)
    env["MATRICXON_CLUSTER_ROLE"] = config.cluster_role
    # Matricxon's model-pull downloader (plain httpx) honors these like Ollama's Go binary (ollama_process._build_env).
    if proxy_url:
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
    return env


def write_env_file(config: MatricxonServerConfig) -> None:
    """Persists `config`'s local-mode fields into Matricxon's own .env (its Settings loads
    `env_file=".env"`,../matricxon/app/config.py, so no change there), not only this process's env when this app
    launches it
    (_build_env). On: after a laptop restart, Matricxon started another way (scripts/start.sh directly,
    systemd,...) fell back to its empty default models directory. This applies the config however the process
    starts; called on every save (routers/matricxon_admin.py _save_and_broadcast). No-op if the project directory
    can't be resolved yet, so saving for an unfinished install doesn't fail.
    """
    if config.mode == "remote":
        return  # remote mode: pAIring manages no Matricxon and must not write into any checkout's.env
    project_dir = resolve_project_dir(config.project_dir)
    if project_dir is None:
        return
    env_file.write_env_values(
        {
            "MATRICXON_MODELS_DIR": config.models_path,
            "MATRICXON_PROMPT_CACHE_DIR": config.prompt_cache_path,
            "MATRICXON_ADVERTISE_URL": config.advertise_url,
            "MATRICXON_MAX_LOADED_MODELS": (
                str(config.max_loaded_models) if config.max_loaded_models is not None else None
            ),
            "MATRICXON_DEFAULT_KEEP_ALIVE_SECONDS": (
                str(config.keep_alive_minutes * 60) if config.keep_alive_minutes is not None else None
            ),
            **_concurrency_env_values(config),
            "MATRICXON_MEMORY_SAFETY_MARGIN": (
                str(config.memory_safety_margin) if config.memory_safety_margin is not None else None
            ),
            "MATRICXON_ENABLE_QUANTIZED_NATIVE_COMPUTE": "true" if config.enable_quantized_native_compute else "false",
            "MATRICXON_GEMV_BACKEND": config.gemv_backend,
            "MATRICXON_TORCH_THREADS": str(config.torch_threads) if config.torch_threads is not None else None,
            "MATRICXON_DEVICE": config.device,
            "MATRICXON_GPU_WEIGHT_MODE": config.gpu_weight_mode,
            "MATRICXON_LOG_LEVEL": str(config.log_level),
            "MATRICXON_CLUSTER_ROLE": config.cluster_role if config.cluster_role != "off" else None,
        },
        path=project_dir / ".env",
    )


def _run_script(project_dir: Path, script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(project_dir / "scripts" / script)],
        cwd=project_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# Matricxon's one-line startup refusals (e.g. a missing GPU device): "... matricxon cannot start: <why>".
_CANNOT_START_LINE = re.compile(r"matricxon cannot start: (.+)$")
_STARTUP_ERROR_LINE = re.compile(r"^(?:[\w.]+\.)?(\w*(?:Error|Exception)): (.+)$")


def _startup_failure_reason(output: str) -> str | None:
    """The cause of a failed start from its log tail: Matricxon's "matricxon cannot start: <why>" (e.g. a CUDA
    device with no GPU), else the last "SomeError: message" traceback line, so Settings shows the cause, not a
    traceback. None if neither.
    """
    for line in reversed(output.strip().splitlines()):
        refusal = _CANNOT_START_LINE.search(line.strip())
        if refusal:
            return refusal.group(1)
        match = _STARTUP_ERROR_LINE.match(line.strip())
        if match:
            return f"{match.group(1)}: {match.group(2)}"
    return None


def _start(project_dir: Path, config: MatricxonServerConfig, proxy_url: str | None = None) -> None:
    result = _run_script(project_dir, "start.sh", _build_env(config, proxy_url))
    if result.returncode != 0:
        output = (result.stderr or result.stdout).strip()
        reason = _startup_failure_reason(output)
        if reason:
            raise RuntimeError(f"Matricxon failed to start — {reason}")
        raise RuntimeError(f"scripts/start.sh failed: {output}")


def _stop(project_dir: Path) -> None:
    # scripts/stop.sh already waits gracefully then SIGKILLs; no polling loop like
    # ollama_process._stop/tracked_process.terminate's raw signals.
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
    """POST.../start. RuntimeError if no checkout is found or it doesn't come up healthy in time; ValueError if not
    the primary (as ollama_process.start).
    """
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
    """Backs a config Save while Matricxon runs locally. It reads MATRICXON_* only at startup (like Ollama), so this
    stops and restarts it. Callers (routers/matricxon_admin.py) should have warned about the interruption.
    """
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
