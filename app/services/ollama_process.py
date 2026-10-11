"""Starts/stops the local Ollama: the supervisor behind Settings > External servers > Ollama "local" mode.
Unlike other admin toggles, a local-mode parameter can't apply live: Ollama reads OLLAMA_* only at startup,
so a change kills and restarts it, interrupting every in-flight reply on every local instance sharing it (not
a quiet reconcile like instance_service).

Binary discovery and the `ollama serve` launch mirror scripts/start.sh (same PATH-then-fallback search,
same OLLAMA_MODELS); this and that script are the only places that start Ollama. No PID file (unlike
comfyui_process.py): the binary name "ollama" is distinctive, so `pgrep -x`/`pkill -x` work, simpler than
comfyui_process's /proc/{pid}/cmdline (ComfyUI's generic "python" comm isn't distinctive).

Only the primary (app.config.IS_PRIMARY) may start/stop it, as comfyui_service.py.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import time

import httpx

from app.config import BASE_DIR, EXTERNAL_DIR, IS_PRIMARY
from app.schemas import OllamaServerConfig, OllamaServerStatus
from app.services.ollama_pool import LOCAL_OLLAMA_HOST

logger = logging.getLogger("llama_chat")

# The last entry is where ollama_installer puts a binary it downloaded: checked last so a system-wide install wins.
_FALLBACK_BIN_PATHS = (
    os.path.expanduser("~/bin/ollama"),
    "/usr/local/bin/ollama",
    "/usr/bin/ollama",
    str(EXTERNAL_DIR / "ollama" / "bin" / "ollama"),
)

_STOP_GRACE_SECONDS = 5.0
_START_TIMEOUT_SECONDS = 20.0
_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# As comfyui_service._lock: a Start/Stop must not interleave with a status poll or another start/stop.
_lock = asyncio.Lock()


def _find_binary(binary_path: str | None = None) -> str | None:
    """`binary_path` is the admin's override (Settings > External servers' "Already have Ollama installed
    elsewhere?"): checked first and exclusively if given; if missing or not executable, falls through to the
    PATH/fixed-location search.
    """
    if binary_path and os.path.isfile(binary_path) and os.access(binary_path, os.X_OK):
        return binary_path
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in _FALLBACK_BIN_PATHS:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def is_installed(binary_path: str | None = None) -> bool:
    """Whether an `ollama` binary is reachable; Settings > External servers uses it to show Start/Stop or "Install"
    (ollama_installer)."""
    return _find_binary(binary_path) is not None


def auto_detect_binary() -> str | None:
    """What _find_binary resolves to with no override: a public wrapper (routers/ollama_admin.py auto-detected-path)
    so Settings shows binary_path's placeholder as a real value.
    """
    return _find_binary()


def _process_name(binary_path: str | None = None) -> str:
    """The exact process name `_is_running`/`_stop` match: "ollama", or an overridden `binary_path`'s file name
    (pgrep/pkill -x need an exact match; otherwise a differently named override would look never-running).
    Truncated to 15 bytes like the kernel's comm (TASK_COMM_LEN), as instance_process.py's "pAIring-server-N"; a real
    gap.
    """
    return os.path.basename(binary_path)[:15] if binary_path else "ollama"


def _is_running(binary_path: str | None = None) -> bool:
    name = _process_name(binary_path)
    return subprocess.run(["pgrep", "-x", name], capture_output=True, check=False).returncode == 0


def _stop(binary_path: str | None = None) -> None:
    name = _process_name(binary_path)
    if not _is_running(binary_path):
        return
    subprocess.run(["pkill", "-x", name], check=False)
    deadline = time.monotonic() + _STOP_GRACE_SECONDS
    while time.monotonic() < deadline and _is_running(binary_path):
        time.sleep(0.2)
    if _is_running(binary_path):
        subprocess.run(["pkill", "-9", "-x", name], check=False)


def auto_detect_models_path() -> str:
    """What `_build_env` resolves OLLAMA_MODELS to with no override. Always concrete (pAIring's own models/ folder is
    created on demand, not searched for), so the models_path placeholder is real (unlike auto_detect_binary).
    """
    return str(BASE_DIR / "models")


def _build_env(config: OllamaServerConfig, proxy_url: str | None = None) -> dict[str, str]:
    env = {**os.environ, "OLLAMA_MODELS": config.models_path or auto_detect_models_path()}
    # None means "let Ollama use its default": omitted, not the string "None".
    if config.num_parallel is not None:
        env["OLLAMA_NUM_PARALLEL"] = str(config.num_parallel)
    if config.keep_alive is not None:
        env["OLLAMA_KEEP_ALIVE"] = config.keep_alive
    if config.max_loaded_models is not None:
        env["OLLAMA_MAX_LOADED_MODELS"] = str(config.max_loaded_models)
    if config.context_length is not None:
        env["OLLAMA_CONTEXT_LENGTH"] = str(config.context_length)
    # Ollama's model-pull requests (its Go net/http honors these vars; http_proxy_service). Local mode only: this
    # app doesn't control a remote Ollama's env.
    if proxy_url:
        env["HTTP_PROXY"] = proxy_url
        env["HTTPS_PROXY"] = proxy_url
        env["http_proxy"] = proxy_url
        env["https_proxy"] = proxy_url
    return env


def _start(config: OllamaServerConfig, proxy_url: str | None = None) -> None:
    binary = _find_binary(config.binary_path)
    if binary is None:
        raise RuntimeError("Could not find the 'ollama' binary (checked PATH, ~/bin, /usr/local/bin, /usr/bin).")
    log_dir = BASE_DIR / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(log_dir / "ollama.log", "ab")
    subprocess.Popen(
        [binary, "serve"],
        env=_build_env(config, proxy_url),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


async def _ping_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(f"{LOCAL_OLLAMA_HOST}/api/tags")
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


async def get_status(binary_path: str | None = None) -> OllamaServerStatus:
    """The synchronous pgrep check is the truth; the health ping is informational for the admin UI (as
    comfyui_service.get_status). `pid` is always None: identity is by process *name*, not a tracked PID.
    `binary_path` is the admin's override (see _find_binary; routers/ollama_admin.py passes
    OllamaServerConfig.binary_path).
    """
    installed = is_installed(binary_path)
    if not _is_running(binary_path):
        return OllamaServerStatus(running=False, installed=installed)
    return OllamaServerStatus(running=True, installed=installed, healthy=await _ping_health())


async def start(config: OllamaServerConfig, proxy_url: str | None = None) -> OllamaServerStatus:
    """POST.../start. RuntimeError if the binary isn't found or it doesn't come up healthy in time; ValueError if
    not the primary (the router makes a 4xx/502; same as comfyui_service.start). `proxy_url`: the admin's HTTP proxy, if
    any (_build_env).
    """
    if not IS_PRIMARY:
        raise ValueError("Ollama can only be managed from the primary instance.")
    async with _lock:
        if _is_running(config.binary_path):
            return await get_status(config.binary_path)
        logger.info("Starting Ollama...")
        await asyncio.to_thread(_start, config, proxy_url)
        if not await _wait_until_healthy(_START_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"Ollama did not come up within {_START_TIMEOUT_SECONDS:.0f}s — check data/logs/ollama.log."
            )
    return await get_status(config.binary_path)


async def stop(binary_path: str | None = None) -> OllamaServerStatus:
    """POST.../stop. ValueError if not the primary, as start. `_stop`'s polling loop blocks up to
    _STOP_GRACE_SECONDS, so asyncio.to_thread keeps it off the event loop (other requests aren't frozen).
    `binary_path` also picks the process *name* to target (_process_name): an override not named "ollama" needs it to be
    stoppable.
    """
    if not IS_PRIMARY:
        raise ValueError("Ollama can only be managed from the primary instance.")
    async with _lock:
        await asyncio.to_thread(_stop, binary_path)
    return await get_status(binary_path)


async def apply_local_config(config: OllamaServerConfig, proxy_url: str | None = None) -> OllamaServerStatus:
    """Backs a config Save while Ollama runs locally: it reads OLLAMA_* only at startup, so this stops and
    restarts it. Callers (routers/ollama_admin.py) should have warned about the interruption. If Ollama wasn't
    running, the full stop+start still runs and the config applies at next start. `proxy_url`: see _build_env.
    """
    if not IS_PRIMARY:
        raise ValueError("Ollama can only be managed from the primary instance.")
    async with _lock:
        await asyncio.to_thread(_stop, config.binary_path)
        logger.info("Starting Ollama with the updated local configuration...")
        await asyncio.to_thread(_start, config, proxy_url)
        if not await _wait_until_healthy(_START_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"Ollama did not come back up within {_START_TIMEOUT_SECONDS:.0f}s — check data/logs/ollama.log."
            )
    return await get_status(config.binary_path)
