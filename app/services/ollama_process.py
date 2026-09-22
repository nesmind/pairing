"""
Starts/stops the local Ollama process — the supervisor behind Settings > External servers > Ollama's "local"
mode. Unlike every other admin toggle in this app, changing a local-mode parameter genuinely can't apply live:
Ollama only reads its OLLAMA_* env vars at its own process startup, so a config change means killing and
restarting Ollama itself, interrupting every in-flight reply across every local instance sharing it — not a
quiet background reconcile like app.services.instance_service's supervisor.

Binary discovery and the actual `ollama serve` launch mirror scripts/start.sh's own logic exactly (same
PATH-then-fallback-paths search, same OLLAMA_MODELS directory) — this is the one place in the Python app,
besides that shell script, that ever starts Ollama itself. No PID tracking file is needed here (unlike
app/services/comfyui_process.py): Ollama's own binary name ("ollama") is distinctive, so `pgrep -x`/`pkill -x`
identify it directly — simpler than comfyui_process.py's /proc/{pid}/cmdline approach, which exists only
because ComfyUI's generic "python" comm/argv0 isn't distinctive enough on its own.

Only the *primary* process (app.config.IS_PRIMARY) may start/stop it — same reasoning as
app/services/comfyui_service.py's identical guard.
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

# The last entry is where app.services.ollama_installer places a binary
# it downloaded itself — checked last since a system-wide install (PATH,
# or one of the other fixed locations) should always take precedence
# over one this app manages on its own.
_FALLBACK_BIN_PATHS = (
    os.path.expanduser("~/bin/ollama"),
    "/usr/local/bin/ollama",
    "/usr/bin/ollama",
    str(EXTERNAL_DIR / "ollama" / "bin" / "ollama"),
)

_STOP_GRACE_SECONDS = 5.0
_START_TIMEOUT_SECONDS = 20.0
_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# Same reasoning as comfyui_service._lock: a live Start/Stop click racing
# a status poll (or another start/stop) must never interleave.
_lock = asyncio.Lock()


def _find_binary(binary_path: str | None = None) -> str | None:
    """`binary_path` is the admin's own override (see Settings > External
    servers' "Already have Ollama installed elsewhere?" link) — checked
    first and, if given, exclusively. Falls through to the normal PATH/
    fixed-location search below if it doesn't exist/isn't executable,
    rather than reporting "not installed" outright."""
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
    """Whether an `ollama` binary is reachable at all — used by the
    Settings > External servers page to decide whether to show
    Start/Stop or an "Install" prompt (see app.services.ollama_installer,
    a later addition)."""
    return _find_binary(binary_path) is not None


def auto_detect_binary() -> str | None:
    """What _find_binary resolves to with no override at all — a public
    wrapper (see app/routers/ollama_admin.py's auto-detected-path
    endpoint) so Settings > External servers can show the binary_path
    field's own placeholder as a real, current value instead of just the
    words "auto-detected from PATH"."""
    return _find_binary()


def _process_name(binary_path: str | None = None) -> str:
    """The exact process name `_is_running`/`_stop` match against — "ollama" normally, or (for an admin-
    overridden `binary_path`, see _find_binary) that file's own name instead: pgrep/pkill -x need an exact
    match, and an override not literally named "ollama" would otherwise report as never-running even while
    genuinely up. Truncated to 15 bytes to match the kernel's own comm field (TASK_COMM_LEN) — the same
    constraint this codebase's sibling-instance naming (instance_process.py's "pAIring-server-N") already
    accounts for; confirmed live as a real, not just theoretical, gap."""
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
    """What `_build_env` resolves OLLAMA_MODELS to with no admin override — unlike
    auto_detect_binary, always a concrete path (pAIring's own self-contained models/ folder is
    created on demand, not searched for), so Settings > External servers can show a real
    placeholder for the models_path field the same way it does for binary_path."""
    return str(BASE_DIR / "models")


def _build_env(config: OllamaServerConfig, proxy_url: str | None = None) -> dict[str, str]:
    env = {**os.environ, "OLLAMA_MODELS": config.models_path or auto_detect_models_path()}
    # None means "don't set it, let Ollama use its own built-in default"
    # — simply omitted rather than passed as the string "None".
    if config.num_parallel is not None:
        env["OLLAMA_NUM_PARALLEL"] = str(config.num_parallel)
    if config.keep_alive is not None:
        env["OLLAMA_KEEP_ALIVE"] = config.keep_alive
    if config.max_loaded_models is not None:
        env["OLLAMA_MAX_LOADED_MODELS"] = str(config.max_loaded_models)
    if config.context_length is not None:
        env["OLLAMA_CONTEXT_LENGTH"] = str(config.context_length)
    # Ollama's own outbound model-pull requests (its Go binary's net/http
    # honors these standard vars) — see app.services.http_proxy_service.
    # Only reachable in local mode, since a remote Ollama isn't a process
    # this app spawns/controls the env of.
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
    """The fast synchronous pgrep check is the real source of truth; the
    health ping is read-only, informational status for the admin UI
    only — same split app.services.comfyui_service.get_status uses.
    `pid` is omitted (always None) since identity here is by process
    *name*, not a tracked PID — see this module's own docstring.
    `binary_path` is the admin's own override, if any — see
    _find_binary's own docstring; callers pass
    OllamaServerConfig.binary_path through (see app/routers/ollama_admin.py)."""
    installed = is_installed(binary_path)
    if not _is_running(binary_path):
        return OllamaServerStatus(running=False, installed=installed)
    return OllamaServerStatus(running=True, installed=installed, healthy=await _ping_health())


async def start(config: OllamaServerConfig, proxy_url: str | None = None) -> OllamaServerStatus:
    """Backs POST .../start. Raises RuntimeError if the binary can't be
    found or it doesn't come back up healthy in time, ValueError if
    called on anything but the primary instance (router turns both into
    a clean 4xx/502, mirroring comfyui_service.start's own contract).
    `proxy_url` is Settings > System's admin-configured HTTP proxy, if
    any — see _build_env."""
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
    """Backs POST .../stop. Raises ValueError on anything but the
    primary instance, same as start. `_stop`'s polling loop blocks for
    real wall-clock time (up to _STOP_GRACE_SECONDS) — asyncio.to_thread
    keeps that off the event loop, so other users' unrelated requests
    (a page load, /health) aren't frozen out while this runs.
    `binary_path` also decides which process *name* to actually target
    (see _process_name) — an overridden binary not literally named
    "ollama" needs this to be stoppable at all, the same reason
    get_status/start take it."""
    if not IS_PRIMARY:
        raise ValueError("Ollama can only be managed from the primary instance.")
    async with _lock:
        await asyncio.to_thread(_stop, binary_path)
    return await get_status(binary_path)


async def apply_local_config(config: OllamaServerConfig, proxy_url: str | None = None) -> OllamaServerStatus:
    """Backs a config Save while Ollama is already running in local mode
    — every other admin toggle in this app applies live, but Ollama only
    reads its OLLAMA_* env vars at its own process startup, so this
    stops and restarts it with the new values. Callers (see
    app/routers/ollama_admin.py) are expected to have already warned
    about the interruption before calling. A no-op restart (still
    exercises the full stop+start path) if Ollama wasn't already
    running — the new config simply takes effect the next time it's
    started. `proxy_url` — see _build_env."""
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
