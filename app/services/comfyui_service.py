"""
The ComfyUI process-supervision admin feature (Settings > System): lets
an admin point this app at an already-installed ComfyUI (its venv python
+ main.py path) and start/stop it directly, the same way an admin manages
"Local instances" — but unlike that feature, this is an explicit,
imperative action, not a declarative "reconcile to N" target. See
app/services/comfyui_process.py for the subprocess/file primitives this
orchestrates (split out purely to stay under CLAUDE.md's file-size rule,
same reasoning app/services/instance_service.py was split from
instance_process.py for).

Deliberately has no reconcile_on_startup/terminate_all_siblings
equivalent: ComfyUI is a heavyweight, potentially GPU-bound external
process an admin explicitly starts/stops ("get it running from the
settings section" — the literal ask this feature was built for), not a
lightweight disposable copy of this app's own process the way a sibling
instance is. It is expected to keep running across an app restart, and
this app never auto-starts or auto-stops it on its own boot/shutdown.

Only the *primary* process (app.config.IS_PRIMARY) may start/stop it —
see start/stop's own guard — since two processes racing to spawn/kill the
same external executable would corrupt each other's view of it in a
multi-instance deployment.
"""

import asyncio

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import COMFYUI_HOST, IS_PRIMARY
from app.schemas import ComfyUIStatus
from app.services import comfyui_installer, comfyui_process, settings_service

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# Only one caller at a time may spawn/terminate/rewrite tracking state —
# a live Start/Stop click racing a status poll could otherwise corrupt
# the tracking file. Same reasoning as instance_service._lock.
_lock = asyncio.Lock()


async def _ping_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(COMFYUI_HOST)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def get_status(db: AsyncSession) -> ComfyUIStatus:
    """The fast synchronous PID check (is_alive) is the real source of
    truth; the health ping is read-only, informational status for the
    admin UI only, the same split instance_service.get_status uses."""
    tracked = comfyui_process.read_tracking()
    config = await settings_service.get_comfyui_config(db)
    installed = comfyui_installer.is_installed() or bool(config.main_py_path)
    if tracked is None or not config.main_py_path or not comfyui_process.is_alive(tracked["pid"], config.main_py_path):
        return ComfyUIStatus(running=False, installed=installed)
    return ComfyUIStatus(running=True, installed=installed, pid=tracked["pid"], healthy=await _ping_health())


async def start(db: AsyncSession) -> ComfyUIStatus:
    """Backs POST .../start. Raises ValueError (router turns this into a
    400) if the launch path isn't configured yet, or if called on
    anything but the primary instance."""
    if not IS_PRIMARY:
        raise ValueError("ComfyUI can only be managed from the primary instance.")
    config = await settings_service.get_comfyui_config(db)
    if not config.python_path or not config.main_py_path:
        raise ValueError("Set the Python and main.py paths before starting ComfyUI.")

    async with _lock:
        tracked = comfyui_process.read_tracking()
        if tracked and comfyui_process.is_alive(tracked["pid"], config.main_py_path):
            return await get_status(db)
        pid = comfyui_process.spawn(config.python_path, config.main_py_path, config.extra_args)
        comfyui_process.write_tracking({"pid": pid} if pid is not None else None)
    return await get_status(db)


async def stop(db: AsyncSession) -> ComfyUIStatus:
    """Backs POST .../stop. Raises ValueError on anything but the
    primary instance, same as start."""
    if not IS_PRIMARY:
        raise ValueError("ComfyUI can only be managed from the primary instance.")
    config = await settings_service.get_comfyui_config(db)

    async with _lock:
        tracked = comfyui_process.read_tracking()
        if tracked and config.main_py_path:
            comfyui_process.terminate(tracked["pid"], config.main_py_path)
        comfyui_process.write_tracking(None)
    return await get_status(db)
