"""ComfyUI process-supervision admin (Settings > System): point this app at an installed ComfyUI (venv
python + main.py) and start/stop it, an explicit imperative action unlike "Local instances"' declarative
"reconcile to N". Primitives: comfyui_process.py (split for the file-size rule, like instance_service.py).

No reconcile_on_startup/terminate_all_siblings equivalent: ComfyUI is a heavyweight, possibly GPU-bound
process an admin starts and stops explicitly, expected to outlive an app restart. This app never starts or
stops it on its own boot/shutdown.

Only the primary (app.config.IS_PRIMARY) may start/stop it (see start/stop): two processes racing to
spawn/kill one executable would corrupt each other's view.
"""

import asyncio

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import COMFYUI_HOST, IS_PRIMARY
from app.schemas import ComfyUIStatus
from app.services import comfyui_installer, comfyui_process, settings_service

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# One caller at a time may spawn/terminate/rewrite tracking state; a Start/Stop racing a status poll could
# corrupt the tracking file (like instance_service._lock).
_lock = asyncio.Lock()


async def _ping_health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(COMFYUI_HOST)
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def get_status(db: AsyncSession) -> ComfyUIStatus:
    """The PID check (is_alive) is the truth; the health ping is informational for the admin UI (as in
    instance_service.get_status)."""
    tracked = comfyui_process.read_tracking()
    config = await settings_service.get_comfyui_config(db)
    installed = comfyui_installer.is_installed() or bool(config.main_py_path)
    if tracked is None or not config.main_py_path or not comfyui_process.is_alive(tracked["pid"], config.main_py_path):
        return ComfyUIStatus(running=False, installed=installed)
    return ComfyUIStatus(running=True, installed=installed, pid=tracked["pid"], healthy=await _ping_health())


async def start(db: AsyncSession) -> ComfyUIStatus:
    """POST.../start. Raises ValueError (router: 400) if the launch path isn't configured or this isn't the primary."""
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
    """POST.../stop. Raises ValueError on non-primary, like start."""
    if not IS_PRIMARY:
        raise ValueError("ComfyUI can only be managed from the primary instance.")
    config = await settings_service.get_comfyui_config(db)

    async with _lock:
        tracked = comfyui_process.read_tracking()
        if tracked and config.main_py_path:
            comfyui_process.terminate(tracked["pid"], config.main_py_path)
        comfyui_process.write_tracking(None)
    return await get_status(db)
