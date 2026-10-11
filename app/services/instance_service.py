"""The "local instances" admin (Settings > System): an admin picks how many local app processes (1-8) run,
sharing this machine's DB/models/knowledge, with no manual steps (modeled on ollama_pool.py; see
ROADMAP.md "Concurrency: running multiple Ollama instances"). Process primitives: instance_process.py
(split for the file-size rule); this file is the async orchestration: locking and the three entry points
used by app/main.py and routers/instances_admin.py.

Only the *primary* (app.config.IS_PRIMARY, index 0, started via scripts/start.sh or systemd) calls
reconcile_on_startup/set_instance_count/terminate_all_siblings; siblings never spawn. Reboot survival
needs no per-instance units: under systemd, KillMode=control-group (default; scripts/install_on_fresh_server.sh)
stops siblings with the primary, and reconcile_on_startup respawns them. On plain scripts/start.sh there is
no cgroup, so the shutdown hook (app/main.py) and scripts/stop.sh's process match clean up.

The count lives in the AppSetting table (settings_service.get_instance_count/set_instance_count), not .env:
nothing needs it before the DB is up (unlike DATABASE_URL/AUTO_MIGRATE).
"""

import asyncio
import os

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import APP_PORT
from app.schemas import InstancesConfig, InstanceStatus
from app.services import instance_process, settings_service

MIN_INSTANCES = 1
MAX_INSTANCES = 8

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)

# One caller at a time may spawn/terminate/rewrite tracking; reconcile_on_startup and a live
# set_instance_count could otherwise race (both spawning index 2).
_lock = asyncio.Lock()


async def _reconcile_locked(db: AsyncSession) -> dict[int, dict]:
    """Shared body of reconcile_on_startup/set_instance_count; run only under `_lock`. Returns the fresh tracking
    dict."""
    desired = await settings_service.get_instance_count(db)
    tracked = instance_process.read_tracking()
    alive = {index: info for index, info in tracked.items() if instance_process.is_alive(info["pid"])}

    to_spawn, to_terminate = instance_process.plan_changes(desired, set(alive))
    for index in sorted(to_terminate):
        instance_process.terminate_sibling(alive[index]["pid"])
        del alive[index]
    for index in sorted(to_spawn):
        pid = instance_process.spawn_sibling(index)
        if pid is not None:
            alive[index] = {"pid": pid}

    instance_process.write_tracking(alive)
    return alive


async def reconcile_on_startup(db: AsyncSession) -> None:
    """Called once from app.main's startup handler (primary only): makes running siblings match the saved count,
    adopting ones still alive from before the last restart instead of re-spawning.
    """
    async with _lock:
        await _reconcile_locked(db)


async def set_instance_count(db: AsyncSession, count: int) -> InstancesConfig:
    """Backs PUT /api/settings/instances: persists the target and reconciles live in the same call, no restart
    (like db_config_service.save_and_apply).
    """
    if not (MIN_INSTANCES <= count <= MAX_INSTANCES):
        raise ValueError(f"count must be between {MIN_INSTANCES} and {MAX_INSTANCES}")
    await settings_service.set_instance_count(db, count)
    async with _lock:
        await _reconcile_locked(db)
    return await get_status(db)


async def _ping_health(port: int) -> bool:
    try:
        async with httpx.AsyncClient(timeout=_HEALTH_CHECK_TIMEOUT) as client:
            resp = await client.get(f"http://localhost:{port}/health")
            return resp.status_code == 200
    except httpx.HTTPError:
        return False


async def get_status(db: AsyncSession) -> InstancesConfig:
    """Saved target plus what runs now; index 0 (this primary) is always included. The health ping is
    informational for the admin UI; spawn/terminate depend only on the PID check in _reconcile_locked.
    """
    desired = await settings_service.get_instance_count(db)
    tracked = instance_process.read_tracking()
    alive = {index: info for index, info in tracked.items() if instance_process.is_alive(info["pid"])}

    ports = {0: APP_PORT, **{i: instance_process.port_for_index(i) for i in alive}}
    healthy = dict(
        zip(ports, await asyncio.gather(*(_ping_health(port) for port in ports.values())), strict=True),
    )
    instances = [
        InstanceStatus(
            index=index,
            port=port,
            pid=alive[index]["pid"] if index else os.getpid(),
            alive=True,
            healthy=healthy[index],
        )
        for index, port in sorted(ports.items())
    ]
    return InstancesConfig(count=desired, instances=instances)


async def terminate_all_siblings() -> None:
    """Called from app.main's shutdown handler (primary only); matters most on the plain scripts/stop.sh path
    (no cgroup cleanup, unlike a systemd stop/reboot).
    """
    async with _lock:
        tracked = instance_process.read_tracking()
        for info in tracked.values():
            instance_process.terminate_sibling(info["pid"])
        instance_process.write_tracking({})
