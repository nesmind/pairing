"""
The "local instances" admin feature (Settings > System): lets an admin
pick how many local app-process instances (1-8) should be running,
sharing this machine's DB/models/knowledge base, with no manual steps
beyond picking the number — see ROADMAP.md's "Concurrency: running
multiple Ollama instances" for the sibling feature this is modeled
after on the Ollama side (app/services/ollama_pool.py). The actual
subprocess/file primitives live in app/services/instance_process.py
(split out purely to stay under CLAUDE.md's file-size rule) — this file
is the async orchestration around them: locking, and the three entry
points app/main.py and app/routers/instances_admin.py call.

Only the *primary* process (app.config.IS_PRIMARY — index 0, the one a
human actually starts via scripts/start.sh or systemd) ever calls
reconcile_on_startup/set_instance_count/terminate_all_siblings; a
spawned sibling never spawns siblings of its own. Reboot survival needs
no per-instance systemd units: on a systemd-managed host, KillMode=
control-group (the default — see scripts/install_on_fresh_server.sh)
already stops every sibling alongside the primary, and
reconcile_on_startup respawns them all the next time the primary comes
up. On the plain scripts/start.sh path there's no cgroup safety net, so
the shutdown hook (see app/main.py) and scripts/stop.sh's process match
are what keep things clean there instead.

The desired count lives in the AppSetting table (see
app.services.settings_service.get_instance_count/set_instance_count),
not .env — nothing needs it before the database is already up and
queryable, unlike DATABASE_URL/AUTO_MIGRATE.
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

# Only one caller at a time may spawn/terminate/rewrite tracking state —
# reconcile_on_startup and a live set_instance_count from the Settings
# page could otherwise race (e.g. both deciding to spawn index 2).
_lock = asyncio.Lock()


async def _reconcile_locked(db: AsyncSession) -> dict[int, dict]:
    """Shared body for reconcile_on_startup/set_instance_count — must
    only ever run under `_lock`. Returns the fresh tracking dict."""
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
    """Called once from app.main's startup handler, primary only: makes
    the currently-running siblings match the saved instance count,
    adopting anything still alive from before this primary process's own
    last restart rather than blindly re-spawning everything."""
    async with _lock:
        await _reconcile_locked(db)


async def set_instance_count(db: AsyncSession, count: int) -> InstancesConfig:
    """Backs PUT /api/settings/instances: persists the new target, then
    reconciles live (spawns/terminates immediately) in the same call —
    no restart needed, same shape as db_config_service.save_and_apply."""
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
    """Current saved target plus what's actually observed running right
    now, index 0 (this primary — always alive/healthy, since it's
    answering this very request) always included. The health ping is
    read-only status for the admin UI only — spawn/terminate decisions
    never depend on it, only on the fast synchronous PID check in
    _reconcile_locked."""
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
    """Called from app.main's shutdown handler, primary only — see this
    module's own docstring for why this matters most on the plain
    scripts/stop.sh path (no cgroup to clean them up automatically
    there, unlike a real systemd stop/reboot)."""
    async with _lock:
        tracked = instance_process.read_tracking()
        for info in tracked.values():
            instance_process.terminate_sibling(info["pid"])
        instance_process.write_tracking({})
