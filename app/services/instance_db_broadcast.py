"""Propagates a live database switch (app.database.switch_database) to every other local instance
(instance_process/instance_service); split from db_config_service.py for the file-size rule.

switch_database rebinds only its own process's engine, and siblings are separate OS processes. Without
this only the instance that received PUT /api/settings/database (instance_proxy_http.EXEMPT_PREFIXES stops
proxying it, but doesn't reach the rest) switches; the others keep the old database until restart. A real
incident: an admin was intermittently logged out depending on which of two instances answered.
"""

import httpx

from app.config import INSTANCE_INDEX
from app.services import instance_process

_SWITCH_TIMEOUT = httpx.Timeout(10.0, connect=2.0)


def _other_live_ports() -> list[int]:
    """Other instances currently up on this machine (self excluded), by the tracking-file + is_alive check that
    instance_pool.pick_instance uses (no live registry exists).
    """
    tracked = instance_process.read_tracking()
    indices = {0} | {index for index, info in tracked.items() if instance_process.is_alive(info["pid"])}
    indices.discard(INSTANCE_INDEX)
    return [instance_process.port_for_index(index) for index in indices]


async def broadcast_database_switch(database_url: str) -> list[str]:
    """Calls each other live instance's internal switch endpoint (routers/db_admin.py POST
    /api/settings/database/internal-switch). Best-effort: a failed sibling doesn't undo the others, as each
    rebinds independently. Returns a description of each failure (empty if all switched) so the admin knows
    which instance(s) need a restart.
    """
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=_SWITCH_TIMEOUT) as client:
        for port in _other_live_ports():
            try:
                resp = await client.post(
                    f"http://127.0.0.1:{port}/api/settings/database/internal-switch",
                    json={"database_url": database_url},
                )
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                failures.append(f"instance on port {port}: {exc}")
    return failures
