"""
Propagates a live database switch (see app.database.switch_database) to
every other local "instance" (app.services.instance_process/
instance_service) running on this machine — split out purely to keep
app/services/db_config_service.py under CLAUDE.md's file-size rule.

app.database.switch_database only ever rebinds *its own process's* live
engine; in an instance_count > 1 deployment, every sibling is a
completely separate OS process with its own memory, so without this,
only whichever single instance actually received the
PUT /api/settings/database request (see instance_proxy_http.py's
EXEMPT_PREFIXES, which correctly keeps this route from ever being
proxied to a random *different* single instance, but doesn't address the
*rest* of the fleet) would ever pick up the new database — every other
instance keeps silently running against the old one until a full
restart. Confirmed as a real production incident: an admin got
intermittently logged out depending on which instance's load-balanced
response they happened to get, since only one of two instances had
actually switched.
"""

import httpx

from app.config import INSTANCE_INDEX
from app.services import instance_process

_SWITCH_TIMEOUT = httpx.Timeout(10.0, connect=2.0)


def _other_live_ports() -> list[int]:
    """Every other instance currently up on this machine, self excluded
    — the same tracking-file + liveness check app.services.instance_pool's
    own pick_instance() already relies on for the exact same "which
    instances actually exist right now" question (there's no separate
    live registry anywhere, just instances.json + a fresh is_alive()
    check every time)."""
    tracked = instance_process.read_tracking()
    indices = {0} | {index for index, info in tracked.items() if instance_process.is_alive(info["pid"])}
    indices.discard(INSTANCE_INDEX)
    return [instance_process.port_for_index(index) for index in indices]


async def broadcast_database_switch(database_url: str) -> list[str]:
    """Calls every other live instance's own internal switch endpoint
    (see app/routers/db_admin.py's POST /api/settings/database/internal-switch)
    so its own live engine gets rebound too — best-effort: one
    unreachable/failed sibling doesn't undo the switch already applied to
    this process and whichever others succeeded, since each instance
    only ever rebinds its own connection independently. Returns a
    human-readable description of each failure (empty list if every
    instance switched cleanly), so the caller can tell the admin exactly
    which instance(s) still need attention (most likely a restart)."""
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
