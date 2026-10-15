"""Propagates a live engine or image-engine config change (mode switch or remote host edit) to every other local
instance, the gap instance_db_broadcast fixed for database switches: ollama_pool/sdcpp_pool.refresh_from_config
update only their own process's memory, so with instance_count > 1 every instance except the one that got the
admin's save keeps routing through a stale host list until restart.

A small deliberate copy of instance_db_broadcast._other_live_ports rather than importing its private helper or
refactoring that incident-fixing module: ~10 lines, not worth the regression risk.
"""

import httpx

from app.config import INSTANCE_INDEX
from app.services import instance_process

_TIMEOUT = httpx.Timeout(10.0, connect=2.0)


def _other_live_ports() -> list[int]:
    tracked = instance_process.read_tracking()
    indices = {0} | {index for index, info in tracked.items() if instance_process.is_alive(info["pid"])}
    indices.discard(INSTANCE_INDEX)
    return [instance_process.port_for_index(index) for index in indices]


async def broadcast_refresh(path: str) -> list[str]:
    """POSTs to every other live instance's internal refresh endpoint at `path` (e.g.
    "/api/settings/ollama/internal-refresh").
    No body: each instance re-reads the config from the shared DB. Best-effort: a failed sibling doesn't undo
    the rest. Returns a description of each failure (empty if all refreshed).
    """
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for port in _other_live_ports():
            try:
                resp = await client.post(f"http://127.0.0.1:{port}{path}")
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                failures.append(f"instance on port {port}: {exc}")
    return failures
