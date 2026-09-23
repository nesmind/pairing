"""
Propagates a live ML engine/ComfyUI server-config change (mode switch or
remote host list edit) to every other local instance — the same real gap
app.services.instance_db_broadcast.py already fixed for a database
switch, for the same reason: app.services.ollama_pool.refresh_from_config/
comfyui_pool.refresh_from_config only ever updates *its own process's*
in-memory pool. In an instance_count > 1 deployment, every sibling is a
completely separate OS process with its own memory, so without this,
every instance but the one that actually received the admin's save
request keeps routing chat/image-generation requests through its own
stale host list until a full restart.

A small, deliberate duplicate of instance_db_broadcast._other_live_ports
rather than importing its private helper or refactoring that already-
shipped, incident-fixing module to share it — this is only ~10 lines and
not worth the churn/regression risk on working code for it.
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
    """POSTs to every other live instance's own internal refresh endpoint
    at `path` (e.g. "/api/settings/ollama/internal-refresh") — no body
    needed, since each instance just re-reads the current config from the
    shared DB itself. Best-effort: one unreachable/failed sibling doesn't
    undo anything already applied to this process or whichever others
    succeeded. Returns a human-readable description of each failure
    (empty list if every instance refreshed cleanly)."""
    failures: list[str] = []
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        for port in _other_live_ports():
            try:
                resp = await client.post(f"http://127.0.0.1:{port}{path}")
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                failures.append(f"instance on port {port}: {exc}")
    return failures
