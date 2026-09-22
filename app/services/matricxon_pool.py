"""
Routes chat/embedding/model-management requests across Matricxon host(s) — the Matricxon-flavored analogue of
app/services/ollama_pool.py, deliberately its own independent module (not a thin wrapper around ollama_pool's
state) so Matricxon traffic never shares a host list, cooldown state, or failure history with Ollama's — the two
are unrelated backends an admin can each configure/start/stop on their own (see app.services.engine_service for
which one actually serves live traffic right now). Both wrap the same generic app/services/server_pool.py
HostPool, the same way app/services/comfyui_pool.py already does for a third, unrelated backend.

"local" mode resolves to LOCAL_MATRICXON_HOST alone (the one process this app itself manages, see
app/services/matricxon_process.py); "remote" mode resolves to the admin's own configured remote_hosts instead.
Cached here via refresh_from_config rather than re-read from the DB on every request — see ollama_pool.py's own
docstring for the full reasoning, identical here.
"""

import httpx

from app.schemas import MatricxonServerConfig
from app.services.server_pool import HostPool, ping_host

# Matches ../matricxon/app/config.py's own default port (MATRICXON_PORT=8420) — like ollama_pool.LOCAL_OLLAMA_HOST,
# this is a fixed fact (app.services.matricxon_process never passes an explicit host/port override to
# scripts/start.sh), not an admin-editable deployment setting.
LOCAL_MATRICXON_HOST = "http://localhost:8420"

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
_pool.set_hosts([LOCAL_MATRICXON_HOST])


def refresh_from_config(config: MatricxonServerConfig) -> None:
    """Recomputes the live host list from `config` — call this whenever the admin saves Settings > External
    servers > Matricxon (see app/routers/matricxon_admin.py) or Matricxon becomes the active engine (see
    app.services.engine_service.refresh_active_pool)."""
    hosts = config.remote_hosts if config.mode == "remote" else [LOCAL_MATRICXON_HOST]
    _pool.set_hosts(hosts)


def get_effective_hosts() -> list[str]:
    return _pool.get_hosts()


def pick_host(exclude: frozenset[str] = frozenset()) -> str:
    return _pool.pick_host(exclude)


def mark_failure(host: str) -> None:
    _pool.mark_failure(host)


def mark_recovered(host: str) -> None:
    _pool.mark_recovered(host)


def track_request(host: str):
    return _pool.track_request(host)


async def stream_with_failover(make_stream):
    async for chunk in _pool.stream_with_failover(make_stream):
        yield chunk


async def check_hosts() -> dict[str, bool]:
    return await _pool.check_hosts(_HEALTH_CHECK_TIMEOUT, path="/api/tags")


async def check_host(host: str) -> bool:
    """Same probe as check_hosts, for one arbitrary host not necessarily saved to the live pool yet — Settings >
    External servers uses this to verify a remote-mode candidate URL before saving (see
    app/routers/matricxon_admin.py's check_host endpoint)."""
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/api/tags")
