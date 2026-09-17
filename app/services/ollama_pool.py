"""
Routes chat/embedding requests across Ollama host(s) — a thin,
Ollama-flavored wrapper around app/services/server_pool.py's generic
HostPool (the actual selection/cooldown/failover algorithm lives there
now, shared with app/services/comfyui_pool.py). Every public function
name here is unchanged from before that extraction, so none of this
module's own callers (app/services/ollama_client.py,
app/routers/health.py) needed to change their own call shape — only how
the pool's host list gets populated does.

The live host list is admin-configured from Settings > External servers
(see app.services.settings_service.get_ollama_server_config) — "local"
mode resolves to LOCAL_OLLAMA_HOST alone (the one process this app
itself manages, see app/services/ollama_process.py); "remote" mode
resolves to the admin's own configured remote_hosts list instead. Cached
here via refresh_from_config rather than re-read from the DB on every
chat request (this sits on the hot generation path) — loaded once at
startup (see app.services.startup_service) and refreshed immediately
whenever the admin saves a change, the same write-through pattern
app.services.instance_pool.set_cached_proxy_mode already uses.
"""

import httpx

from app.schemas import OllamaServerConfig
from app.services.server_pool import HostPool, ping_host

# The one process this app itself manages (see app/services/ollama_process.py) always binds to Ollama's own
# built-in default — that supervisor never passes an explicit host/port to the `ollama serve` subprocess it
# launches — so this is a fixed fact, not a deployment setting: no .env/environment variable feeds it, and there
# is no UI field for it either (mirrors app.config.COMFYUI_HOST's identical "deployment fact, not admin-editable"
# reasoning). "Remote" mode's hosts, by contrast, are fully admin-editable — see OllamaServerConfig.remote_hosts.
LOCAL_OLLAMA_HOST = "http://localhost:11434"

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
# A single-host pool by default (matching this module's own pre-tab
# behavior) so anything calling pick_host before the first
# refresh_from_config (startup_service runs it early, but tests/tools
# importing this module directly shouldn't hit an empty pool) still works.
_pool.set_hosts([LOCAL_OLLAMA_HOST])


def refresh_from_config(config: OllamaServerConfig) -> None:
    """Recomputes the live host list from `config` — call this whenever
    the admin saves Settings > External servers > Ollama (see
    app/routers/ollama_admin.py) or at startup (see
    app.services.startup_service.run_startup_tasks)."""
    hosts = config.remote_hosts if config.mode == "remote" else [LOCAL_OLLAMA_HOST]
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
    """Pings every currently-configured host (Ollama's own cheap
    `GET /api/tags`) — see HostPool.check_hosts and
    app/routers/health.py, the only caller."""
    return await _pool.check_hosts(_HEALTH_CHECK_TIMEOUT, path="/api/tags")


async def check_host(host: str) -> bool:
    """Same probe as check_hosts, for one arbitrary host that isn't
    necessarily saved to the live pool yet — Settings > External servers
    uses this to let an admin verify a remote-mode candidate URL is
    actually reachable before saving the form (see
    app/routers/ollama_admin.py's check_host endpoint)."""
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/api/tags")
