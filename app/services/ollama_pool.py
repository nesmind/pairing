"""Routes chat/embedding requests across Ollama host(s): a thin wrapper around server_pool.HostPool (the
selection/cooldown/failover algorithm, shared with comfyui_pool.py). Public names are unchanged, so callers
(ollama_client.py, routers/health.py) were unaffected by that extraction.

The host list is admin-configured (Settings > External servers, settings_service.get_ollama_server_config):
"local" resolves to LOCAL_OLLAMA_HOST alone (the process ollama_process.py manages), "remote" to the
admin's remote_hosts. Cached via refresh_from_config, not read from the DB per chat request (hot path):
loaded at startup (startup_service) and refreshed on every admin save, like instance_pool.set_cached_proxy_mode.
"""

import httpx

from app.schemas import OllamaServerConfig
from app.services.server_pool import HostPool, ping_host

# The managed process binds Ollama's built-in default (ollama_process never passes host/port to `ollama
# serve`), so this is a fixed fact, not a setting: no env var or UI field (like app.config.COMFYUI_HOST).
# "Remote" hosts are admin-editable (OllamaServerConfig.remote_hosts).
LOCAL_OLLAMA_HOST = "http://localhost:11434"

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
# A single-host pool by default so pick_host works before the first refresh_from_config (startup_service runs
# it early, but tests/tools importing this module shouldn't see an empty pool).
_pool.set_hosts([LOCAL_OLLAMA_HOST])


def refresh_from_config(config: OllamaServerConfig) -> None:
    """Recomputes the live host list from `config`; call on admin save (routers/ollama_admin.py) or at startup
    (startup_service.run_startup_tasks).
    """
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
    """Pings every configured host (Ollama's cheap `GET /api/tags`); see HostPool.check_hosts, routers/health.py (only
    caller)."""
    return await _pool.check_hosts(_HEALTH_CHECK_TIMEOUT, path="/api/tags")


async def check_host(host: str) -> bool:
    """Same probe as check_hosts for one arbitrary, possibly unsaved host; Settings > External servers verifies a
    remote URL before saving (routers/ollama_admin.py check_host).
    """
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/api/tags")
