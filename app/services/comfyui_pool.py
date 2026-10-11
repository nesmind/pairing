"""Routes ComfyUI requests across host(s): the twin of ollama_pool.py (see its docstring: write-through
cache, local vs remote), both wrapping server_pool.HostPool. Simpler: no streaming chat to fail over, only
request/response calls (HostPool.call_with_failover in comfyui_client).
"""

import httpx

from app.config import COMFYUI_HOST
from app.schemas import ComfyUIProcessConfig
from app.services.server_pool import HostPool, ping_host

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
_pool.set_hosts([COMFYUI_HOST])


def refresh_from_config(config: ComfyUIProcessConfig) -> None:
    """Recomputes the live host list from `config`; call on admin save (routers/comfyui_admin.py) and at
    startup (startup_service.run_startup_tasks).
    """
    hosts = config.remote_hosts if config.mode == "remote" else [COMFYUI_HOST]
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


async def call_with_failover(make_call):
    return await _pool.call_with_failover(make_call)


async def check_hosts() -> dict[str, bool]:
    return await _pool.check_hosts(_HEALTH_CHECK_TIMEOUT, path="/")


async def check_host(host: str) -> bool:
    """One arbitrary host for the per-row "check before you save" button (see ollama_pool.check_host)."""
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/")
