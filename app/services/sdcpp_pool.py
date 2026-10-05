"""Routes stable-diffusion.cpp requests across host(s) — the sd-server twin of comfyui_pool.py (same HostPool)."""

import httpx

from app.config import SDCPP_HOST
from app.schemas import SdCppConfig
from app.services.server_pool import HostPool, ping_host

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
_pool.set_hosts([SDCPP_HOST])


def refresh_from_config(config: SdCppConfig) -> None:
    """Call on every config save and at startup (see startup_service.run_startup_tasks)."""
    _pool.set_hosts(config.remote_hosts if config.mode == "remote" else [SDCPP_HOST])


def get_effective_hosts() -> list[str]:
    return _pool.get_hosts()


async def call_with_failover(make_call):
    return await _pool.call_with_failover(make_call)


async def check_host(host: str) -> bool:
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/")
