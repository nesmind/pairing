"""Routes chat/embedding/model-management across Matricxon host(s): the analogue of ollama_pool.py, its own
module (not a wrapper of ollama_pool's state) so Matricxon traffic never shares host list, cooldown or
failure history with Ollama; they're unrelated backends configured independently (engine_service says
which serves traffic). Both wrap server_pool.HostPool, as comfyui_pool.py does.

"local" mode resolves to LOCAL_MATRICXON_HOST alone (the process matricxon_process.py manages); "remote"
to the admin's remote_hosts. Cached via refresh_from_config, not re-read per request (see ollama_pool.py).
"""

import asyncio

import httpx

from app.schemas import MatricxonServerConfig
from app.services import matricxon_http
from app.services.cluster_routing import NodeLoads, WarmUp
from app.services.server_pool import HostPool, RouteInfo, ping_host

# Matches../matricxon/app/config.py's default (MATRICXON_PORT=8420). Like ollama_pool.LOCAL_OLLAMA_HOST it is
# a fixed fact (matricxon_process passes no host/port override to scripts/start.sh), not admin-editable.
LOCAL_MATRICXON_HOST = "http://localhost:8420"

_HEALTH_CHECK_TIMEOUT = httpx.Timeout(3.0, connect=2.0)
_COOLDOWN_SECONDS = 30.0
_AFFINITY_TIMEOUT = httpx.Timeout(1.5, connect=0.5)
_SPILL_TIMEOUT = httpx.Timeout(15.0, connect=1.0)
_WARM_UP_TIMEOUT = httpx.Timeout(900.0, connect=2.0)  # a cold model load can take minutes

_pool = HostPool(cooldown_seconds=_COOLDOWN_SECONDS)
_pool.set_hosts([LOCAL_MATRICXON_HOST])
_all_hosts: list[str] = [LOCAL_MATRICXON_HOST]  # every node; the pool holds only those that serve chats


def refresh_from_config(config: MatricxonServerConfig) -> None:
    """Recomputes the live host list from `config`; call on admin save (routers/matricxon_admin.py) or when
    Matricxon becomes the active engine (engine_service.refresh_active_pool).
    """
    global _all_hosts
    hosts = config.remote_hosts if config.mode == "remote" else [LOCAL_MATRICXON_HOST]
    _all_hosts = hosts + [h for h in config.cluster_hosts if h not in hosts]
    serving = [h for h in _all_hosts if h not in config.voter_hosts]
    _pool.set_hosts(serving or _all_hosts)  # voter-only nodes get no chats (unless nothing else exists)
    matricxon_http.set_secret(config.admin_secret)


def get_effective_hosts() -> list[str]:
    """Every node of the cluster, voter-only ones included (cluster admin); chats use only the pool."""
    return list(_all_hosts)


def pick_host(exclude: frozenset[str] = frozenset()) -> str:
    return _pool.pick_host(exclude)


def mark_failure(host: str) -> None:
    _pool.mark_failure(host)


def mark_recovered(host: str) -> None:
    _pool.mark_recovered(host)


def track_request(host: str):
    return _pool.track_request(host)


async def stream_with_failover(make_stream, prefer: str | None = None, info: RouteInfo | None = None):
    async for chunk in _pool.stream_with_failover(make_stream, prefer=prefer, info=info):
        yield chunk


async def _load_model(host: str, model: str) -> None:
    async with matricxon_http.client(timeout=_WARM_UP_TIMEOUT) as client:
        resp = await client.post(f"{host}/api/generate", json={"model": model})
    resp.raise_for_status()


_loads = NodeLoads(matricxon_http.client)
_warm_up = WarmUp(_load_model)


async def route_info(model: str) -> RouteInfo:
    """What the chat-serving nodes report for `model` (cached ~2 s); empty with one node."""
    return await _loads.route_info(_pool.get_hosts(), model)


def note_model_used(host: str, model: str) -> None:
    _loads.note(host, model)


async def route_hints(tag: str | None, model: str) -> tuple[str | None, RouteInfo]:
    """(node holding chat `tag` in RAM, what the nodes report for `model`), asked in one round; also starts warming
    an idle node when every node with the model is busy (cluster_routing.WarmUp)."""
    holder, info = await asyncio.gather(find_holder(tag), route_info(model))
    _warm_up.start_if_needed(_pool, model, info)
    return holder, info


async def find_holder(tag: str | None) -> str | None:
    """The host holding chat `tag` in RAM now (its processed conversation), or None. Asks every host at once with
    a short timeout; a host that doesn't answer or lacks the endpoint doesn't count. Skipped with one host.
    """
    hosts = _pool.get_hosts()
    if not tag or len(hosts) < 2:
        return None
    async with matricxon_http.client(timeout=_AFFINITY_TIMEOUT) as client:
        answers = await asyncio.gather(*(_holds(client, host, tag) for host in hosts))
    holders = [host for host, holds in zip(hosts, answers, strict=True) if holds]
    return holders[0] if holders else None


async def lease_holder(lease: str) -> tuple[bool, str | None]:
    """(does the cluster run consensus?, the host that owns `lease` now). Asks every host at once."""
    hosts = _pool.get_hosts()
    async with matricxon_http.client(timeout=_AFFINITY_TIMEOUT) as client:
        statuses = await asyncio.gather(*(_consensus_status(client, host) for host in hosts))
    for host, status in zip(hosts, statuses, strict=True):
        if status and lease in status.get("holds", []):
            return True, host
    return any(statuses), None


async def _consensus_status(client: httpx.AsyncClient, host: str) -> dict | None:
    try:
        resp = await client.get(f"{host}/api/consensus/status")
        body = resp.json() if resp.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        return None
    return body if body.get("enabled") else None


async def _holds(client: httpx.AsyncClient, host: str, tag: str) -> bool:
    try:
        resp = await client.get(f"{host}/api/cache/holds", params={"tag": tag})
        return resp.status_code == 200 and bool(resp.json().get("ram"))
    except (httpx.HTTPError, ValueError):
        return False


async def spill_chat(host: str, tag: str) -> int:
    """Best effort: asks `host` to write the chat's cached conversation to the shared disk (models stay loaded)
    for the host now serving it. Returns how many conversations moved.
    """
    try:
        async with matricxon_http.client(timeout=_SPILL_TIMEOUT) as client:
            resp = await client.post(f"{host}/api/cache/spill", json={"tag": tag})
        return int(resp.json().get("spilled", 0)) if resp.status_code == 200 else 0
    except (httpx.HTTPError, ValueError):
        return 0


async def check_hosts() -> dict[str, bool]:
    return await _pool.check_hosts(_HEALTH_CHECK_TIMEOUT, path="/api/tags")


async def check_host(host: str) -> bool:
    """Same probe as check_hosts for one arbitrary host, possibly unsaved; Settings > External servers verifies a
    remote URL before saving (routers/matricxon_admin.py check_host).
    """
    return await ping_host(host, _HEALTH_CHECK_TIMEOUT, path="/api/tags")
