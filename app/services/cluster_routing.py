"""Model-aware routing for a Matricxon cluster, on top of server_pool.HostPool's load and cache-holder rules:
what each node reports about itself (`NodeLoads`: loaded models, active replies from every client, threads,
free memory) and warming an idle node before it is needed (`WarmUp`), so overflow chats find the model ready
instead of waiting for a cold load (~100 s for 3 GB from a USB stick). Preferring loaded nodes is capped in
`HostPool.pick_host` (`warm_slack`), so load keeps spreading.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

from app.services.server_pool import HostPool, RouteInfo

logger = logging.getLogger("llama_chat")

ClientFactory = Callable[..., httpx.AsyncClient]
# Free memory needed to load a model, as a multiple of its size (KV cache and work buffers on top).
ROOM_FACTOR = 1.2


@dataclass(frozen=True)
class NodeLoad:
    models: dict[str, int]  # loaded model -> size in bytes
    active: int = 0
    threads: int = 0  # 0: unknown (an older node answering only /api/ps)
    free_memory: int | None = None


class NodeLoads:
    """Each node's `GET /api/node/load` (older nodes: `/api/ps`), cached for `ttl` seconds so a chat costs at most
    one request per node; `note` records a model a chat just used, without waiting for the next read."""

    def __init__(self, client: ClientFactory, ttl: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self._client = client
        self._ttl = ttl
        self._clock = clock
        self._seen: dict[str, tuple[float, NodeLoad]] = {}

    async def route_info(self, hosts: list[str], model: str) -> RouteInfo:
        """What the hosts report, for a request for `model`; empty for a single host (nothing to choose)."""
        if len(hosts) < 2 or not model:
            return RouteInfo()
        now = self._clock()
        stale = [h for h in hosts if h not in self._seen or now - self._seen[h][0] > self._ttl]
        if stale:
            async with self._client(timeout=httpx.Timeout(1.5, connect=0.5)) as client:
                answers = await asyncio.gather(*(self._read(client, h) for h in stale))
            for host, load in zip(stale, answers, strict=True):
                self._seen[host] = (now, load)
        loads = {h: self._seen[h][1] for h in hosts}
        size = max((load.models.get(model, 0) for load in loads.values()), default=0)
        most = max((load.threads for load in loads.values()), default=0)
        return RouteInfo(
            warm=frozenset(h for h, load in loads.items() if model in load.models),
            active={h: load.active for h, load in loads.items()},
            capacity={h: load.threads / most for h, load in loads.items() if load.threads and most},
            no_room=frozenset(
                h
                for h, load in loads.items()
                if size and load.free_memory is not None and load.free_memory < size * ROOM_FACTOR
            ),
        )

    def note(self, host: str, model: str) -> None:
        at, load = self._seen.get(host, (self._clock(), NodeLoad(models={})))
        self._seen[host] = (at, NodeLoad({**load.models, model: load.models.get(model, 0)}, load.active,
                                         load.threads, load.free_memory))  # fmt: skip

    @staticmethod
    async def _read(client: httpx.AsyncClient, host: str) -> NodeLoad:
        try:
            resp = await client.get(f"{host}/api/node/load")
            if resp.status_code == 200:
                body = resp.json()
                models = {m["name"]: int(m.get("size_bytes", 0)) for m in body.get("models", [])}
                return NodeLoad(models, int(body.get("active", 0)), int(body.get("threads", 0)),
                                body.get("free_memory_bytes"))  # fmt: skip
            resp = await client.get(f"{host}/api/ps")  # an older node
            entries = resp.json().get("models", []) if resp.status_code == 200 else []
            return NodeLoad({m.get("name") or m.get("model"): int(m.get("size", 0)) for m in entries})
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return NodeLoad(models={})


class WarmUp:
    """When every node that has the model is already busy and another node sits idle without it (and has room),
    loads the model there in the background (Matricxon `POST /api/generate` with no prompt: load only), at most
    once per node and model per `cooldown` seconds. The load counts as that node's work in the pool, so chats keep
    going to ready nodes meanwhile."""

    def __init__(
        self,
        load: Callable[[str, str], Awaitable[None]],
        cooldown: float = 600.0,
        busy_at: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._load = load
        self._cooldown = cooldown
        self._busy_at = busy_at
        self._clock = clock
        self._last: dict[tuple[str, str], float] = {}
        self._tasks: set[asyncio.Task] = set()

    def pick(self, pool: HostPool, model: str, info: RouteInfo) -> str | None:
        """The idle node to warm for `model`, or None."""
        healthy = pool.healthy_hosts()
        warm_nodes = [h for h in healthy if h in info.warm]
        if not warm_nodes or any(pool.load(h, info) < self._busy_at for h in warm_nodes):
            return None
        now = self._clock()
        for host in healthy:
            if host in info.warm or host in info.no_room or pool.load(host, info) > 0:
                continue
            if now - self._last.get((host, model), -1e9) >= self._cooldown:
                return host
        return None

    def start_if_needed(self, pool: HostPool, model: str, info: RouteInfo) -> str | None:
        host = self.pick(pool, model, info)
        if host is None:
            return None
        self._last[(host, model)] = self._clock()
        logger.info("cluster: warming %s with %s (every node that has it is busy)", host, model)
        task = asyncio.create_task(self._run(pool, host, model))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return host

    async def _run(self, pool: HostPool, host: str, model: str) -> None:
        async with pool.track_request(host):
            try:
                await self._load(host, model)
            except Exception as exc:  # noqa: BLE001 - a warm-up is best effort
                logger.warning("cluster: warming %s with %s failed: %s", host, model, exc)
