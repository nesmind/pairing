"""Cluster page: cluster mode on/off and each host's readiness checklist
(GET /api/cluster/readiness, see../matricxon/docs/CLUSTER_STORAGE.md). On = local Matricxon's
`cluster_role` "seed", applied at its next start.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import IMAGES_DIR
from app.schemas.cluster import ClusterCheck, ClusterHost, ClusterPaths, ClusterState, ClusterTimers
from app.services import (
    engine_service,
    image_engine_service,
    matricxon_http,
    matricxon_pool,
    matricxon_process,
    server_pool_broadcast,
    settings_service,
)
from app.services.cluster_first_node import FirstNode
from app.services.cluster_join import ClusterJoin
from app.services.cluster_nodes import ClusterNodes
from app.services.cluster_settings import ClusterSettings
from app.services.cluster_watch import ClusterWatch
from app.services.fs_probe import FilesystemProbe, tree_size

logger = logging.getLogger("llama_chat")
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)
_WATCH = ClusterWatch()  # one per process: compares each reading with the last
_FANOUT = 32
_PATH = "/api/cluster/readiness"


class ClusterService:
    def __init__(
        self,
        db: AsyncSession,
        hosts: list[str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        probe: FilesystemProbe | None = None,
        with_fresh: bool = False,
    ) -> None:
        self._with_fresh = with_fresh  # one extra call per node, only for the Cluster page
        self._probe = probe or FilesystemProbe()
        self._db = db
        self._hosts = hosts
        self._transport = transport

    async def _push_to_seed(self, config, target: str | None = None) -> str | None:
        return await ClusterSettings(self._db, self._transport, self._hosts).push_to_seed(config, target)

    async def state(self, message: str | None = None, only: list[str] | None = None) -> ClusterState:
        """The cluster as it is now; `only` reads just those hosts."""
        config = await settings_service.get_matricxon_server_config(self._db)
        supported = await engine_service.get_active_engine(self._db) == "matricxon"
        hosts = await self._read_hosts(only) if supported else []
        wanted = config.cluster_role
        # Only the seed follows the saved role; joiners have their own.
        seeds = [h for h in hosts if h.host not in config.cluster_hosts]
        restart = config.mode == "local" and any(h.reachable and h.role != wanted for h in seeds)
        images = await self._images_root()
        paths = ClusterPaths(
            models_path=config.models_path,
            prompt_cache_path=config.prompt_cache_path,
            advertise_url=config.advertise_url,
            first_node=config.first_node,
        )
        quorum = self.quorum_of(hosts)
        if only is None and supported:
            _WATCH.observe(hosts, quorum)
        return ClusterState(
            enabled=wanted != "off",
            supported=supported,
            restart_needed=restart,
            paths=paths,
            app_checks=[self._images_check(images)],
            cluster_id=ClusterJoin.id_of(hosts) if wanted != "off" else None,
            quorum=quorum,
            message=message,
            node_hosts=list(config.cluster_hosts),
            timers=self._timers_of(hosts),
            hosts=hosts,
            admin_auth=bool(config.admin_secret),
        )

    async def set_enabled(self, enabled: bool) -> ClusterState:
        config = await settings_service.get_matricxon_server_config(self._db)
        target = FirstNode.required(config) if enabled else None
        if not enabled and len(matricxon_pool.get_effective_hosts()) > 1:
            # One node leaves; the rest keep running consensus.
            raise ValueError("Remove the other nodes first (Nodes map), then turn cluster mode off.")
        config.cluster_role = "seed" if enabled else "off"
        logger.info("cluster: cluster mode turned %s", "ON" if enabled else "OFF")
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_process.write_env_file(config)
        return await self.state(await self._push_to_seed(config, target))

    async def set_paths(self, paths: ClusterPaths) -> ClusterState:
        """Saves the shared folders (blank = default), first node and address; nothing is moved."""
        config = await settings_service.get_matricxon_server_config(self._db)
        config.models_path = (paths.models_path or "").strip() or None
        config.prompt_cache_path = (paths.prompt_cache_path or "").strip() or None
        FirstNode.choose(config, paths.first_node)
        if paths.advertise_url is not None:
            config.advertise_url = paths.advertise_url.strip().rstrip("/") or None
        await settings_service.set_matricxon_server_config(self._db, config)
        matricxon_process.write_env_file(config)
        if config.first_node not in matricxon_pool.get_effective_hosts():  # a first node typed in by address
            matricxon_pool.refresh_from_config(config)
            await server_pool_broadcast.broadcast_refresh("/api/settings/matricxon/internal-refresh")
        used, note = await ClusterSettings(self._db, self._transport, self._hosts).publish_paths(
            config.models_path, config.prompt_cache_path
        )
        if used:
            return await self.state(note)
        # Before consensus only the first node is told, so its checklist shows the real folders
        first = config.first_node
        return await self.state(await self._push_to_seed(config, first) if first else None)

    async def activate(self) -> ClusterState:
        """Asks every reachable host to announce itself in the shared folder (each refuses until fully green)."""
        config = await settings_service.get_matricxon_server_config(self._db)
        if config.cluster_role == "off":
            raise ValueError("Turn cluster mode on first (the switch above), then activate the node.")
        hosts = self._hosts if self._hosts is not None else matricxon_pool.get_effective_hosts()
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            for host in hosts:
                try:
                    await client.post(host + "/api/cluster/activate")
                except httpx.HTTPError:
                    pass
        waiting = await ClusterNodes(self._db, self._transport).ensure_voters()
        state = await self.state(waiting)
        idle = [h.host for h in state.hosts if h.reachable and h.ready and not h.active]
        if idle and not waiting:  # ready but did not announce (e.g. restarting)
            state.message = (
                f"{', '.join(idle)} did not activate (its Matricxon may still be restarting): try again in a moment."
            )
        return state

    async def images_check(self) -> ClusterCheck:
        """Whether the saved-images folder is on shared storage."""
        return self._images_check(await self._images_root())

    async def _images_root(self) -> Path:
        return await image_engine_service.images_root(self._db, IMAGES_DIR)

    def _images_check(self, folder: Path) -> ClusterCheck:
        """Must be shared: with several pAIring instances, an image made on one would 404 on the others."""
        why = "every pAIring instance must read and write the same saved images"
        size = tree_size(folder)
        if not self._probe.is_shared(folder):
            kind = self._probe.fstype(folder) or "unknown"
            hint = (
                f"Move {folder} ({size / 1e9:.2f} GB) to shared storage mounted on every machine running pAIring, "
                f"then set it in the image engine's settings (Settings > Inference engines). "
                f"Now on a local '{kind}' filesystem."
            )
            return ClusterCheck(
                id="images",
                status="action_needed",
                what="images: not on shared storage",
                why=why,
                size_bytes=size,
                hint=hint,
            )
        if not os.access(folder, os.W_OK | os.X_OK):
            return ClusterCheck(
                id="images",
                status="action_needed",
                what="images: folder not writable",
                why=why,
                size_bytes=size,
                hint=f"{folder} is not writable by this app.",
            )
        return ClusterCheck(id="images", status="ok", what="images: shared and usable", why=why, size_bytes=size)

    async def _read_hosts(self, only: list[str] | None = None) -> list[ClusterHost]:
        """Every host's readiness, queried in parallel."""
        hosts = only
        if hosts is None:
            hosts = self._hosts if self._hosts is not None else matricxon_pool.get_effective_hosts()
        gate = asyncio.Semaphore(_FANOUT)

        async def one(client: httpx.AsyncClient, host: str) -> ClusterHost:
            async with gate:
                return await self._read_host(client, host, self._with_fresh)

        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            return list(await asyncio.gather(*(one(client, host) for host in hosts)))

    @staticmethod
    async def _read_consensus(client: httpx.AsyncClient, host: str) -> dict | None:
        """The consensus status, or None if the node runs none."""
        try:
            resp = await client.get(host + "/api/consensus/status")
            body = resp.json() if resp.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return None
        return body if body.get("enabled") else None

    @staticmethod
    def _timers_of(hosts: list[ClusterHost]) -> ClusterTimers | None:
        seen = next((h.consensus["timers"] for h in hosts if h.consensus and h.consensus.get("timers")), None)
        return ClusterTimers(**seen) if seen else None

    @staticmethod
    def quorum_of(hosts: list[ClusterHost]) -> str:
        """ "off" without consensus; "ok" with one majority leader (or a listed node hearing an unlisted
        leader); else "lost" (changes paused, chats not).
        """
        statuses = [h.consensus for h in hosts if h.consensus]
        if not statuses:
            return "off"
        leaders = [s for s in statuses if s.get("role") == "leader" and s.get("has_quorum")]
        if not leaders:  # a follower with a majority may hear an unlisted leader
            return "ok" if any(s.get("has_quorum") and s.get("leader") for s in statuses) else "lost"
        return "ok" if len(leaders) == 1 else "lost"

    @staticmethod
    async def _read_fresh(client: httpx.AsyncClient, host: str) -> bool | None:
        """Whether the node holds no data of its own (the First node box prefers one that does)."""
        try:
            resp = await client.get(host + "/api/cluster/fresh")
            fresh = resp.json().get("fresh") if resp.is_success else None
        except (httpx.HTTPError, ValueError, AttributeError):
            return None
        return fresh if isinstance(fresh, bool) else None

    @staticmethod
    async def _read_host(client: httpx.AsyncClient, host: str, with_fresh: bool = False) -> ClusterHost:
        try:
            resp = await client.get(host + _PATH)
        except httpx.HTTPError as exc:
            return ClusterHost(host=host, reachable=False, error=str(exc) or type(exc).__name__)
        if resp.status_code == 404:
            return ClusterHost(
                host=host, reachable=False, error="this Matricxon version has no cluster mode — update it"
            )
        if resp.is_error:
            return ClusterHost(host=host, reachable=False, error=f"HTTP {resp.status_code}")
        body = resp.json()
        consensus = await ClusterService._read_consensus(client, host)
        return ClusterHost(
            host=host,
            consensus=consensus,
            fresh=await ClusterService._read_fresh(client, host) if with_fresh and not consensus else None,
            reachable=True,
            role=body["role"],
            node_id=body["node_id"],
            ready=body["ready"],
            active=body["active"],
            checks=[ClusterCheck(**c) for c in body["checks"]],
            nodes=body["nodes"],
        )
