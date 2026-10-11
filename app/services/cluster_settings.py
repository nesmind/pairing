"""Cluster-wide settings written through the cluster: any voter (the leader if found) is asked, so
nothing depends on the seed. Nodes apply them themselves (Matricxon's cluster/shared_settings.py).
"""

import asyncio
import logging

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.cluster import ClusterCache, ClusterHost, ClusterTimers
from app.services import matricxon_http, matricxon_pool

logger = logging.getLogger("llama_chat")
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


class ConsensusOffError(Exception):
    """No consensus, nothing to write through."""


class ClusterSettings:
    def __init__(
        self,
        db: AsyncSession,
        transport: httpx.AsyncBaseTransport | None = None,
        hosts: list[str] | None = None,
    ) -> None:
        self._db = db
        self._transport = transport
        self._hosts = hosts

    async def entry_voter(self) -> tuple[str, dict] | None:
        """(address, status) of the best voter: the leader, else one with a majority."""
        hosts = self._hosts if self._hosts is not None else matricxon_pool.get_effective_hosts()
        statuses = await asyncio.gather(*(self.status(h) for h in hosts))
        voters = [(h, s) for h, s in zip(hosts, statuses, strict=True) if s and s.get("is_voter")]

        def rank(item: tuple[str, dict]) -> tuple[bool, bool]:
            status = item[1]
            return not (status.get("role") == "leader" and status.get("has_quorum")), not status.get("has_quorum")

        return min(voters, key=rank, default=None)

    async def voters_seen(self) -> list[ClusterHost]:
        """Every host running consensus, with its status."""
        hosts = self._hosts if self._hosts is not None else matricxon_pool.get_effective_hosts()
        statuses = await asyncio.gather(*(self.status(h) for h in hosts))
        return [ClusterHost(host=h, reachable=True, consensus=s) for h, s in zip(hosts, statuses, strict=True) if s]

    async def status(self, host: str) -> dict | None:
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.get(host + "/api/consensus/status")
                body = resp.json() if resp.status_code == 200 else {}
            except (httpx.HTTPError, ValueError):
                return None
        return body if body.get("enabled") else None

    async def publish(self, key: str, value: dict) -> str | None:
        """Writes one setting through a voter. Returns the error, or None."""
        entry = await self.entry_voter()
        if entry is None:
            raise ConsensusOffError("no voter is reachable")
        async with matricxon_http.client(timeout=httpx.Timeout(30.0, connect=5.0), transport=self._transport) as client:
            try:
                resp = await client.put(entry[0] + "/api/consensus/config", json={"key": key, "value": value})
            except httpx.HTTPError as exc:
                return f"{entry[0]} could not be reached: {exc}"
        if resp.is_error or not resp.json().get("applied"):
            return "The cluster did not accept the change in time (is there a quorum?)."
        logger.info("cluster: setting %s changed through %s", key, entry[0])
        return None

    async def publish_paths(self, models: str | None, cache: str | None) -> tuple[bool, str | None]:
        """(went through the cluster?, a problem); not through the cluster without consensus."""
        try:
            problem = await self.publish("cluster/paths", {"models_dir": models or "", "prompt_cache_dir": cache or ""})
        except ConsensusOffError:
            return False, None
        return True, problem and f"Saved, but {problem}"

    async def set_timers(self, timers: ClusterTimers) -> str | None:
        return await self.publish("cluster/timers", timers.model_dump())

    async def set_cache(self, cache: ClusterCache) -> str | None:
        return await self.publish("cluster/cache", cache.model_dump())

    async def push_to_seed(self, config, target: str | None = None) -> str | None:
        """Tells the seed (`target`, else the first non-added node) to apply the settings itself: it writes
        its own .env and restarts. Returns an error message, or None.
        """
        hosts = self._hosts if self._hosts is not None else matricxon_pool.get_effective_hosts()
        seeds = [target] if target else [h for h in hosts if h not in config.cluster_hosts]
        if not seeds:
            return None
        body = {
            "role": "off" if config.cluster_role == "off" else "seed",
            "models_dir": config.models_path or "",
            "prompt_cache_dir": config.prompt_cache_path or "",
            # None leaves the node's setting alone ("" would delete it)
            "advertise_url": config.advertise_url,
        }
        async with matricxon_http.client(timeout=_TIMEOUT, transport=self._transport) as client:
            try:
                resp = await client.post(seeds[0] + "/api/cluster/configure", json=body)
            except httpx.HTTPError as exc:
                return f"Saved, but {seeds[0]} could not be told: {exc}"
        if resp.status_code == 404:
            return f"Saved, but {seeds[0]} has no cluster mode: update Matricxon."
        if resp.is_error:
            detail = resp.json().get("detail", f"HTTP {resp.status_code}") if resp.content else resp.status_code
            return f"Saved, but {seeds[0]} refused it: {detail}"
        return None
