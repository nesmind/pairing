"""The "Shared storage" map node: green when every answering node passes the models, cache and key checks."""

from app.schemas.cluster import ClusterCheck, ClusterHost, ClusterState, MapNode

STORAGE_CHECKS = {"models", "prompt_cache", "cache_key"}


def _short_text(text: str, limit: int = 34) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StorageNode:
    @staticmethod
    def _worst_checks(checks: list[ClusterCheck]) -> list[ClusterCheck]:
        """Per check, the first non-ok result across hosts."""
        picked: dict[str, ClusterCheck] = {}
        for check in checks:
            if check.id not in picked or (picked[check.id].status == "ok" and check.status != "ok"):
                picked[check.id] = check
        return list(picked.values())

    def node(self, state: ClusterState, hosts: list[ClusterHost] | None = None) -> MapNode:
        """`hosts`: nodes drawn (default all). None drawn: green."""
        hosts = state.hosts if hosts is None else hosts
        reachable = [h for h in hosts if h.reachable]
        rows = self._worst_checks([c for h in reachable for c in h.checks if c.id in STORAGE_CHECKS])
        status = "up" if not hosts else "warn" if not rows or any(c.status != "ok" for c in rows) else "up"
        paths = state.paths
        details = {
            "Models folder": paths.models_path or "Matricxon's default",
            "Chat caches folder": paths.prompt_cache_path or "Matricxon's default",
        }
        problem = next(
            ((h.host, c) for h in reachable for c in h.checks if c.id in STORAGE_CHECKS and c.status != "ok"), None
        )
        if problem:  # said on the node itself
            details["Needs attention"] = f"{problem[0]}: {problem[1].what}" + (
                f" ({problem[1].hint})" if problem[1].hint else ""
            )
        elif hosts and not reachable:
            details["Needs attention"] = "no node answers, so the folders cannot be checked"
        return MapNode(
            id="storage",
            kind="storage",
            name="Shared storage",
            subtitle=_short_text(problem[1].what) if problem else "models + chat caches",
            status=status,
            details=details,
            checks=rows[:4],
        )
