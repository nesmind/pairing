"""
asyncio-native sibling of ../matricxon/app/pull/download_coordinator.py's DownloadCoordinator — split out of
app.services.matricxon_direct_puller purely for CLAUDE.md's file-size rule (that module's own real concern is
resolving/downloading/registering a model, not lock bookkeeping). Serializes concurrent downloads that
resolve to the same destination file: real bug found via a pAIring report (2026-09-21) — a retry racing an
orphaned earlier pull of the exact same tag both streamed to the same `.partial` path, and whichever finished
first renamed it away out from under the other. One `asyncio.Lock` per distinct `dest` (not one global lock —
two different tags must still download fully in parallel, only the same one needs to wait its turn), created
lazily and dropped again once nothing's waiting on it, so this never grows unbounded across a long-running
process pulling many different models over its lifetime.
"""

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path


class DownloadCoordinator:
    def __init__(self) -> None:
        self._locks: dict[Path, asyncio.Lock] = {}
        self._waiters: dict[Path, int] = defaultdict(int)

    @asynccontextmanager
    async def guard(self, dest: Path) -> AsyncIterator[None]:
        """Blocks until no other caller is inside `guard(dest)` for this exact `dest`, then holds it until
        the `with` block exits (success or exception) — the caller decides what "already handled by someone
        else" means once it has the lock (see MatricxonDirectPuller.pull_stream, which checks `dest.exists()`
        first)."""
        lock = self._locks.setdefault(dest, asyncio.Lock())
        self._waiters[dest] += 1
        try:
            async with lock:
                yield
        finally:
            self._waiters[dest] -= 1
            if self._waiters[dest] == 0:
                del self._locks[dest]
                del self._waiters[dest]


# One process-wide instance — every pull (each request builds its own MatricxonDirectPuller, see
# app.services.matricxon_admin.pull_model_stream) needs to coordinate against the same registry to actually
# catch a same-tag race between two separate requests.
_coordinator = DownloadCoordinator()


def get_download_coordinator() -> DownloadCoordinator:
    return _coordinator
