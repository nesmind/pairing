"""asyncio sibling of../matricxon/app/pull/download_coordinator.py's DownloadCoordinator, split from
matricxon_direct_puller for the file-size rule. Serializes concurrent downloads to the same destination:
a retry racing an orphaned pull of the same tag both streamed to one `.partial`, and the first to finish
renamed it from under the other. One `asyncio.Lock` per `dest` (different tags still run in
parallel), created lazily and dropped when idle, so it doesn't grow unbounded.
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
        """Blocks until no other caller is in `guard(dest)` for this `dest`, then holds it until the block exits.
        The caller decides what "already handled" means (MatricxonDirectPuller.pull_stream checks `dest.exists`).
        """
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


# One process-wide instance: every pull builds its own MatricxonDirectPuller
# (matricxon_admin.pull_model_stream), so same-tag races across requests need one shared registry.
_coordinator = DownloadCoordinator()


def get_download_coordinator() -> DownloadCoordinator:
    return _coordinator
