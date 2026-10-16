"""A git checkout as an update source: fetch, compare with upstream, fast-forward."""

import re
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from app.services.update_process import UpdateProcess

_PROGRESS = re.compile(r"^(Counting objects|Compressing objects|Receiving objects|Resolving deltas):\s+(\d+)%")
_MAX_CHANGES = 20
# Fetch GitHub over HTTPS even when the remote is SSH: no keys needed, and the proxy applies. Remotes stay as they are.
_HTTPS_FETCH = (
    "-c",
    "url.https://github.com/.insteadOf=git@github.com:",
    "-c",
    "url.https://github.com/.insteadOf=ssh://git@github.com/",
)


class GitSource:
    def __init__(self, path: Path, process: UpdateProcess) -> None:
        self.path = path
        self._process = process

    @staticmethod
    def usable(path: Path) -> bool:
        """git is on PATH and `path` is a checkout."""
        return shutil.which("git") is not None and (path / ".git").exists()

    async def _git(self, *args: str, timeout: float = 60.0) -> str:
        return await self._process.output(["git", *args], cwd=self.path, timeout=timeout)

    async def head(self) -> str:
        return await self._git("rev-parse", "--short", "HEAD")

    async def upstream(self) -> str:
        """Short sha of the tracked branch; RuntimeError if none is set."""
        return await self._git("rev-parse", "--short", "@{u}")

    async def fetch(self) -> None:
        await self._git(*_HTTPS_FETCH, "fetch", "--quiet", timeout=45.0)

    async def counts(self) -> tuple[int, int]:
        """(behind, ahead) relative to upstream."""
        out = await self._git("rev-list", "--left-right", "--count", "@{u}...HEAD")
        behind, ahead = out.split()
        return int(behind), int(ahead)

    async def pending_changes(self) -> list[str]:
        out = await self._git("log", "--oneline", f"-n{_MAX_CHANGES}", "HEAD..@{u}")
        return out.splitlines()

    async def changed_files(self, old: str, new: str) -> list[str]:
        return (await self._git("diff", "--name-only", old, new)).splitlines()

    async def fetch_stream(self) -> AsyncIterator[dict]:
        """git fetch with percent events; other lines as status."""
        async for line in self._process.stream(["git", *_HTTPS_FETCH, "fetch", "--progress"], cwd=self.path):
            match = _PROGRESS.match(line)
            if match:
                yield {"activity": match[1].lower(), "completed": int(match[2]), "total": 100, "unit": "percent"}
            else:
                yield {"status": line}

    async def fast_forward_stream(self) -> AsyncIterator[str]:
        async for line in self._process.stream(["git", "merge", "--ff-only", "@{u}"], cwd=self.path):
            yield line
