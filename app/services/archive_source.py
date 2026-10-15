"""A plain directory (no git) as an update source: download the branch archive from GitHub and unpack it over
the install. `preserve` names top-level entries (user data, .venv, .env) that are never overwritten. The
installed commit is kept in a marker file so the next check can tell whether an update is out.
"""

import asyncio
import shutil
import tarfile
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

MARKER = ".source_commit"
_API_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
_DOWNLOAD_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_REPORT_EVERY = 256 * 1024


class ArchiveSource:
    def __init__(self, path: Path, repo: str, branch: str, preserve: frozenset[str], proxy_url: str | None) -> None:
        self.path = path
        self.repo = repo
        self.branch = branch
        self.preserve = preserve | {MARKER, ".git"}
        self._proxy_url = proxy_url

    def current(self) -> str | None:
        marker = self.path / MARKER
        return (marker.read_text().strip() or None) if marker.is_file() else None

    async def latest(self) -> str:
        """Newest commit sha (short) on the branch; httpx.HTTPError on failure."""
        async with httpx.AsyncClient(timeout=_API_TIMEOUT, proxy=self._proxy_url) as client:
            resp = await client.get(
                f"https://api.github.com/repos/{self.repo}/commits/{self.branch}",
                headers={"Accept": "application/vnd.github.sha"},
            )
            resp.raise_for_status()
            return resp.text.strip()[:7]

    async def download_stream(self, dest: Path) -> AsyncIterator[dict]:
        """Saves the branch tarball to `dest`, yielding byte progress (total is often unknown)."""
        url = f"https://codeload.github.com/{self.repo}/tar.gz/{self.branch}"
        async with httpx.AsyncClient(timeout=_DOWNLOAD_TIMEOUT, follow_redirects=True, proxy=self._proxy_url) as client:
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0)) or None
                completed = reported = 0
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=64 * 1024):
                        f.write(chunk)
                        completed += len(chunk)
                        if completed - reported >= _REPORT_EVERY:
                            reported = completed
                            yield {"completed": completed, "total": total, "unit": "bytes"}
                yield {"completed": completed, "total": total or completed, "unit": "bytes"}

    async def install(self, archive: Path, commit: str | None) -> int:
        """Unpacks `archive` over the install; returns the number of files written."""
        return await asyncio.to_thread(self._install, archive, commit)

    def _install(self, archive: Path, commit: str | None) -> int:
        with tempfile.TemporaryDirectory(prefix="pairing-update-") as tmp:
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp, filter="data")
            roots = list(Path(tmp).iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise RuntimeError("Unexpected archive layout.")
            written = 0
            for entry in roots[0].iterdir():
                if entry.name in self.preserve:
                    continue
                target = self.path / entry.name
                if entry.is_dir():
                    shutil.copytree(entry, target, dirs_exist_ok=True)
                    written += sum(1 for p in entry.rglob("*") if p.is_file())
                else:
                    shutil.copy2(entry, target)
                    written += 1
        if commit:
            (self.path / MARKER).write_text(commit + "\n")
        return written
