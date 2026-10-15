"""Checks and updates one project from GitHub: a git pull of just the changes when it's a checkout and git is
installed, otherwise a full archive download. Then installs requirements.txt into its venv (git: only when that
file changed). Events follow the installers' shape (step/total_steps/step_label, completed/total/unit, status,
then {"done": True} or {"error": ...}), so Settings draws the same progress bar.
"""

import asyncio
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.schemas.update import SourceUpdateStatus
from app.services.archive_source import ArchiveSource
from app.services.git_source import GitSource
from app.services.pip_progress import PipProgressTracker
from app.services.update_process import UpdateProcess


@dataclass(frozen=True)
class UpdateTarget:
    name: str
    path: Path
    repo: str
    branch: str
    python: Path | None
    preserve: frozenset[str]


class SourceUpdater:
    _locks: dict[Path, asyncio.Lock] = {}

    def __init__(self, target: UpdateTarget, proxy_url: str | None = None) -> None:
        self.target = target
        self._process = UpdateProcess(proxy_url)
        self._git = GitSource(target.path, self._process)
        self._archive = ArchiveSource(target.path, target.repo, target.branch, target.preserve, proxy_url)
        self._lock = self._locks.setdefault(target.path, asyncio.Lock())
        self.total_steps = 3

    @property
    def method(self) -> str:
        return "git" if GitSource.usable(self.target.path) else "download"

    async def status(self) -> SourceUpdateStatus:
        base = SourceUpdateStatus(name=self.target.name, path=str(self.target.path), method=self.method)
        try:
            return await (self._git_status(base) if base.method == "git" else self._archive_status(base))
        except (RuntimeError, httpx.HTTPError) as exc:
            return base.model_copy(update={"available": True, "note": f"Could not check for updates: {exc}"})

    async def _git_status(self, base: SourceUpdateStatus) -> SourceUpdateStatus:
        current = await self._git.head()
        try:
            await self._git.upstream()
        except RuntimeError:
            return base.model_copy(
                update={"current": current, "note": "This checkout has no upstream branch to update from."}
            )
        await self._git.fetch()
        latest = await self._git.upstream()
        behind, ahead = await self._git.counts()
        note = None
        available = True
        if ahead and behind:
            available, note = False, "This checkout has local commits that are not on GitHub, so it can't fast-forward."
        changes = await self._git.pending_changes() if behind else []
        return base.model_copy(
            update={
                "available": available,
                "current": current,
                "latest": latest,
                "behind": behind,
                "update_available": behind > 0,
                "changes": changes,
                "note": note,
            }
        )

    async def _archive_status(self, base: SourceUpdateStatus) -> SourceUpdateStatus:
        current = self._archive.current()
        latest = await self._archive.latest()
        note = None if current else "Installed version unknown; updating downloads the full latest version."
        return base.model_copy(
            update={
                "available": True,
                "current": current,
                "latest": latest,
                "update_available": None if current is None else current != latest,
                "note": note,
            }
        )

    def _meta(self, step: int, label: str) -> dict:
        return {"step": step, "total_steps": self.total_steps, "step_label": label}

    async def update_stream(self) -> AsyncIterator[dict]:
        if self._lock.locked():
            yield {"error": f"An update of {self.target.name} is already running."}
            return
        async with self._lock:
            try:
                stream = self._git_update() if self.method == "git" else self._archive_update()
                async for event in stream:
                    yield event
            except Exception as exc:  # noqa: BLE001 - every failure is shown to the admin
                yield {"error": f"Update failed: {exc}"}

    async def _git_update(self) -> AsyncIterator[dict]:
        before = await self._git.head()
        label = "Downloading changes"
        yield {**self._meta(1, label), "status": f"Fetching {self.target.name} changes from GitHub..."}
        async for event in self._git.fetch_stream():
            activity = event.pop("activity", None)
            yield {**self._meta(1, f"{label} — {activity}" if activity else label), **event}
        yield {**self._meta(2, "Applying changes"), "status": "Applying changes..."}
        async for line in self._git.fast_forward_stream():
            yield {**self._meta(2, "Applying changes"), "status": line}
        after = await self._git.head()
        changed = await self._git.changed_files(before, after) if before != after else []
        if "requirements.txt" in changed:
            async for event in self._pip():
                yield event
        else:
            yield {**self._meta(3, "Dependencies"), "status": "Dependencies unchanged."}
        yield {"done": True, "changed": before != after, "current": after, "files": len(changed)}

    async def _archive_update(self) -> AsyncIterator[dict]:
        latest = await self._archive.latest()
        label = "Downloading the latest version"
        yield {**self._meta(1, label), "status": f"Downloading {self.target.name} ({latest}) from GitHub..."}
        with tempfile.TemporaryDirectory(prefix="pairing-dl-") as tmp:
            archive = Path(tmp) / "source.tar.gz"
            async for event in self._archive.download_stream(archive):
                yield {**self._meta(1, label), **event}
            yield {**self._meta(2, "Unpacking"), "status": "Unpacking over the install (your data is kept)..."}
            written = await self._archive.install(archive, latest)
        yield {**self._meta(2, "Unpacking"), "status": f"Updated {written} files."}
        async for event in self._pip():
            yield event
        yield {"done": True, "changed": True, "current": latest, "files": written}

    async def _pip(self) -> AsyncIterator[dict]:
        python = self.target.python
        requirements = self.target.path / "requirements.txt"
        if python is None or not python.exists() or not requirements.is_file():
            yield {**self._meta(3, "Dependencies"), "status": "No Python environment found; skipped dependencies."}
            return
        meta = self._meta(3, "Installing dependencies")
        yield {**meta, "status": "Installing dependencies..."}
        tracker = PipProgressTracker("Installing dependencies")
        raw = ["--progress-bar", "raw"] if await PipProgressTracker.supports_raw(python) else []
        argv = [str(python), "-m", "pip", "install", *raw, "-r", str(requirements)]
        async for line in self._process.stream(argv, cwd=self.target.path):
            yield {**meta, **tracker.event_for(line)}
