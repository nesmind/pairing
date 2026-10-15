"""Tests for services/source_updater.py, git_source.py and archive_source.py: a real local git "remote" in tmp_path
and a hand-built tarball, so no network is used.
"""

import io
import subprocess
import tarfile
from pathlib import Path

import pytest

from app.services.archive_source import MARKER, ArchiveSource
from app.services.source_updater import SourceUpdater, UpdateTarget


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, name: str, text: str) -> None:
    (repo / name).write_text(text)
    _git(repo, "add", name)
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f"edit {name}")


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, Path]:
    """(upstream working repo, a clone of it that tracks main)."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _git(upstream, "init", "-q", "-b", "main")
    _commit(upstream, "app.py", "v1")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(upstream), str(clone))
    return upstream, clone


def _updater(path: Path, preserve: frozenset[str] = frozenset()) -> SourceUpdater:
    return SourceUpdater(UpdateTarget("Test", path, "o/r", "main", None, preserve))


async def _events(updater: SourceUpdater) -> list[dict]:
    return [event async for event in updater.update_stream()]


@pytest.mark.asyncio
async def test_git_status_reports_pending_changes(checkout):
    upstream, clone = checkout
    _commit(upstream, "app.py", "v2")

    status = await _updater(clone).status()

    assert status.method == "git"
    assert status.update_available is True
    assert status.behind == 1
    assert status.changes and "edit app.py" in status.changes[0]
    assert status.available is True


@pytest.mark.asyncio
async def test_git_status_allows_a_checkout_with_local_changes(checkout):
    _, clone = checkout
    (clone / "app.py").write_text("local edit")

    status = await _updater(clone).status()

    assert status.available is True
    assert status.note is None


@pytest.mark.asyncio
async def test_git_update_fast_forwards_and_skips_unchanged_requirements(checkout):
    upstream, clone = checkout
    _commit(upstream, "app.py", "v2")

    events = await _events(_updater(clone))

    assert (clone / "app.py").read_text() == "v2"
    assert events[-1]["done"] is True and events[-1]["changed"] is True
    assert any(e.get("status") == "Dependencies unchanged." for e in events)
    assert all(e.get("total_steps") == 3 for e in events[:-1])


@pytest.mark.asyncio
async def test_git_update_reports_git_error_when_local_changes_collide(checkout):
    upstream, clone = checkout
    _commit(upstream, "app.py", "v2")
    (clone / "app.py").write_text("local edit")

    events = await _events(_updater(clone))

    assert "error" in events[-1]
    assert (clone / "app.py").read_text() == "local edit"


def _tarball(tmp_path: Path, files: dict[str, str]) -> Path:
    archive = tmp_path / "src.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, text in files.items():
            data = text.encode()
            info = tarfile.TarInfo(f"repo-main/{name}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


@pytest.mark.asyncio
async def test_archive_install_overlays_files_and_keeps_preserved_ones(tmp_path):
    install = tmp_path / "install"
    (install / "data").mkdir(parents=True)
    (install / "data" / "db.sqlite").write_text("mine")
    (install / "app.py").write_text("old")
    archive = _tarball(tmp_path, {"app.py": "new", "pkg/mod.py": "x", "data/db.sqlite": "theirs"})
    source = ArchiveSource(install, "o/r", "main", frozenset({"data"}), None)

    written = await source.install(archive, "abc1234")

    assert (install / "app.py").read_text() == "new"
    assert (install / "pkg" / "mod.py").read_text() == "x"
    assert (install / "data" / "db.sqlite").read_text() == "mine"
    assert (install / MARKER).read_text().strip() == "abc1234"
    assert source.current() == "abc1234"
    assert written == 2


@pytest.mark.asyncio
async def test_archive_status_compares_the_marker_with_github(tmp_path, monkeypatch):
    (tmp_path / MARKER).write_text("abc1234\n")

    async def fake_latest(_self):
        return "def5678"

    monkeypatch.setattr(ArchiveSource, "latest", fake_latest)
    status = await _updater(tmp_path).status()

    assert status.method == "download"
    assert (status.current, status.latest, status.update_available) == ("abc1234", "def5678", True)


@pytest.mark.asyncio
async def test_archive_status_without_a_marker_is_unknown(tmp_path, monkeypatch):
    async def fake_latest(_self):
        return "def5678"

    monkeypatch.setattr(ArchiveSource, "latest", fake_latest)
    status = await _updater(tmp_path).status()

    assert status.update_available is None
    assert status.note


@pytest.mark.asyncio
async def test_a_second_concurrent_update_is_refused(checkout):
    _, clone = checkout
    updater = _updater(clone)
    async with updater._lock:
        events = await _events(updater)
    assert "already running" in events[0]["error"]
