"""Tests for services/git_source.py: GitHub fetches go over HTTPS even when the remote is SSH."""

import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from app.services.git_source import _HTTPS_FETCH, GitSource
from app.services.update_process import UpdateProcess


class _RecordingProcess(UpdateProcess):
    """Records argv instead of running it."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    async def output(self, argv: list[str], cwd: Path | None = None, timeout: float = 60.0) -> str:
        self.calls.append(argv)
        return ""

    async def stream(self, argv: list[str], cwd: Path | None = None) -> AsyncIterator[str]:
        self.calls.append(argv)
        for line in ():
            yield line


@pytest.mark.asyncio
async def test_both_fetch_paths_rewrite_github_ssh_to_https(tmp_path: Path) -> None:
    process = _RecordingProcess()
    source = GitSource(tmp_path, process)
    await source.fetch()
    async for _ in source.fetch_stream():
        pass
    for argv in process.calls:
        assert argv[: 1 + len(_HTTPS_FETCH)] == ["git", *_HTTPS_FETCH]
        assert "fetch" in argv


@pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
@pytest.mark.parametrize("remote", ["git@github.com:org/repo.git", "ssh://git@github.com/org/repo.git"])
def test_the_rewrite_turns_an_ssh_remote_into_https(tmp_path: Path, remote: str) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "remote", "add", "origin", remote], cwd=tmp_path, check=True)
    url = subprocess.run(
        ["git", *_HTTPS_FETCH, "ls-remote", "--get-url", "origin"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert url == "https://github.com/org/repo.git"
    # The stored remote itself is left unchanged.
    stored = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=tmp_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    assert stored == remote
