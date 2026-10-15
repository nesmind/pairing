"""Subprocess helpers for Admin settings > Update (git, pip). Unlike the installers' line reader, output is split
on "\\r" too, so git's in-place progress lines arrive one by one.
"""

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path

_SPLIT = re.compile(rb"[\r\n]")


class UpdateProcess:
    """Runs one command for an updater, with git prompts off and the admin's proxy applied."""

    def __init__(self, proxy_url: str | None = None) -> None:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_SSH_COMMAND": "ssh -o BatchMode=yes"}
        if proxy_url:
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                env[key] = proxy_url
        self.env = env

    async def stream(self, argv: list[str], cwd: Path | None = None) -> AsyncIterator[str]:
        """Yields output lines as they come; RuntimeError on a non-zero exit."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        buffer = b""
        tail: list[str] = []
        try:
            while chunk := await process.stdout.read(4096):
                *lines, buffer = _SPLIT.split(buffer + chunk)
                for raw in lines:
                    if line := raw.decode(errors="replace").strip():
                        tail = [*tail[-4:], line]
                        yield line
            if line := buffer.decode(errors="replace").strip():
                tail = [*tail[-4:], line]
                yield line
            if await process.wait() != 0:
                raise RuntimeError(tail[-1] if tail else f"{argv[0]} exited with code {process.returncode}")
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def output(self, argv: list[str], cwd: Path | None = None, timeout: float = 60.0) -> str:
        """The command's stripped output; RuntimeError on failure or timeout."""
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd else None,
            env=self.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(process.communicate(), timeout)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise RuntimeError(f"{argv[0]} {argv[1]} timed out") from exc
        if process.returncode != 0:
            raise RuntimeError(err.decode(errors="replace").strip() or f"{argv[0]} {argv[1]} failed")
        return out.decode(errors="replace").strip()
