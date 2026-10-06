"""Builds stable-diffusion.cpp from source — the "Build from source" choice of Settings > Image's Install (Linux).
Clones the repo's default branch (or the admin's install_version tag/branch) into <install>/src, compiles
`sd-server` with CMake for this machine's own CPU in <install>/build, and swaps it into the install folder. Needs git,
cmake and a C++ compiler on the PATH (never installed by pAIring).

The source and the build tree are kept, so the code can be read or patched and the next build only recompiles what
changed (an update fetches into the same clone, and refuses if tracked files were edited). The "models" folder,
src/ and build/ are never touched by an install swap."""

import asyncio
import logging
import os
import re
import shutil
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

from app.config import SDCPP_GITHUB_REPO

logger = logging.getLogger("llama_chat")

_TOTAL_STEPS = 4
_KEEP = {"models", "src", "src.partial", "build"}  # what an install swap leaves alone
_PERCENT_RE = re.compile(r"^\[\s*(\d+)%\]\s*(.*)")
_TAIL_LINES = 15
_building = False  # one build at a time: a second would fight the first over src/ and build/


class BuildError(RuntimeError):
    """A build command failed; the message carries the tail of its output."""


def missing_tools() -> list[str]:
    """Names of the build tools that aren't on the PATH."""
    missing = [tool for tool in ("git", "cmake") if shutil.which(tool) is None]
    if shutil.which("c++") is None and shutil.which("g++") is None:
        missing.append("a C++ compiler (g++)")
    return missing


def _jobs() -> int:
    """Half the cores: a full build is long, and a laptop CPU should stay cool enough to keep working."""
    return max(1, (os.cpu_count() or 2) // 2)


def _env(proxy_url: str | None) -> dict[str, str]:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if proxy_url:
        env.update(HTTP_PROXY=proxy_url, HTTPS_PROXY=proxy_url)
    return env


async def _stream_command(argv: list[str], cwd: Path | None, env: dict[str, str]) -> AsyncIterator[str]:
    """Yields each output line (stderr merged in) of a low-priority subprocess; BuildError on a non-zero exit. The
    process is killed if the consumer goes away (the admin closed the page)."""
    if shutil.which("nice"):
        argv = ["nice", "-n", "10", *argv]
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    tail: list[str] = []
    try:
        async for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            tail = [*tail, line][-_TAIL_LINES:]
            yield line
        if await proc.wait() != 0:
            raise BuildError(f"`{' '.join(argv[-3:])}` failed (exit {proc.returncode}):\n" + "\n".join(tail))
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()


async def _run(argv: list[str], cwd: Path | None, env: dict[str, str]) -> str:
    """Runs a command to completion and returns its last output line."""
    last = ""
    async for line in _stream_command(argv, cwd, env):
        last = line or last
    return last


async def _output(argv: list[str], cwd: Path | None, env: dict[str, str]) -> str:
    """Runs a command to completion and returns all of its output."""
    return "\n".join([line async for line in _stream_command(argv, cwd, env)])


async def _prepare_source(repo: str, ref: str | None, src: Path, env: dict[str, str]) -> None:
    """Makes `src` hold the wanted revision: an update of the kept clone, or a fresh one (cloned beside it and renamed
    into place, so an interrupted clone never leaves a half repo behind)."""
    if (src / ".git").is_dir():
        if (await _output(["git", "status", "--porcelain", "--untracked-files=no"], src, env)).strip():
            raise BuildError(f"{src} has local changes - commit or discard them, or delete the folder to re-clone it.")
        await _run(["git", "fetch", "--depth", "1", "origin", ref or "HEAD"], src, env)
        await _run(["git", "reset", "--hard", "FETCH_HEAD"], src, env)
        await _run(["git", "submodule", "update", "--init", "--recursive", "--depth", "1"], src, env)
        return
    partial = src.with_name("src.partial")
    shutil.rmtree(partial, ignore_errors=True)
    shutil.rmtree(src, ignore_errors=True)  # a leftover that is not a repo
    clone = ["git", "clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules"]
    await _run([*clone, *(["--branch", ref] if ref else []), f"https://github.com/{repo}", str(partial)], None, env)
    partial.rename(src)


def _swap_in(built: Path, target_dir: Path) -> None:
    """Replaces the old engine files in `target_dir` (all but models/, src/ and build/) with the build output."""
    for child in target_dir.iterdir():
        if child.name in _KEEP:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for name in ("sd-server", "sd-cli"):
        if (built / name).exists():
            shutil.copy2(built / name, target_dir / name)
            (target_dir / name).chmod(0o755)
    for lib in built.glob("*.so*"):  # present only for a shared-library build
        shutil.copy2(lib, target_dir / lib.name)


async def build_stream(
    repo: str | None, ref: str | None, proxy_url: str | None, target_dir: Path
) -> AsyncIterator[dict]:
    """Same event shape as the other installers: progress events, then {"done": True, ...} or {"error": ...}."""
    repo = repo or SDCPP_GITHUB_REPO
    if missing := missing_tools():
        yield {
            "error": f"Building from source needs {', '.join(missing)}. On Debian/Ubuntu: "
            "sudo apt install git cmake build-essential"
        }
        return

    global _building
    if _building:
        yield {"error": "A stable-diffusion.cpp build is already running - wait for it to finish."}
        return
    _building = True
    env = _env(proxy_url)
    src, build = target_dir / "src", target_dir / "build"
    target_dir.mkdir(parents=True, exist_ok=True)

    def meta(step: int, label: str) -> dict:
        return {"step": step, "total_steps": _TOTAL_STEPS, "step_label": label}

    try:
        updating = (src / ".git").is_dir()
        verb = "Updating the source of" if updating else "Cloning"
        yield {
            **meta(1, "Updating" if updating else "Cloning"),
            "status": f"{verb} {repo}{f' ({ref})' if ref else ''}...",
        }
        await _prepare_source(repo, ref, src, env)
        commit = await _run(["git", "rev-parse", "--short", "HEAD"], src, env)

        yield {**meta(2, "Configuring"), "status": "Configuring the build..."}
        await _run(["cmake", "-S", str(src), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release"], None, env)

        yield {**meta(3, "Compiling"), "status": f"Compiling with {_jobs()} threads - this can take a while..."}
        last_pct = -1
        argv = ["cmake", "--build", str(build), "--target", "sd-server", "--parallel", str(_jobs())]
        async for line in _stream_command(argv, None, env):
            if (m := _PERCENT_RE.match(line)) and int(m[1]) != last_pct:
                last_pct = int(m[1])
                yield {**meta(3, "Compiling"), "status": m[2], "completed": last_pct, "total": 100, "unit": "percent"}

        yield {**meta(4, "Installing"), "status": "Installing..."}
        built = build / "bin"
        if not (built / "sd-server").exists():
            yield {"error": "The build finished but no sd-server binary was produced."}
            return
        _swap_in(built, target_dir)
        (target_dir / "models").mkdir(exist_ok=True)
        (target_dir / ".install_complete").write_text(commit)
    except (BuildError, OSError) as exc:
        yield {"error": f"Building stable-diffusion.cpp failed: {exc}"}
        return
    finally:
        _building = False

    logger.info("Built stable-diffusion.cpp %s from source into %s", commit, target_dir)
    yield {
        "done": True,
        "step": _TOTAL_STEPS,
        "total_steps": _TOTAL_STEPS,
        "binary_path": str(target_dir / "sd-server"),
        "build": "source",
        "version": commit,
        "models_dir": str(target_dir / "models"),
        "source_dir": str(src),
    }
