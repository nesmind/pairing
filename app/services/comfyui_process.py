"""
Process-level primitives for supervising a real ComfyUI subprocess —
spawning/terminating/liveness-checking it and persisting its PID. Mirrors
app/services/instance_process.py's split (pure process primitives here,
async orchestration/locking in app/services/comfyui_service.py), but for
a genuinely different executable this app doesn't own the source of,
unlike instance_process.py's own sibling-copies-of-run.py case — see
is_alive's own docstring for exactly where that difference matters.
"""

import json
import logging
import os
import shlex
import signal
import subprocess
import time
from pathlib import Path

from app.config import DATA_DIR

logger = logging.getLogger("llama_chat")

_TRACKING_FILE = DATA_DIR / "comfyui.json"
_TERMINATE_GRACE_SECONDS = 5.0


def read_tracking() -> dict | None:
    """Returns {"pid": int} for the ComfyUI process this app most
    recently started, or None if it's never been started (or the
    tracking file is missing/corrupt) — same tolerant-of-missing-file
    stance as instance_process.read_tracking."""
    if not _TRACKING_FILE.exists():
        return None
    try:
        return json.loads(_TRACKING_FILE.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("comfyui.json unreadable (%s) — treating as not running.", exc)
        return None


def write_tracking(info: dict | None) -> None:
    if info is None:
        _TRACKING_FILE.unlink(missing_ok=True)
    else:
        _TRACKING_FILE.write_text(json.dumps(info))


def is_alive(pid: int, main_py_path: str) -> bool:
    """True iff `pid` is still running *and* still looks like the
    ComfyUI process this app started at `main_py_path` — never a bare
    os.kill(pid, 0), for the same reason instance_process.is_alive isn't
    one either.

    Unlike instance_process.is_alive, this reads /proc/{pid}/cmdline, not
    comm: ComfyUI's own comm is just "python"/"python3" (kernel-truncated,
    indistinguishable from any other Python process on the box), so comm-
    matching can't identify it at all. cmdline works here specifically
    because this app controls the exact invocation (see spawn below) and
    ComfyUI itself never calls setproctitle to rewrite its own argv memory
    the way this app's own run.py does — that rewrite is exactly why
    instance_process.is_alive had to fall back to comm instead of cmdline
    for *that* process. Do not "fix" this to match instance_process.py's
    approach; the two processes have opposite constraints.
    """
    try:
        reaped_pid, _status = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
        if reaped_pid == 0:
            return True
    except ChildProcessError:
        pass
    except OSError:
        return False

    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False
    return main_py_path in cmdline


def spawn(python_path: str, main_py_path: str, extra_args: str | None) -> int | None:
    """Launches ComfyUI as a detached subprocess, cwd'd to its own
    install directory (unlike instance_process.spawn_sibling, which
    shares this app's own BASE_DIR — ComfyUI expects to run from its own
    root to find its own relative asset paths). `extra_args` is one raw
    string (e.g. "--listen 0.0.0.0 --port 8188") rather than a modeled
    set of flags — this app only needs to launch ComfyUI, not understand
    its CLI. start_new_session=True detaches it the same way
    spawn_sibling's does, so it survives this app process dying
    uncleanly. Returns None (and logs) if it appears to exit immediately
    — e.g. a bad path or the port already in use."""
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "comfyui.log"
    log_file = open(log_path, "ab")
    argv = [python_path, main_py_path, *shlex.split(extra_args or "")]
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(Path(main_py_path).parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        # A nonexistent python_path/main_py_path (a real, likely admin
        # typo) raises straight out of Popen itself, before there's ever
        # a process to poll() below — same "return None, log it, no
        # exception surfaces to the caller" contract as the immediate-
        # exit case, so a bad path degrades to a clean "not running"
        # instead of a raw 500.
        logger.warning("Could not launch ComfyUI (%s) — check the configured paths.", exc)
        return None
    time.sleep(1.0)
    if proc.poll() is not None:
        logger.warning("ComfyUI exited immediately (code %s) — check %s", proc.returncode, log_path)
        return None
    logger.info("Spawned ComfyUI (pid %d).", proc.pid)
    return proc.pid


def terminate(pid: int, main_py_path: str) -> None:
    """SIGTERM, then a grace period, then SIGKILL if still around — same
    shape as instance_process.terminate_sibling. A longer grace period
    than siblings get (5s vs 3s): ComfyUI may be mid-generation and
    benefits from a real chance to shut down cleanly rather than being
    killed while writing a file."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not is_alive(pid, main_py_path):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
