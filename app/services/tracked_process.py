"""Liveness and termination for a subprocess this app started and tracks by pid (stable-diffusion.cpp's sd-server,
sdcpp_process.py): a pid alone can be reused by an unrelated process, so each check also matches a path in its
command line."""

import os
import signal
import time
from pathlib import Path

_TERMINATE_GRACE_SECONDS = 5.0


def is_alive(pid: int, path: str) -> bool:
    """True iff `pid` runs *and* its command line still contains `path`; never a bare os.kill(pid, 0) (as
    instance_process.is_alive).

    Reads /proc/{pid}/cmdline, not comm: a generic program name isn't distinctive, while this app controls the
    invocation (and so the path in it). instance_process.is_alive uses comm instead, because run.py sets its own
    process title; the constraints are opposite.
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
    return path in cmdline


def terminate(pid: int, path: str) -> None:
    """SIGTERM, a grace period (it may be writing an image), then SIGKILL (like instance_process.terminate_sibling)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not is_alive(pid, path):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
