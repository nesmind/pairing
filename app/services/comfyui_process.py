"""Process primitives for supervising a ComfyUI subprocess: spawn, terminate, liveness and PID persistence.
Like instance_process.py (primitives here, orchestration and locking in comfyui_service.py), but for an
executable whose source this app doesn't own; see is_alive for where that matters.
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
    """{"pid": int} for the ComfyUI this app last started, or None if never started or the tracking file is
    missing/corrupt (tolerant like instance_process.read_tracking).
    """
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
    """True iff `pid` runs *and* still looks like the ComfyUI started at `main_py_path`; never a bare
    os.kill(pid, 0) (as instance_process.is_alive).

    Reads /proc/{pid}/cmdline, not comm: ComfyUI's comm is just "python"/"python3", indistinguishable from
    other Python processes. cmdline works because this app controls the invocation (see spawn) and ComfyUI
    doesn't call setproctitle, which this app's run.py does (hence instance_process.is_alive uses comm).
    Do not "fix" this to match instance_process.py; the constraints are opposite.
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
    """Launches ComfyUI detached, in its own install directory (it needs its own root for relative asset
    paths, unlike instance_process.spawn_sibling). `extra_args` is one raw string
    (e.g. "--listen 0.0.0.0 --port 8188"); this app doesn't model its CLI. start_new_session=True lets it
    survive an unclean exit of this app. Returns None (and logs) if it exits immediately (bad path, port in use).
    """
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
        # A nonexistent python_path/main_py_path (likely a typo) raises from Popen before any poll; handle it
        # like an immediate exit (return None, log) so the result is "not running", not a 500.
        logger.warning("Could not launch ComfyUI (%s) — check the configured paths.", exc)
        return None
    time.sleep(1.0)
    if proc.poll() is not None:
        logger.warning("ComfyUI exited immediately (code %s) — check %s", proc.returncode, log_path)
        return None
    logger.info("Spawned ComfyUI (pid %d).", proc.pid)
    return proc.pid


def terminate(pid: int, main_py_path: str) -> None:
    """SIGTERM, a grace period, then SIGKILL (like instance_process.terminate_sibling). Grace is 5s vs 3s:
    ComfyUI may be mid-generation or writing a file.
    """
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
