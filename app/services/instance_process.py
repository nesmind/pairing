"""Process primitives for "local instances": spawn, terminate and liveness-check a sibling process and persist
which exist. Split from instance_service.py (async orchestration) for the file-size rule, like db_config_url.py.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.config import APP_PORT, BASE_DIR, DATA_DIR, INSTANCE_INDEX

logger = logging.getLogger("llama_chat")

_TRACKING_FILE = DATA_DIR / "instances.json"
_TERMINATE_GRACE_SECONDS = 3.0


def port_for_index(index: int) -> int:
    """The port a local instance listens on. `APP_PORT` is *this process's* port (a sibling's is already offset by
    its index, see spawn_sibling), so adding `index` double-counts. Subtracting this process's INSTANCE_INDEX
    recovers the base (index 0's) port from any instance. Without it a non-primary's broadcast_refresh
    (server_pool_broadcast) POSTed to its own port, leaving the primary's caches (e.g. engine_service)
    stale: the "switching engine doesn't take effect" bug.
    """
    return (APP_PORT - INSTANCE_INDEX) + index


def plan_changes(desired_count: int, alive_indices: set[int]) -> tuple[set[int], set[int]]:
    """Pure decision logic: given the wanted count and the sibling indices (1-based; 0 is the primary) alive,
    returns (indices_to_spawn, indices_to_terminate). Separate from side effects so it's easy to unit-test.
    """
    wanted = set(range(1, desired_count))
    return wanted - alive_indices, alive_indices - wanted


def read_tracking() -> dict[int, dict]:
    """{index: {"pid": int}} for each sibling this primary recorded spawning. A missing or corrupt file means
    "nothing tracked" (self-healing, like note_service's default-notes backfill).
    """
    if not _TRACKING_FILE.exists():
        return {}
    try:
        raw = json.loads(_TRACKING_FILE.read_text())
        return {int(k): v for k, v in raw.items()}
    except (OSError, ValueError) as exc:
        logger.warning("instances.json unreadable (%s) — treating as empty.", exc)
        return {}


def write_tracking(tracked: dict[int, dict]) -> None:
    _TRACKING_FILE.write_text(json.dumps({str(k): v for k, v in tracked.items()}))


def is_alive(pid: int) -> bool:
    """True iff `pid` runs *and* looks like one of our own run.py processes; never a bare os.kill(pid, 0), which
    can't tell a live sibling from an unrelated process reusing the PID.

    Reaps first (os.waitpid WNOHANG) so a dead child doesn't zombie if we're its parent. ChildProcessError
    means we aren't (e.g. the primary restarted and the sibling was reparented to init): fall back to
    /proc/{pid}/comm, which works regardless.

    Reads comm, not cmdline: run.py calls setproctitle very early, which rewrites argv memory, so cmdline
    no longer shows "run.py" (this broke a restart-adoption test). comm is capped at 15 bytes (TASK_COMM_LEN),
    so "pAIring-server-N" truncates to "pAIring-server-" (see scripts/stop.sh): enough to prove identity; the
    index comes from the tracking file's key.
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
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return False
    return comm.startswith("pAIring-server")


def spawn_sibling(index: int) -> int | None:
    """Launches sibling `index` detached, sharing this machine's DB/models/knowledge (same env and cwd); only
    APP_PORT/AUTO_MIGRATE/PAIRING_INSTANCE_INDEX differ. AUTO_MIGRATE=false reuses the multi-instance safety net
    (app.config.AUTO_MIGRATE): the primary migrated, and a sibling racing to migrate is the failure it avoids.

    start_new_session=True detaches it (like scripts/start.sh's `nohup... & disown`) so it survives the primary
    dying uncleanly. Returns None (and logs) if it exits immediately (e.g. port in use).
    """
    port = port_for_index(index)
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"app-instance-{index}.log"
    log_file = open(log_path, "ab")
    env = {**os.environ, "APP_PORT": str(port), "AUTO_MIGRATE": "false", "PAIRING_INSTANCE_INDEX": str(index)}
    proc = subprocess.Popen(
        [sys.executable, str(BASE_DIR / "run.py")],
        env=env,
        cwd=str(BASE_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(1.0)
    if proc.poll() is not None:
        logger.warning(
            "Instance %d (port %d) exited immediately (code %s) — check %s",
            index,
            port,
            proc.returncode,
            log_path,
        )
        return None
    logger.info("Spawned instance %d on port %d (pid %d).", index, port, proc.pid)
    return proc.pid


def terminate_sibling(pid: int) -> None:
    """SIGTERM, a brief grace period, then SIGKILL (like scripts/stop.sh's pkill). No-op if already gone."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + _TERMINATE_GRACE_SECONDS
    while time.monotonic() < deadline:
        if not is_alive(pid):
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
