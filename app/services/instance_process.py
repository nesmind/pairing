"""
Process-level primitives for the "local instances" feature — spawning/
terminating/liveness-checking a sibling app process and persisting which
ones are currently known to exist. Split out of
app/services/instance_service.py (which does the async orchestration
around these) purely to stay under CLAUDE.md's file-size rule, the same
reasoning app/services/db_config_url.py was split from
db_config_service.py for.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from app.config import APP_PORT, BASE_DIR, DATA_DIR

logger = logging.getLogger("llama_chat")

_TRACKING_FILE = DATA_DIR / "instances.json"
_TERMINATE_GRACE_SECONDS = 3.0


def port_for_index(index: int) -> int:
    return APP_PORT + index


def plan_changes(desired_count: int, alive_indices: set[int]) -> tuple[set[int], set[int]]:
    """Pure decision logic, no I/O: given how many instances are wanted
    and which sibling indices (1-based; index 0 is the primary and is
    never included) are currently confirmed alive, returns
    (indices_to_spawn, indices_to_terminate). Kept separate from the
    subprocess/file/network side effects below so it's trivially unit-
    testable."""
    wanted = set(range(1, desired_count))
    return wanted - alive_indices, alive_indices - wanted


def read_tracking() -> dict[int, dict]:
    """Returns {index: {"pid": int}} for every sibling this primary has
    previously recorded spawning. Tolerant of a missing or corrupt file
    — treated as "nothing tracked yet" rather than raising, the same
    self-healing stance other startup code in this app takes toward
    state that should exist but might not (e.g. note_service's default-
    notes backfill)."""
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
    """True iff `pid` is still running *and* still looks like one of our
    own run.py processes — never a bare os.kill(pid, 0), which can't
    tell a live sibling from an unrelated process that happened to reuse
    the same PID after a reboot or long uptime.

    Tries reaping first (os.waitpid with WNOHANG): if this primary
    process is the actual parent (the common case — we spawned it this
    run), a dead child needs reaping or it zombies. ChildProcessError
    means we're not its parent (e.g. the primary itself restarted since
    spawning it, so the sibling was reparented to init) — falls back to
    a /proc/{pid}/comm check, which works regardless of parentage.

    Deliberately reads comm, not cmdline: run.py calls setproctitle
    *very* early, and setproctitle rewrites a process's argv memory too
    — by the time this ever runs, /proc/{pid}/cmdline no longer contains
    "run.py" (or anything resembling the original launch command) for
    any of our own processes, primary or sibling alike (confirmed live,
    not just in theory — this broke a real restart-adoption test).
    comm is capped at 15 bytes by the kernel (TASK_COMM_LEN), so every
    sibling's proctitle "pAIring-server-N" truncates to the identical
    "pAIring-server-" regardless of N (see scripts/stop.sh's own comment
    on this same fact) — enough to prove "this is one of our own
    processes," which is all an identity check here needs; the actual
    index is already known from the tracking file's own key.
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
    """Launches sibling `index` as a detached subprocess sharing this
    machine's DB/models/knowledge (same env, same working directory) —
    only APP_PORT/AUTO_MIGRATE/PAIRING_INSTANCE_INDEX differ.
    AUTO_MIGRATE=false reuses the existing multi-instance migration
    safety net (see app.config.AUTO_MIGRATE's own docstring): the
    primary already migrated, so a sibling racing to migrate the same
    database on its own cold start is exactly the failure mode that
    setting exists to avoid.

    start_new_session=True detaches the sibling from this process's own
    session/process group (the same effect scripts/start.sh gets from
    `nohup ... & disown`), so it keeps running even if this primary
    process dies uncleanly. Returns None (and logs) if the sibling
    appears to have exited immediately, e.g. a port already in use.
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
    """SIGTERM, then a brief grace period, then SIGKILL if it's still
    around — the same graceful-then-forceful shape scripts/stop.sh's
    pkill gives the primary. Silent no-op if the process is already
    gone."""
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
