"""Process primitives for the stable-diffusion.cpp `sd-server` subprocess — tracking file + spawn. Liveness and
termination are reused from comfyui_process (they only need a pid and a path that appears in its cmdline)."""

import json
import logging
import shlex
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

from app.config import DATA_DIR, SDCPP_HOST
from app.services.comfyui_process import is_alive, terminate

__all__ = ["is_alive", "terminate", "read_tracking", "write_tracking", "spawn"]

logger = logging.getLogger("llama_chat")

_TRACKING_FILE = DATA_DIR / "sdcpp.json"


def read_tracking() -> dict | None:
    if not _TRACKING_FILE.exists():
        return None
    try:
        return json.loads(_TRACKING_FILE.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("sdcpp.json unreadable (%s) — treating as not running.", exc)
        return None


def write_tracking(info: dict | None) -> None:
    if info is None:
        _TRACKING_FILE.unlink(missing_ok=True)
    else:
        _TRACKING_FILE.write_text(json.dumps(info))


def build_argv(binary_path: str, model_path: str, extra_args: str | None) -> list[str]:
    """Listens on SDCPP_HOST's port (loopback only unless extra_args overrides -l)."""
    port = urlparse(SDCPP_HOST).port or 8189
    return [
        binary_path,
        "-l",
        "127.0.0.1",
        "--listen-port",
        str(port),
        "-m",
        model_path,
        *shlex.split(extra_args or ""),
    ]


def spawn(binary_path: str, model_path: str, extra_args: str | None) -> int | None:
    """Detached launch, cwd = the binary's own dir (its shared libs sit beside it). None if it fails or exits
    immediately (bad path/model, port in use) — see data/logs/sdcpp.log."""
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sdcpp.log"
    log_file = open(log_path, "ab")
    try:
        proc = subprocess.Popen(
            build_argv(binary_path, model_path, extra_args),
            cwd=str(Path(binary_path).parent),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        logger.warning("Could not launch sd-server (%s) — check the configured paths.", exc)
        return None
    time.sleep(1.5)
    if proc.poll() is not None:
        logger.warning("sd-server exited immediately (code %s) — check %s", proc.returncode, log_path)
        return None
    logger.info("Spawned sd-server (pid %d).", proc.pid)
    return proc.pid
