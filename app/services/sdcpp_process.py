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

__all__ = ["is_alive", "terminate", "read_tracking", "write_tracking", "spawn", "StartupError"]

logger = logging.getLogger("llama_chat")

_TRACKING_FILE = DATA_DIR / "sdcpp.json"
_STARTUP_WAIT_S = 30.0  # a big model can take a while to load before sd-server listens
_POLL_S = 0.25
_READY_MARKER = "listening on"
# sd-server log lines meaning the model file itself isn't understood (not a setup problem)
_UNSUPPORTED_MARKERS = ("not in model metadata", "model metadata validation failed")


class StartupError(ValueError):
    """sd-server exited while starting; the message is meant for the admin (router -> 400)."""


def _explain_exit(log_text: str) -> str:
    if any(marker in log_text for marker in _UNSUPPORTED_MARKERS):
        return (
            "This model is not supported by the installed stable-diffusion.cpp (its weights use a layout "
            "it can't read, e.g. an embedded tiny VAE). Choose another model."
        )
    return "sd-server exited while starting — see data/logs/sdcpp.log for the reason."


def _log_since(log_path: Path, offset: int) -> str:
    try:
        with log_path.open("rb") as handle:
            handle.seek(offset)
            return handle.read().decode(errors="replace")
    except OSError:
        return ""


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
    """Detached launch, cwd = the binary's own dir (its shared libs sit beside it). Blocking: waits until
    sd-server reports it is listening. None if it can't be launched; raises StartupError if it exits while
    starting (bad/unsupported model, port in use) — details in data/logs/sdcpp.log."""
    log_dir = DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sdcpp.log"
    offset = log_path.stat().st_size if log_path.exists() else 0
    with open(log_path, "ab") as log_file:
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
    deadline = time.monotonic() + _STARTUP_WAIT_S
    while time.monotonic() < deadline:
        time.sleep(_POLL_S)
        text = _log_since(log_path, offset)
        if proc.poll() is not None:
            logger.warning("sd-server exited while starting (code %s) — check %s", proc.returncode, log_path)
            raise StartupError(_explain_exit(text))
        if _READY_MARKER in text:
            break
    logger.info("Spawned sd-server (pid %d).", proc.pid)
    return proc.pid
