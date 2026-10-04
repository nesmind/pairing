"""The primary server's own PID file, the single source of truth for scripts/start.sh, stop.sh and status.sh.

Self-written by the process (os.getpid()), not captured by the launching shell, so it is right however the
process was started (background job, setsid, supervisor, container). Same idea as Matricxon's PidFile."""

import os
from pathlib import Path


class PidFile:
    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(str(os.getpid()))

    def remove(self) -> None:
        self._path.unlink(missing_ok=True)
