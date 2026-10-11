"""The primary server's PID file, the source of truth for scripts/start.sh, stop.sh and status.sh.

Written by the process itself (os.getpid), so it is right however it was launched. Same idea as Matricxon's PidFile.
"""

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
