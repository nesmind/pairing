"""Entrypoint: `python run.py` starts the web server. Separate from app/main.py so `app.main` stays importable
(tests, `uvicorn app.main:app`) without starting a server.
"""

import os
from pathlib import Path

import setproctitle
import uvicorn

from app.config import APP_HOST, APP_PORT, INSTANCE_INDEX
from app.pid_file import PidFile

if __name__ == "__main__":
    # Every launch path (scripts/start.sh, the Dockerfile, the systemd unit from scripts/install_on_fresh_server.sh)
    # runs through this file, so setting the title here covers them all without firing on `import app.main` (tests,
    # `uvicorn app.main:app`). Otherwise ps/top/ss -p show a generic "python".
    #
    # A sibling spawned by instance_process.py gets an index-suffixed title; the kernel's 15-byte comm
    # truncates "pAIring-server-N" to "pAIring-server-" (scripts/stop.sh), so this affects what `ps`/`htop` show,
    # not exact-name matching.
    proctitle = "pAIring-server" if INSTANCE_INDEX == 0 else f"pAIring-server-{INSTANCE_INDEX}"
    setproctitle.setproctitle(proctitle)
    # Only the primary writes one; it stops its siblings itself on shutdown (scripts/stop.sh). start.sh sets
    # PAIRING_PID_FILE to an absolute path; the default matches.
    pid_file = PidFile(Path(os.environ.get("PAIRING_PID_FILE", "run/pairing.pid")))
    if INSTANCE_INDEX == 0:
        pid_file.write()
    try:
        uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=False)
    finally:
        if INSTANCE_INDEX == 0:
            pid_file.remove()
