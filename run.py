"""
Entrypoint script: `python run.py` starts the web server.

Kept separate from app/main.py so `app.main` stays importable (e.g. by
tests, or by a production ASGI server like `uvicorn app.main:app`)
without also starting a server as a side effect of importing it.
"""

import setproctitle
import uvicorn

from app.config import APP_HOST, APP_PORT, INSTANCE_INDEX

if __name__ == "__main__":
    # Every launch path this project actually uses (scripts/start.sh,
    # the Dockerfile, the systemd unit scripts/install_on_fresh_server.sh
    # installs) runs through this exact file, so setting it here once —
    # rather than in app/main.py — covers all of them without also
    # firing as a side effect of just importing app.main (tests,
    # `uvicorn app.main:app` run directly). Without this, every tool
    # that reads a process's kernel-level name (ps, top, ss -p, ...)
    # shows the generic "python" — indistinguishable from any other
    # Python script running on the same machine.
    #
    # A sibling spawned by app/services/instance_process.py gets its own
    # index-suffixed title — note the kernel's 15-byte comm field
    # truncates "pAIring-server-N" to "pAIring-server-" regardless of N
    # (see scripts/stop.sh's own comment on this), so this only ever
    # matters for what plain `ps`/`htop` show, not for exact-name
    # process matching.
    proctitle = "pAIring-server" if INSTANCE_INDEX == 0 else f"pAIring-server-{INSTANCE_INDEX}"
    setproctitle.setproctitle(proctitle)
    uvicorn.run("app.main:app", host=APP_HOST, port=APP_PORT, reload=False)
