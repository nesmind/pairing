#!/usr/bin/env bash
#
# Starts the pAIring app itself — the thing a machine restart stops,
# since it doesn't run as a system service on this machine yet (see
# scripts/install_on_fresh_server.sh for the systemd-service version of
# this, meant for a fresh server rather than an existing dev machine).
#
# Safe to re-run: skips starting a second pAIring if one's already up,
# rather than launching a duplicate.
#
# Usage: bash scripts/start.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/data/logs"
mkdir -p "$LOG_DIR"

echo "pAIring directory: $APP_DIR"
echo

if pgrep -x "pAIring-server" > /dev/null 2>&1; then
    echo "✓ pAIring is already running."
else
    echo "Starting pAIring..."
    # `cd` first: app/main.py resolves its own paths from BASE_DIR now,
    # so this isn't load-bearing for correctness the way it used to be,
    # but costs nothing and matches how the app is normally run.
    (cd "$APP_DIR" && nohup "$APP_DIR/.venv/bin/python" "$APP_DIR/run.py" > "$LOG_DIR/app.log" 2>&1 &)
    disown 2>/dev/null || true
    sleep 3
    # Matched by exact process *name* ("pAIring-server"), not by
    # grepping the command line for run.py's path — run.py renames the
    # process via setproctitle on startup (so ps/top/ss show
    # "pAIring-server" instead of the generic "python"), and this
    # project's own directory being named "pAIring" means a path-based
    # match would also catch unrelated processes whose command line
    # happens to mention that
    
    if pgrep -x "pAIring-server" > /dev/null 2>&1; then
        echo "✓ pAIring started (log: $LOG_DIR/app.log)"
    else
        echo "✗ pAIring failed to start — check $LOG_DIR/app.log"
        exit 1
    fi
fi

APP_PORT="$(grep -E '^APP_PORT=' "$APP_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
APP_PORT="${APP_PORT:-8000}"

echo
echo "Web UI: http://localhost:$APP_PORT"
echo "Check status any time with: bash scripts/status.sh"

