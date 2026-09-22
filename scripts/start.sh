#!/usr/bin/env bash

# Usage: bash scripts/start.sh

set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$APP_DIR/data/logs"
mkdir -p "$LOG_DIR"

echo "pAIring directory: $APP_DIR"

if pgrep -x "pAIring-server" > /dev/null 2>&1; then
    echo "✓ pAIring is already running."
else
    echo "Starting pAIring..."

    LAUNCHER=(nohup)
    command -v setsid > /dev/null 2>&1 && LAUNCHER=(setsid)

    (cd "$APP_DIR" && "${LAUNCHER[@]}" "$APP_DIR/.venv/bin/python" -u "$APP_DIR/run.py" > "$LOG_DIR/app.log" 2>&1 < /dev/null &)
    disown 2>/dev/null || true
    sleep 3

    if pgrep -x "pAIring-server" > /dev/null 2>&1; then
        echo "✓ pAIring started (log: $LOG_DIR/app.log)"
    else
        echo "✗ pAIring failed to start — check $LOG_DIR/app.log"
        exit 1
    fi
fi

APP_PORT="$(grep -E '^APP_PORT=' "$APP_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
APP_PORT="${APP_PORT:-8000}"

echo "Web UI: http://localhost:$APP_PORT"
echo "Check status any time with: bash scripts/status.sh"
