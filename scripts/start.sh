#!/usr/bin/env bash
# Starts pAIring detached (works on Linux and macOS). The server writes its own PID file (run/pairing.pid),
# which stop.sh and status.sh read.
#
# Usage: bash scripts/start.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# Colored like uvicorn's own "INFO:" lines (plain text when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PID_FILE="${PAIRING_PID_FILE:-$APP_DIR/run/pairing.pid}"
LOG_DIR="$APP_DIR/data/logs"
LOG_FILE="$LOG_DIR/app.log"
PORT="$(grep -E '^APP_PORT=' "$APP_DIR/.env" 2>/dev/null | tail -1 | cut -d= -f2 || true)"
PORT="${PORT:-8000}"

mkdir -p "$(dirname "$PID_FILE")" "$LOG_DIR"

info "pAIring directory: $APP_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    info "pAIring is already running (pid $(cat "$PID_FILE"))"
    exit 1
fi

rm -f "$PID_FILE"

if pgrep -x "pAIring-server" > /dev/null 2>&1; then
    err "a pAIring server is running without a PID file (started by an older script) - run scripts/stop.sh first"
    exit 1
fi

if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    err "port $PORT is already in use - is another pAIring running?"
    exit 1
fi

if command -v setsid >/dev/null 2>&1; then
    LAUNCHER=(setsid)
else
    LAUNCHER=(nohup)
fi

info "Starting pAIring..."
PAIRING_PID_FILE="$PID_FILE" "${LAUNCHER[@]}" "$APP_DIR/.venv/bin/python" -u "$APP_DIR/run.py" >> "$LOG_FILE" 2>&1 < /dev/null &
SERVER_PID=$!
disown

for _ in $(seq 1 40); do
    [[ -f "$PID_FILE" ]] && break
    # Died during startup (bad config, import error...) - stop waiting.
    kill -0 "$SERVER_PID" 2>/dev/null || break
    sleep 0.25
done

if [[ ! -f "$PID_FILE" ]]; then
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        err "pAIring did not finish starting within 10s (pid $SERVER_PID still running) - check $LOG_FILE"
    else
        err "pAIring exited during startup - last lines of $LOG_FILE:"
        tail -n 5 "$LOG_FILE"
    fi
    exit 1
fi

info "pAIring started (pid $(cat "$PID_FILE")) on port $PORT, logging to $LOG_FILE"
info "Web UI: http://localhost:$PORT"
info "Check status any time with: bash scripts/status.sh"
