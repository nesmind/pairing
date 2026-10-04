#!/usr/bin/env bash
# Reports whether pAIring is running (PID file run/pairing.pid)
#
# Exit code: 0 running and healthy, 1 stopped, 2 running but not answering.
set -uo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# Colored like uvicorn's own "INFO:" lines (plain text when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

if [ -f "$APP_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$APP_DIR/.env"
    set +a
fi

APP_PORT="${APP_PORT:-8000}"
PID_FILE="${PAIRING_PID_FILE:-$APP_DIR/run/pairing.pid}"

info "--- Python environment ---"
VENV_PYTHON="$APP_DIR/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    info "venv:   $APP_DIR/.venv"
    info "python: $(readlink -f "$VENV_PYTHON") ($("$VENV_PYTHON" --version 2>&1))"
    info "pip:    $("$VENV_PYTHON" -m pip --version 2>&1 | awk '{print $2}')"
else
    err "No .venv found at $APP_DIR/.venv — see README.md's Setup section."
fi

info "--- pAIring (port $APP_PORT) ---"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    PID="$(cat "$PID_FILE")"
elif pgrep -x "pAIring-server" > /dev/null 2>&1; then
    PID="$(pgrep -x "pAIring-server" | head -1)"
    info "no PID file (started by an older script) - using the running process"
else
    err "pAIring: stopped — start with: bash scripts/start.sh"
    exit 1
fi

CODE="$(curl -s -o /dev/null -m 3 -w '%{http_code}' "http://localhost:$APP_PORT/login" 2>/dev/null || true)"
STATUS=0
if [ "$CODE" = "200" ]; then
    info "pAIring: running (pid $PID), healthy on http://localhost:$APP_PORT"
else
    err "pAIring: running (pid $PID), but not answering on http://localhost:$APP_PORT"
    STATUS=2
fi

for i in 1 2 3 4 5 6 7; do
    PORT=$((APP_PORT + i))
    CODE="$(curl -s -o /dev/null -m 2 -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null || true)"
    if [ "$CODE" = "200" ]; then
        info "instance $i (port $PORT): healthy"
    fi
done
exit "$STATUS"
