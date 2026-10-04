#!/usr/bin/env bash
# Stops a pAIring server started via scripts/start.sh: SIGTERM to the PID in run/pairing.pid
# scripts/stop.sh.
#
# Usage: bash scripts/stop.sh
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

# Colored like uvicorn's own "INFO:" lines (plain text when not a terminal).
if [ -t 1 ]; then G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'; else G=""; R=""; N=""; fi
info() { printf '%sINFO:%s %s\n' "$G" "$N" "$*"; }
err() { printf '%sERROR:%s %s\n' "$R" "$N" "$*"; }

PID_FILE="${PAIRING_PID_FILE:-$APP_DIR/run/pairing.pid}"
# The kernel truncates "pAIring-server-N" to this 15-byte name, so one pattern matches every instance.
PROCESS_NAME="pAIring-server-?"

stop_by_name() {
    if ! pgrep -x "$PROCESS_NAME" > /dev/null 2>&1; then
        return 1
    fi
    pkill -x "$PROCESS_NAME" 2>/dev/null || true
    for _ in $(seq 1 20); do
        pgrep -x "$PROCESS_NAME" > /dev/null 2>&1 || return 0
        sleep 0.5
    done
    pkill -9 -x "$PROCESS_NAME" 2>/dev/null || true
    return 0
}

if [[ ! -f "$PID_FILE" ]]; then
    if stop_by_name; then
        info "pAIring stopped (it had no PID file - started by an older script)"
    else
        info "pAIring is not running (no pid file)"
    fi
    exit 0
fi

PID="$(cat "$PID_FILE")"

if ! kill -0 "$PID" 2>/dev/null; then
    info "pAIring is not running (stale pid file, removing it)"
    rm -f "$PID_FILE"
    stop_by_name && info "stopped leftover pAIring instances" || true
    exit 0
fi

kill -TERM "$PID" 2>/dev/null || true

for _ in $(seq 1 30); do
    if ! kill -0 "$PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        stop_by_name && info "stopped leftover pAIring instances" || true
        info "pAIring stopped (pid $PID)"
        exit 0
    fi
    sleep 0.5
done

info "pAIring did not stop gracefully, sending SIGKILL"
kill -KILL "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
stop_by_name && info "stopped leftover pAIring instances" || true
