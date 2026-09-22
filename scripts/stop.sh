#!/usr/bin/env bash
#
# Stops the pAIring app cleanly. Companion to scripts/start.sh.
#
# Usage: bash scripts/stop.sh

set -uo pipefail

PROCESS_NAME="pAIring-server-?"
GRACE_SECONDS=10

if ! pgrep -x "$PROCESS_NAME" > /dev/null 2>&1; then
    echo "pAIring wasn't running."
    exit 0
fi

pkill -x "$PROCESS_NAME" 2>/dev/null

deadline=$(($(date +%s) + GRACE_SECONDS))
while pgrep -x "$PROCESS_NAME" > /dev/null 2>&1; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
        echo "pAIring didn't stop gracefully within ${GRACE_SECONDS}s — forcing it."
        pkill -9 -x "$PROCESS_NAME" 2>/dev/null
        sleep 0.5
        break
    fi
    sleep 0.3
done

if pgrep -x "$PROCESS_NAME" > /dev/null 2>&1; then
    echo "✗ Could not stop pAIring — a process is still running. Check for a stuck process by hand."
    exit 1
fi
echo "✓ Stopped pAIring."
