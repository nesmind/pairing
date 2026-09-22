#!/usr/bin/env bash
#
# Stops the pAIring app cleanly. Companion to scripts/start.sh.
#
# Usage: bash scripts/stop.sh

set -uo pipefail

if pkill -x "pAIring-server-?" 2>/dev/null; then
    echo "✓ Stopped pAIring."
else
    echo "pAIring wasn't running."
fi
