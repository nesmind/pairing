#!/usr/bin/env bash
#
# Restarts the pAIring app — stop.sh then start.sh. Needed after any backend (Python) change, since the
# running server keeps the old code in memory.
#
# Usage: bash scripts/restart.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/stop.sh"
bash "$SCRIPT_DIR/start.sh"
