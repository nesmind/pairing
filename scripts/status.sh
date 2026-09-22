#!/usr/bin/env bash

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -f "$APP_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$APP_DIR/.env"
    set +a
fi

APP_PORT="${APP_PORT:-8000}"

echo "--- Python environment ---"
VENV_PYTHON="$APP_DIR/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
    echo "venv:   $APP_DIR/.venv"
    echo "python: $(readlink -f "$VENV_PYTHON") ($("$VENV_PYTHON" --version 2>&1))"
    echo "pip:    $("$VENV_PYTHON" -m pip --version 2>&1 | awk '{print $2}')"
else
    echo "✗ No .venv found at $APP_DIR/.venv — see README.md's Setup section."
fi

echo "--- pAIring (port $APP_PORT) ---"
if pgrep -x "pAIring-server" > /dev/null 2>&1; then
    echo "✓ running (pid $(pgrep -x "pAIring-server" | tr '\n' ' '))"
    curl -s -o /dev/null -w "  Web UI reachable: HTTP %{http_code}\n" "http://localhost:$APP_PORT/login" 2>/dev/null || echo "  Web UI not reachable yet"
else
    echo "✗ not running — start with: bash scripts/start.sh"
fi

for i in 1 2 3 4 5 6 7; do
    PORT=$((APP_PORT + i))
    CODE="$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:$PORT/health" 2>/dev/null)"
    if [ "$CODE" = "200" ]; then
        echo "  instance $i (port $PORT): ✓ healthy"
    fi
done
