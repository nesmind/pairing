#!/usr/bin/env bash
#
# Installs pAIring itself on Linux or macOS: a Python venv, this app's own
# dependencies, a real SECRET_KEY in .env, and a migrated + seeded database
#
# Interactive when run from a real terminal (asks about dev dependencies,
# the port to listen on, and whether to start pAIring once done) — falls
# back to sane defaults with no prompts when stdin isn't a tty (piped in,
# or run from CI), so it's still safe to script.
#
# Safe to re-run: every step below skips work that's already done, and
# never overwrites an .env you've already configured.
#
# Usage:
#   bash scripts/install.sh          # runtime dependencies only
#   bash scripts/install.sh --dev    # plus pytest/ruff, for working on the project itself

set -euo pipefail

step()    { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
info()    { printf '    %s\n' "$1"; }
success() { printf '\033[1;32m  ✓\033[0m %s\n' "$1"; }
warn()    { printf '\033[1;33m  ! \033[0m%s\n' "$1"; }
die()     { printf '\033[1;31mError:\033[0m %s\n' "$1" >&2; exit 1; }

# Whether to prompt at all — piping this script in (`curl ... | bash`) or
# running it from CI leaves stdin non-interactive, and a `read` there would
# either hang or silently read garbage from whatever was piped in.
INTERACTIVE=false
if [ -t 0 ]; then
    INTERACTIVE=true
fi

INSTALL_DEV=false
DEV_FLAG_GIVEN=false
for arg in "$@"; do
    case "$arg" in
        --dev) INSTALL_DEV=true; DEV_FLAG_GIVEN=true ;;
        *) die "Unknown option: $arg (only --dev is supported)" ;;
    esac
done

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"

OS_NAME="$(uname -s)"
case "$OS_NAME" in
    Linux|Darwin) : ;;
    *) die "This installer targets Linux and macOS (uname reported '$OS_NAME'). On Windows, see README.md." ;;
esac

printf '\033[1;36m'
cat << 'EOF'
  ┌────────────────────────┐
  │        pAIring          │
  └────────────────────────┘
EOF
printf '\033[0m'
echo "pAIring installer — $OS_NAME, running from: $APP_DIR"

# ---------------------------------------------------------------------------
step "1. Checking for Python 3.11+"
# ---------------------------------------------------------------------------
PYTHON_BIN=""
for candidate in python3.13 python3.12 python3.11 python3; do
    if command -v "$candidate" > /dev/null 2>&1; then
        PYTHON_BIN="$candidate"
        break
    fi
done
if [ -z "$PYTHON_BIN" ]; then
    if [ "$OS_NAME" = "Darwin" ]; then
        die "python3 not found. Install it with 'brew install python@3.12' (https://brew.sh), or from https://python.org."
    else
        die "python3 not found. Install it with your distro's package manager, e.g. 'sudo apt install python3 python3-venv python3-pip' on Debian/Ubuntu."
    fi
fi
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_OK="$("$PYTHON_BIN" -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')"
if [ "$PY_OK" != "1" ]; then
    die "Found Python $PY_VERSION via '$PYTHON_BIN', but pAIring needs 3.11+. Install a newer Python and re-run."
fi
success "Using $PYTHON_BIN (Python $PY_VERSION)"

# ---------------------------------------------------------------------------
step "2. Creating the virtual environment"
# ---------------------------------------------------------------------------
if [ -x "$APP_DIR/.venv/bin/python" ]; then
    info ".venv already exists — skipping creation."
else
    if ! "$PYTHON_BIN" -m venv "$APP_DIR/.venv" 2>/tmp/pairing-venv-error; then
        cat /tmp/pairing-venv-error >&2
        if [ "$OS_NAME" = "Linux" ]; then
            die "Could not create .venv — Debian/Ubuntu often needs the venv module installed separately: sudo apt install python3-venv"
        else
            die "Could not create .venv (see error above)."
        fi
    fi
    rm -f /tmp/pairing-venv-error
    success "Created .venv"
fi

# ---------------------------------------------------------------------------
step "3. Installing Python dependencies"
# ---------------------------------------------------------------------------
# Only asks if `--dev` wasn't already given on the command line — an
# explicit flag always wins over the interactive prompt.
if [ "$DEV_FLAG_GIVEN" = false ] && [ "$INTERACTIVE" = true ]; then
    read -r -p "Install dev dependencies too (pytest, ruff — only needed if you're working on pAIring itself)? [y/N] " DEV_ANSWER
    case "$DEV_ANSWER" in
        [Yy]*) INSTALL_DEV=true ;;
    esac
fi

"$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
success "Runtime dependencies installed."
if [ "$INSTALL_DEV" = true ]; then
    "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements-dev.txt"
    success "Dev dependencies installed too (pytest, ruff)."
fi

# ---------------------------------------------------------------------------
step "4. Setting up .env"
# ---------------------------------------------------------------------------
ENV_IS_NEW=false
if [ ! -f "$APP_DIR/.env" ]; then
    cp "$APP_DIR/.env.example" "$APP_DIR/.env"
    ENV_IS_NEW=true
    info "Created .env from .env.example."
else
    info ".env already exists — leaving its settings as configured."
fi
# .env holds SECRET_KEY and (if configured) the MySQL password — owner-only, regardless of whatever the
# filesystem's default umask would otherwise have left it at. Applied every run, not just on first creation, so
# a pre-existing .env with looser permissions (e.g. from before this line existed) gets fixed too.
chmod 600 "$APP_DIR/.env"

# Only offered on a brand-new .env — re-running this script must never
# silently change a port someone already deployed with.
APP_PORT_CHOSEN=8000
if [ "$ENV_IS_NEW" = true ] && [ "$INTERACTIVE" = true ]; then
    read -r -p "Which port should pAIring listen on? [8000] " PORT_ANSWER
    if [ -n "$PORT_ANSWER" ]; then
        if ! [[ "$PORT_ANSWER" =~ ^[0-9]+$ ]] || [ "$PORT_ANSWER" -lt 1 ] || [ "$PORT_ANSWER" -gt 65535 ]; then
            warn "\"$PORT_ANSWER\" isn't a valid port — keeping the default (8000)."
        else
            APP_PORT_CHOSEN="$PORT_ANSWER"
        fi
    fi
fi
if [ "$APP_PORT_CHOSEN" != "8000" ]; then
    sed "s|^APP_PORT=.*|APP_PORT=$APP_PORT_CHOSEN|" "$APP_DIR/.env" > "$APP_DIR/.env.tmp"
    mv "$APP_DIR/.env.tmp" "$APP_DIR/.env"
    success "pAIring will listen on port $APP_PORT_CHOSEN."
fi

if grep -qE '^SECRET_KEY=.+' "$APP_DIR/.env" 2>/dev/null; then
    info ".env already has a SECRET_KEY — leaving it as-is."
else
    NEW_SECRET="$("$APP_DIR/.venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))')"
    if grep -qE '^SECRET_KEY=' "$APP_DIR/.env" 2>/dev/null; then
        # Portable in-place sed: BSD sed (macOS) requires an explicit (even
        # if empty) backup-suffix argument after -i; GNU sed (Linux) treats
        # that same bare argument as the pattern instead and breaks. Writing
        # to a temp file and moving it over works identically on both.
        sed "s|^SECRET_KEY=.*|SECRET_KEY=$NEW_SECRET|" "$APP_DIR/.env" > "$APP_DIR/.env.tmp"
        mv "$APP_DIR/.env.tmp" "$APP_DIR/.env"
    else
        echo "SECRET_KEY=$NEW_SECRET" >> "$APP_DIR/.env"
    fi
    success "Generated a new SECRET_KEY in .env"
fi

# ---------------------------------------------------------------------------
step "5. Setting up the database"
# ---------------------------------------------------------------------------
# Runs the exact same schema-migration + starter-admin-seeding logic the
# app itself runs on every startup (see app/services/startup_service.py) —
# just directly, once, right now, rather than waiting for the first
# request. That's what makes "open the browser and log in" work
# immediately after this script finishes, instead of only after the app's
# own first (silent, in-the-background) startup pass has had time to run.
if "$APP_DIR/.venv/bin/python" -c "
import asyncio
from app.database import init_db, AsyncSessionLocal
from app.services.auth_service import seed_default_users

async def main():
    await init_db()
    async with AsyncSessionLocal() as db:
        await seed_default_users(db)

asyncio.run(main())
"; then
    success "Database ready — starter admin account exists (admin / admin)."
else
    die "Database setup failed (see error above). Fix the issue and re-run this script — it's safe to re-run."
fi

# ---------------------------------------------------------------------------
step "6. Checking for Ollama (not installed by this script)"
# ---------------------------------------------------------------------------
OLLAMA_FOUND=false
if command -v ollama > /dev/null 2>&1; then
    success "Found: $(ollama --version 2>&1 | head -1)"
    OLLAMA_FOUND=true
fi

# ---------------------------------------------------------------------------
step "Done"
# ---------------------------------------------------------------------------
echo
echo "pAIring is installed at: $APP_DIR"
echo "Log in with the starter admin account: admin / admin — change that"
echo "password before this server is reachable by anyone else (Settings >"
echo "System > Users)."

START_NOW=false
if [ "$INTERACTIVE" = true ]; then
    read -r -p $'\nStart pAIring now? [Y/n] ' START_ANSWER
    case "${START_ANSWER:-Y}" in
        [Nn]*) START_NOW=false ;;
        *) START_NOW=true ;;
    esac
fi

if [ "$START_NOW" = true ]; then
    bash "$APP_DIR/scripts/start.sh"
else
    cat << EOF

Next steps:
    bash scripts/start.sh    # starts pAIring
    bash scripts/status.sh   # check what's running
    bash scripts/stop.sh     # stop everything

Then open http://localhost:$APP_PORT_CHOSEN
EOF
fi

echo
echo "Want pAIring to survive a reboot without running start.sh by hand?"
if [ "$OS_NAME" = "Linux" ]; then
    cat << EOF
See scripts/pairing.service — a systemd unit template you can copy to
/etc/systemd/system/ and enable, if you'd rather run it that way.
EOF
else
    cat << EOF
On macOS, wrap the same "bash scripts/start.sh" command in a launchd
agent (systemd's closest macOS equivalent) — scripts/pairing.service is
a Linux-only systemd template, not applicable here.
EOF
fi
if [ "$OLLAMA_FOUND" = false ]; then
    echo
    warn "Remember: Ollama still isn't installed — pAIring will start and you then can download and install it from pAIring UI, on macOS you will have to install it yourself later on"
fi
