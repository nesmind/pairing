#!/usr/bin/env bash
#
# Sets up pAIring itself on a fresh Debian (or Ubuntu) server: system
# packages, a Python venv, and a systemd service for the app. Every step
# prints what it's about to do and checks it worked before moving on, so a
# failure points at exactly where things stopped rather than leaving you
# to guess.
#
#     rsync -av --exclude .venv --exclude data --exclude models \
#         /path/to/pAIring/ youruser@newserver:/opt/pairing/
#
# (knowledge/ is included by default — drop the --exclude for it too if
# you'd rather start with an empty knowledge base on the new server.
# data/ is excluded since it's regenerated fresh; models/ is excluded
# since copying potentially 100+GB of weights over the network isn't
# worth it — pull them again on the new server instead.
#
# Then, ON THE NEW SERVER, from inside that copied folder:
#
#     sudo bash scripts/install_on_fresh_server.sh
#
# Safe to re-run: every step below checks whether it's already done
# before doing it again.

set -euo pipefail

# ---------------------------------------------------------------------------
# Small helpers for readable, consistent output — every real step below is
# wrapped in one of these so progress reads as a clear narrated log instead
# of a wall of raw command output.
# ---------------------------------------------------------------------------
step()    { printf '\n\033[1;34m==>\033[0m %s\n' "$1"; }
info()    { printf '    %s\n' "$1"; }
success() { printf '\033[1;32m  ✓\033[0m %s\n' "$1"; }
warn()    { printf '\033[1;33m  ! \033[0m%s\n' "$1"; }
die()     { printf '\033[1;31mError:\033[0m %s\n' "$1" >&2; exit 1; }

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_USER="${SUDO_USER:-$(whoami)}"
APP_PORT="${APP_PORT:-8000}"
SERVICE_NAME="pairing"

printf '\033[1;36m'
cat << 'EOF'
  ┌────────────────────────┐
  │        pAIring          │
  └────────────────────────┘
EOF
printf '\033[0m'
echo "pAIring server setup — running from: $APP_DIR"
echo "This will need sudo for: apt packages, systemd services."

if [ "$(id -u)" -ne 0 ]; then
    die "This script needs root — re-run with: sudo bash scripts/install_on_fresh_server.sh"
fi

# ---------------------------------------------------------------------------
step "1. Checking this is a Debian-family server"
# ---------------------------------------------------------------------------
if [ ! -f /etc/os-release ] || ! grep -qiE 'debian|ubuntu' /etc/os-release; then
    die "This script targets Debian/Ubuntu (checked /etc/os-release). Aborting rather than guessing at package names for something else."
fi
. /etc/os-release
success "Detected: $PRETTY_NAME"

# ---------------------------------------------------------------------------
step "2. Installing system packages (python3, venv)"
# ---------------------------------------------------------------------------
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip rsync
success "System packages installed."

# ---------------------------------------------------------------------------
step "3. Creating the Python virtual environment"
# ---------------------------------------------------------------------------
if [ -x "$APP_DIR/.venv/bin/python" ]; then
    info ".venv already exists — skipping creation."
else
    sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
    success "Created .venv."
fi
info "Installing Python dependencies (this can take a minute)..."
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet --upgrade pip
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install --quiet -r "$APP_DIR/requirements.txt"
success "Python dependencies installed into .venv."

# ---------------------------------------------------------------------------
step "4. Creating the systemd service for pAIring itself"
# ---------------------------------------------------------------------------
cat > "/etc/systemd/system/${SERVICE_NAME}.service" << EOF
[Unit]
Description=pAIring — on-prem multi-user chat UI for local models
After=network.target

[Service]
Type=simple
User=$APP_USER
WorkingDirectory=$APP_DIR
Environment="APP_PORT=$APP_PORT"
ExecStart=$APP_DIR/.venv/bin/python $APP_DIR/run.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
systemctl daemon-reload
systemctl enable "$SERVICE_NAME" >/dev/null
success "Created and enabled ${SERVICE_NAME}.service."

# ---------------------------------------------------------------------------
step "5. .env / SECRET_KEY"
# ---------------------------------------------------------------------------
if [ -f "$APP_DIR/.env" ] && grep -qE '^SECRET_KEY=.+' "$APP_DIR/.env" 2>/dev/null; then
    info "$APP_DIR/.env already has a SECRET_KEY — leaving it as-is."
else
    NEW_SECRET="$(sudo -u "$APP_USER" "$APP_DIR/.venv/bin/python" -c 'import secrets; print(secrets.token_hex(32))')"
    if [ -f "$APP_DIR/.env" ] && grep -qE '^SECRET_KEY=' "$APP_DIR/.env" 2>/dev/null; then
        # An empty "SECRET_KEY=" placeholder line exists (e.g. copied
        # from .env.example) — replace it in place rather than
        # appending a second, conflicting SECRET_KEY line.
        sed -i "s|^SECRET_KEY=.*|SECRET_KEY=$NEW_SECRET|" "$APP_DIR/.env"
    else
        echo "SECRET_KEY=$NEW_SECRET" >> "$APP_DIR/.env"
    fi
    success "Generated a new SECRET_KEY in $APP_DIR/.env"
fi
# .env holds SECRET_KEY and (if configured) the MySQL password — owner-only, applied unconditionally (not just
# on first creation) so a pre-existing .env brought over from another server gets its permissions fixed too.
chown "$APP_USER":"$APP_USER" "$APP_DIR/.env"
chmod 600 "$APP_DIR/.env"

# ---------------------------------------------------------------------------
step "6. Starting pAIring"
# ---------------------------------------------------------------------------
systemctl start "$SERVICE_NAME"
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    success "${SERVICE_NAME}.service is running."
else
    die "${SERVICE_NAME}.service failed to start — check: journalctl -u $SERVICE_NAME -n 50"
fi

# ---------------------------------------------------------------------------
step "Done"
# ---------------------------------------------------------------------------
SERVER_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
cat << EOF

pAIring is running at:  http://${SERVER_IP:-<this-server>}:${APP_PORT}

Log in with the starter admin account:
    admin / admin   (full access, including the Settings > System tab)

  ⚠  CHANGE THIS PASSWORD before letting anyone else reach this server —
     it's seeded on first run for convenience, not meant to stay as-is
     on anything but a private network. Settings > System > Users can
     reset it (or any other account's) once you're logged in.

Useful commands:
    systemctl status ${SERVICE_NAME}          # is the app running?
    journalctl -u ${SERVICE_NAME} -f           # tail the app's logs

Knowledge base: drop .txt/.md/.pdf/.docx/.odt/.rtf/.html/.csv/.json
files into ${APP_DIR}/knowledge/, then click "Rescan" in Settings (or
restart the service) to make them searchable in every chat for all users.
EOF
