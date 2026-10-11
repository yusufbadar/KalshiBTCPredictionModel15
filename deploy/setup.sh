#!/usr/bin/env bash
# Install (or update) the Lock-In bot on an Ubuntu/Debian server.
# Run as root after unzipping the bundle into /opt/lockin:
#     bash /opt/lockin/deploy/setup.sh
# Safe to run again after uploading a newer bundle: settings, keys and trade
# history live in /opt/lockin/storage and /opt/lockin/secrets and are kept.
set -euo pipefail

APP=/opt/lockin
[ "$(id -u)" -eq 0 ] || { echo "Run this as root (sudo bash $0)"; exit 1; }
[ -f "$APP/requirements-server.txt" ] || { echo "Unzip the bundle into $APP first."; exit 1; }

echo "== packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv python3-pip unzip ufw >/dev/null

echo "== service account (no login, owns only its data)"
id lockin >/dev/null 2>&1 || useradd --system --home-dir "$APP" --shell /usr/sbin/nologin lockin
mkdir -p "$APP/storage" "$APP/secrets"

echo "== python environment"
[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install --quiet --upgrade pip
"$APP/.venv/bin/pip" install --quiet -r "$APP/requirements-server.txt"

echo "== permissions (code read-only for the service; data and keys private)"
chown -R root:root "$APP"
chown -R lockin:lockin "$APP/storage" "$APP/secrets"
chmod 700 "$APP/storage" "$APP/secrets"
find "$APP/secrets" -type f -exec chmod 600 {} +

echo "== service"
install -m 644 "$APP/deploy/lockin.service" /etc/systemd/system/lockin.service
systemctl daemon-reload
systemctl enable lockin >/dev/null
systemctl restart lockin

echo "== firewall: SSH only (the dashboard is not exposed; use an SSH tunnel)"
ufw allow OpenSSH >/dev/null
ufw --force enable >/dev/null

sleep 3
systemctl --no-pager --lines=5 status lockin || true
cat <<MSG

Done. The bot's dashboard is running on this server at 127.0.0.1:8000.

1. Set a dashboard password:
     cd $APP && sudo -u lockin .venv/bin/python -m kalshi_bot --set-password
2. On your own computer open a tunnel and leave that window open:
     ssh -N -L 8000:127.0.0.1:8000 root@THIS_SERVER
   then browse to http://localhost:8000
3. Logs:    journalctl -u lockin -f
   Restart: systemctl restart lockin
MSG
