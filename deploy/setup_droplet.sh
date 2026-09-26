#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 DigitalOcean Droplet (run as root):
#   bash deploy/setup_droplet.sh
# Re-running is safe. Expects the repo cloned at /opt/ctf and keys.cfg filled in.
set -euo pipefail

APP_DIR=/opt/ctf
APP_USER=ctf
DATASET_DIR=/home/$APP_USER/.nyuctf/v20250206

[ "$(id -u)" -eq 0 ] || { echo "Run as root"; exit 1; }
[ -f "$APP_DIR/keys.cfg" ] || { echo "Create $APP_DIR/keys.cfg first (see README)"; exit 1; }

echo "== Packages"
apt-get update -q
apt-get install -y -q git python3-venv python3-pip caddy ufw
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh

echo "== Swap (the agent image build needs memory)"
if ! swapon --show | grep -q /swapfile; then
    fallocate -l 4G /swapfile && chmod 600 /swapfile && mkswap /swapfile && swapon /swapfile
    echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

echo "== App user"
id "$APP_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$APP_USER"
usermod -aG docker "$APP_USER"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"
chmod 600 "$APP_DIR/keys.cfg"

echo "== Python dependencies"
sudo -u "$APP_USER" python3 -m venv "$APP_DIR/.venv"
sudo -u "$APP_USER" "$APP_DIR/.venv/bin/pip" install -q -r "$APP_DIR/requirements.txt"

echo "== NYU CTF dataset (JSON index only; challenge files are fetched per run)"
if [ ! -f "$DATASET_DIR/development_dataset.json" ]; then
    sudo -u "$APP_USER" git clone -q --depth 1 --branch v20250206 --filter=blob:none \
        --no-checkout https://github.com/NYU-LLM-CTF/NYU_CTF_Bench.git "$DATASET_DIR"
    sudo -u "$APP_USER" git -C "$DATASET_DIR" sparse-checkout set --no-cone '*.json'
    sudo -u "$APP_USER" git -C "$DATASET_DIR" checkout -q v20250206 || true
fi

echo "== Agent Docker image (10-20 min the first time)"
docker network inspect ctfnet >/dev/null 2>&1 || docker network create ctfnet
docker build -t ctfenv:multiagent "$APP_DIR/docker/multiagent"

echo "== Web server + service"
cp "$APP_DIR/deploy/Caddyfile" /etc/caddy/Caddyfile
systemctl reload-or-restart caddy
cp "$APP_DIR/deploy/ctf-webapp.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now ctf-webapp
systemctl restart ctf-webapp

echo "== Firewall"
ufw allow OpenSSH >/dev/null
ufw allow 80,443/tcp >/dev/null
ufw --force enable >/dev/null

echo "Done. Open http://$(curl -s https://checkip.amazonaws.com)/"
