#!/usr/bin/env bash
# Set up the Pi's autonomous trainer: a dedicated `trainer` user that is the
# ONLY thing on the appliance allowed to reach the internet (DNS + TCP 443,
# enforced per-UID by nftables), used to send training jobs to Modal and
# apply the gated results. The doggy detector service stays provably offline.
#
# Idempotent. Run AFTER harden-pi.sh and a normal code deploy (the Pi needs
# scripts/modal_pipeline.py, scripts/kitchen_training/, scripts/pi_trainer.py).
#
# Usage: ./scripts/setup-pi-trainer.sh <user@host>   (copies your ~/.modal.toml)
set -euo pipefail

TARGET="${1:?usage: setup-pi-trainer.sh <user@host>}"

echo "==> Copying Modal token"
scp -q "$HOME/.modal.toml" "$TARGET:/tmp/modal.toml"

ssh "$TARGET" "bash -s" <<'REMOTE'
set -euo pipefail

echo "==> trainer user"
id trainer >/dev/null 2>&1 || sudo useradd -m -s /bin/bash trainer
# Group membership + group-traverse on doggy's home: the trainer reads the
# dataset and scripts but never gains write on the detector's files.
sudo usermod -aG doggy trainer
sudo chmod 750 /home/doggy
sudo install -o trainer -g trainer -m 600 /tmp/modal.toml /home/trainer/.modal.toml
rm -f /tmp/modal.toml

echo "==> firewall: per-UID egress exception (DNS + 443) for trainer only"
LAN=$(grep -oE 'ip daddr [0-9.]+/[0-9]+ accept' /etc/nftables.conf | head -1 | awk '{print $3}')
sudo tee /etc/nftables.conf >/dev/null <<NFT
#!/usr/sbin/nft -f
flush ruleset
table inet fw {
  chain input { type filter hook input priority 0; policy accept; }
  chain forward { type filter hook forward priority 0; policy drop; }
  chain output {
    type filter hook output priority 0; policy drop;
    ct state established,related accept
    oifname "lo" accept
    ip daddr $LAN accept
    ip daddr { 255.255.255.255, 224.0.0.0/4 } accept
    ip6 daddr { fe80::/10, fc00::/7, ff00::/8 } accept
    udp dport { 67, 68 } accept
    # The trainer user -- and ONLY the trainer user -- may reach the
    # internet, so it can send training jobs to Modal. DNS + HTTPS.
    meta skuid trainer udp dport 53 accept
    meta skuid trainer tcp dport 53 accept
    meta skuid trainer tcp dport 443 accept
  }
}
NFT
sudo nft -f /etc/nftables.conf

echo "==> modal client venv for trainer"
sudo -u trainer bash -c '
  set -euo pipefail
  [ -x "$HOME/modal-env/bin/modal" ] && exit 0
  python3 -m venv "$HOME/modal-env"
  "$HOME/modal-env/bin/pip" install -q --upgrade pip
  "$HOME/modal-env/bin/pip" install -q modal
'

echo "==> jobs dir writable by both the web (doggy) and the trainer"
sudo mkdir -p /home/doggy/doggy/jobs
sudo chown doggy:trainer /home/doggy/doggy/jobs
sudo chmod 2775 /home/doggy/doggy/jobs

echo "==> root helper: install a staged model bundle (fixed paths, no args)"
sudo tee /usr/local/bin/doggy-install-model >/dev/null <<'HELPER'
#!/usr/bin/env bash
# Installs /home/trainer/staging_ncnn_model as the live detector model.
# Fixed paths on purpose: this is the only file operation the trainer user
# can perform as root.
set -euo pipefail
STAGING=/home/trainer/staging_ncnn_model
LIVE=/home/doggy/doggy/models/kitchen_ncnn_model
[ -d "$STAGING" ] || { echo "no staged bundle" >&2; exit 1; }
rm -rf "$LIVE.prev"
[ -d "$LIVE" ] && mv "$LIVE" "$LIVE.prev"
cp -r "$STAGING" "$LIVE"
chown -R doggy:doggy "$LIVE"
HELPER
sudo chmod 755 /usr/local/bin/doggy-install-model

echo "==> sudoers: trainer may install the staged model + restart the service"
sudo tee /etc/sudoers.d/doggy-trainer >/dev/null <<'SUDO'
trainer ALL=(root) NOPASSWD: /usr/local/bin/doggy-install-model
trainer ALL=(root) NOPASSWD: /usr/bin/systemctl restart doggy
SUDO
sudo chmod 440 /etc/sudoers.d/doggy-trainer

echo "==> systemd: trainer pass every 30 minutes"
sudo tee /etc/systemd/system/doggy-trainer.service >/dev/null <<'UNIT'
[Unit]
Description=watchdoggy trainer pass (cloud training jobs via Modal)
After=network-online.target

[Service]
Type=oneshot
User=trainer
WorkingDirectory=/home/doggy/doggy
ExecStart=/home/trainer/modal-env/bin/python /home/doggy/doggy/scripts/pi_trainer.py
TimeoutStartSec=14400
UNIT
sudo tee /etc/systemd/system/doggy-trainer.timer >/dev/null <<'UNIT'
[Unit]
Description=watchdoggy trainer schedule

[Timer]
OnBootSec=5min
OnUnitActiveSec=30min

[Install]
WantedBy=timers.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable --now doggy-trainer.timer

echo "==> verify"
echo -n "  trainer egress:  "
sudo -u trainer curl --max-time 8 -sS -o /dev/null -w "%{http_code}\n" https://api.modal.com || echo "FAILED"
echo -n "  doggy egress:    "
sudo -u doggy curl --max-time 5 -sS -o /dev/null -w "%{http_code}\n" https://1.1.1.1 2>/dev/null || echo "BLOCKED (good)"
echo    "  timer:           $(systemctl is-enabled doggy-trainer.timer)"
REMOTE
echo "==> Done. The Pi now trains itself; watch: journalctl -u doggy-trainer -f"
