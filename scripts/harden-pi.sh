#!/usr/bin/env bash
# Harden a deployed doggy Pi into a LAN-only appliance (persists across reboots):
#   - SSH: key-only (disable password auth)
#   - Firewall: nftables egress lockdown — no internet; loopback + LAN + DHCP +
#     mDNS/multicast only. The detector runs fully offline (local camera + model),
#     so it never needs to reach the internet after deploy.
#   - Ultralytics telemetry off.
#
# Run this AFTER deploy-to-pi.sh (so the model/deps are already downloaded).
#
# Usage:   ./scripts/harden-pi.sh <user@host> <lan_cidr>
# Example: ./scripts/harden-pi.sh doggy@doggypi.local 192.168.50.0/24
set -euo pipefail

TARGET="${1:?usage: harden-pi.sh <user@host> <lan_cidr>}"
LAN="${2:?usage: harden-pi.sh <user@host> <lan_cidr>   (e.g. 192.168.50.0/24)}"

echo "==> Hardening $TARGET  (LAN allowed: $LAN)"
ssh "$TARGET" "LAN='$LAN' bash -s" <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

echo "==> Ultralytics telemetry off"
( cd "$HOME/doggy" && uv run yolo settings sync=False >/dev/null 2>&1 ) || true

echo "==> SSH: key-only (00- prefix so it wins over cloud-init's 50- drop-in)"
sudo tee /etc/ssh/sshd_config.d/00-hardening.conf >/dev/null <<EOF
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
EOF
sudo systemctl reload ssh 2>/dev/null || sudo systemctl reload sshd 2>/dev/null || true

echo "==> Firewall: nftables egress lockdown"
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
  }
}
NFT
sudo nft -f /etc/nftables.conf
sudo systemctl enable --now nftables

echo "==> verify"
echo -n "  ssh passwordauth: "; sudo sshd -T | grep '^passwordauthentication'
echo -n "  external egress:  "; curl --max-time 5 -sS https://1.1.1.1 -o /dev/null -w "%{http_code}\n" 2>/dev/null || echo "BLOCKED (good)"
echo    "  nftables on boot: $(systemctl is-enabled nftables)"
REMOTE
echo "==> Done. Pi is LAN-only + key-only SSH; both persist across reboots."
