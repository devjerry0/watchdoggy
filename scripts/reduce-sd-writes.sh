#!/usr/bin/env bash
# Reduce SD-card wear on the doggy Pi (appliance longevity).
#
# The single biggest source of constant writes on a headless Pi is systemd-journald
# persisting logs to /var/log/journal. Switching it to volatile (RAM) storage all
# but eliminates that write traffic — logs still work (journalctl), they just live
# in /run and reset on reboot, which is fine for an appliance.
#
# This is the light-touch durability option. It does NOT freeze the filesystem, so
# BT pairing (/var/lib/bluetooth) and live dashboard knobs (.env) stay persistent —
# unlike a full read-only/overlay root, which would make both volatile.
#
# The real fix for card death is stable power (a proper 5V/3A supply, not a flaky
# charger — brownout power-cycling is what bricks cards). This just extends life.
#
# Usage:   ./scripts/reduce-sd-writes.sh <user@host>
# Example: ./scripts/reduce-sd-writes.sh doggy@doggypi.local
set -euo pipefail

TARGET="${1:?usage: reduce-sd-writes.sh <user@host>}"

echo "==> Reducing SD writes on $TARGET (journald -> RAM)"
ssh "$TARGET" 'bash -s' <<'REMOTE'
set -euo pipefail
sudo sed -i \
  -e 's/^#*\s*Storage=.*/Storage=volatile/' \
  -e 's/^#*\s*RuntimeMaxUse=.*/RuntimeMaxUse=32M/' \
  /etc/systemd/journald.conf
grep -qE '^Storage=volatile'  /etc/systemd/journald.conf || echo 'Storage=volatile'  | sudo tee -a /etc/systemd/journald.conf >/dev/null
grep -qE '^RuntimeMaxUse='    /etc/systemd/journald.conf || echo 'RuntimeMaxUse=32M' | sudo tee -a /etc/systemd/journald.conf >/dev/null
sudo rm -rf /var/log/journal 2>/dev/null || true   # stop persisting to the SD card
sudo systemctl restart systemd-journald
echo "  journald: $(grep -E '^(Storage|RuntimeMaxUse)=' /etc/systemd/journald.conf | tr '\n' ' ')"
REMOTE
echo "==> Done. Logs now live in RAM; SD write traffic drops sharply."
