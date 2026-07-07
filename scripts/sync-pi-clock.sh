#!/usr/bin/env bash
# Set the Pi's clock and timezone from THIS machine, fully offline (LAN/SSH only),
# and install a tiny "clock-keeper" so time survives reboots and power cuts.
#
# Why: the appliance is egress-firewalled (no public NTP) and the router may not
# serve NTP either. This machine's clock is the time source instead.
#
#   - sets the Pi's timezone to this machine's zone (override with 2nd arg)
#   - steps the Pi's clock to this machine's current time
#   - installs clock-keeper: saves the epoch hourly + at shutdown, restores it
#     at boot if the system clock is behind (a no-internet fake-hwclock)
#
# Usage:   ./scripts/sync-pi-clock.sh <user@host> [timezone]
# Example: ./scripts/sync-pi-clock.sh doggy@doggypi.local
# Re-runnable: run it whenever you want to correct drift (SSH + sudo on the Pi).
set -euo pipefail

TARGET="${1:?usage: sync-pi-clock.sh <user@host> [timezone]}"
TZ_NAME="${2:-$(readlink /etc/localtime | sed 's|.*/zoneinfo/||')}"
EPOCH="$(date +%s)"

echo "==> Syncing $TARGET to $TZ_NAME @ $EPOCH ($(date))"
ssh "$TARGET" "TZ_NAME='$TZ_NAME' EPOCH='$EPOCH' bash -s" <<'REMOTE'
set -euo pipefail

echo "==> timezone -> $TZ_NAME"
sudo timedatectl set-timezone "$TZ_NAME"

echo "==> stepping clock to the laptop's time"
# date -s works even while systemd-timesyncd is active (set-time would refuse).
sudo date -s "@$EPOCH" >/dev/null
echo "    now: $(date)"

echo "==> installing clock-keeper (offline fake-hwclock)"
sudo mkdir -p /var/lib/clock-keeper
sudo tee /usr/local/sbin/clock-keeper >/dev/null <<'SCRIPT'
#!/bin/sh
# save: persist the current epoch. restore: if the system clock is behind the
# saved epoch (cold boot with no RTC), step forward to it. Never steps back.
FILE=/var/lib/clock-keeper/epoch
case "$1" in
  save)
    date +%s > "$FILE.tmp" && mv "$FILE.tmp" "$FILE" ;;
  restore)
    [ -f "$FILE" ] || exit 0
    saved=$(cat "$FILE")
    now=$(date +%s)
    [ "$now" -lt "$saved" ] && date -s "@$saved" >/dev/null
    exit 0 ;;
esac
SCRIPT
sudo chmod +x /usr/local/sbin/clock-keeper

sudo tee /etc/systemd/system/clock-keeper.service >/dev/null <<'UNIT'
[Unit]
Description=Restore saved clock at boot, save at shutdown (no-internet fake-hwclock)
DefaultDependencies=no
After=local-fs.target
Before=sysinit.target shutdown.target
Conflicts=shutdown.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/clock-keeper restore
ExecStop=/usr/local/sbin/clock-keeper save

[Install]
WantedBy=sysinit.target
UNIT

sudo tee /etc/systemd/system/clock-keeper-save.service >/dev/null <<'UNIT'
[Unit]
Description=Save current clock for clock-keeper

[Service]
Type=oneshot
ExecStart=/usr/local/sbin/clock-keeper save
UNIT

sudo tee /etc/systemd/system/clock-keeper-save.timer >/dev/null <<'UNIT'
[Unit]
Description=Save the clock hourly (tiny write; journald is already volatile)

[Timer]
OnBootSec=15min
OnUnitActiveSec=1h

[Install]
WantedBy=timers.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now clock-keeper.service clock-keeper-save.timer >/dev/null 2>&1
sudo /usr/local/sbin/clock-keeper save
echo "    keeper enabled; saved epoch: $(cat /var/lib/clock-keeper/epoch)"
REMOTE

echo "==> Done. The Pi's time now matches this machine."
echo "    Drift correction: re-run this script occasionally (the Pi drifts a few"
echo "    seconds/day without a time source). Or enable your router's NTP server"
echo "    (ASUS: Administration -> System -> Enable NTP server) and the Pi's"
echo "    existing timesyncd config will keep it synced continuously, still offline."
