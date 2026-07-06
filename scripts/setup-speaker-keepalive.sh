#!/usr/bin/env bash
# Keep a Bluetooth speaker from idle-auto-powering-off.
#
# Cheap BT speakers (e.g. JBL Go 5) don't just sleep when idle — they POWER OFF
# after ~10-15 min of no audio. A powered-off speaker can't receive an alert AND
# can't be reconnected by the host (it's simply gone until turned on by hand),
# which silently breaks the whole deterrent. Fix: play ~1s of silence on a timer,
# often enough to reset the speaker's auto-off countdown.
#
# Installs a user systemd timer (needs the app user's PipeWire session, i.e. linger
# enabled — see setup-bt-speaker.sh). Keep the speaker on WALL POWER, since it now
# never sleeps.
#
# Usage:   ./scripts/setup-speaker-keepalive.sh <user@host> [interval_seconds]
# Example: ./scripts/setup-speaker-keepalive.sh doggy@doggypi.local 240
set -euo pipefail

TARGET="${1:?usage: setup-speaker-keepalive.sh <user@host> [interval_seconds]}"
INTERVAL="${2:-240}"   # default 4 min; lower it if your speaker powers off sooner

echo "==> Installing BT speaker keep-alive on $TARGET (every ${INTERVAL}s)"
ssh "$TARGET" "INTERVAL='$INTERVAL' bash -s" <<'REMOTE'
set -euo pipefail
export XDG_RUNTIME_DIR="/run/user/$(id -u)"

# 1s of silence (16-bit mono 44.1k). pw-play needs a real WAV; streaming it keeps
# the A2DP transport active for ~1s, which resets the speaker's idle timer.
python3 -c "import wave; w=wave.open('$HOME/silence.wav','w'); w.setnchannels(1); w.setsampwidth(2); w.setframerate(44100); w.writeframes(b'\x00\x00'*44100); w.close()"

mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/doggy-keepalive.service" <<UNIT
[Unit]
Description=Keep BT speaker awake (prevent idle auto power-off)
[Service]
Type=oneshot
ExecStart=/usr/bin/pw-play $HOME/silence.wav
UNIT
cat > "$HOME/.config/systemd/user/doggy-keepalive.timer" <<UNIT
[Unit]
Description=Play silence periodically so the BT speaker does not auto power-off
[Timer]
OnBootSec=60
OnUnitActiveSec=${INTERVAL}
[Install]
WantedBy=timers.target
UNIT

systemctl --user daemon-reload
systemctl --user enable --now doggy-keepalive.timer
echo "  keep-alive timer: $(systemctl --user is-active doggy-keepalive.timer)"
REMOTE
echo "==> Done. Keep the speaker on wall power (it no longer sleeps). If it still"
echo "    powers off, re-run with a smaller interval, e.g. 180 or 120."
