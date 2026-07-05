#!/usr/bin/env bash
# Deploy doggy to a Raspberry Pi over SSH: rsync the source, install uv + deps,
# download + NCNN-export the model for the ARM CPU, write a Pi .env, and install
# a systemd service so it runs on boot.
#
# Usage:   ./scripts/deploy-to-pi.sh <user@host> [remote_dir]
# Example: ./scripts/deploy-to-pi.sh doggy@doggypi.local
#
# Re-runnable: safe to run repeatedly. It won't clobber an existing .env, and it
# skips the model download/export if already present.
set -euo pipefail

TARGET="${1:?usage: deploy-to-pi.sh <user@host> [remote_dir]}"
REMOTE_DIR="${2:-doggy}"   # relative to the Pi user's home
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Deploying $REPO_DIR  ->  $TARGET:~/$REMOTE_DIR"

# 1. Sync source. Exclude local venv/build junk, heavy artifacts, and the Mac's
#    .env (the Pi gets its own below).
rsync -az --delete \
  --exclude '.venv' --exclude '.git' --exclude '__pycache__' --exclude '.pytest_cache' \
  --exclude 'models' --exclude 'events' --exclude '*.mp4' --exclude '.env' \
  "$REPO_DIR"/ "$TARGET:$REMOTE_DIR"/

# 2. Remote provisioning.
ssh "$TARGET" "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -euo pipefail
cd "$HOME/$REMOTE_DIR"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "==> installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if command -v apt-get >/dev/null 2>&1; then
  echo "==> installing system libs (portaudio for audio, libGL for opencv)"
  sudo apt-get update -qq
  sudo apt-get install -y -qq libportaudio2 libgl1 libglib2.0-0 || true
fi

echo "==> uv sync (CPU-only torch; slowest step on a Pi, be patient)"
uv sync

echo "==> disabling Ultralytics telemetry (appliance runs offline)"
uv run yolo settings sync=False >/dev/null 2>&1 || true

mkdir -p models sounds
if [ ! -d models/yolo26n_ncnn_model ]; then
  # NCNN needs the ncnn runtime + pnnx converter; uv's venv has no pip, so
  # Ultralytics can't auto-install them — install explicitly with uv pip.
  echo "==> installing NCNN toolchain (ncnn runtime + pnnx converter)"
  uv pip install ncnn pnnx || echo "WARN: ncnn/pnnx install failed; will fall back to .pt"
  echo "==> downloading yolo26n + exporting to NCNN (one-time, slow)"
  uv run python - <<'PY'
import pathlib, shutil
from ultralytics import YOLO
YOLO("yolo26n.pt")                                   # download to cwd
pt = pathlib.Path("yolo26n.pt")
if pt.exists():
    shutil.move(str(pt), "models/yolo26n.pt")
try:
    YOLO("models/yolo26n.pt").export(format="ncnn")  # -> models/yolo26n_ncnn_model/
except Exception as e:
    print("NCNN export failed, will fall back to .pt:", e)
PY
fi

# Pick the fastest model artifact that actually exists.
if [ -d models/yolo26n_ncnn_model ]; then
  MODEL="models/yolo26n_ncnn_model"
else
  MODEL="models/yolo26n.pt"
fi
echo "==> using model: $MODEL"

if [ ! -f .env ]; then
  echo "==> writing Pi .env (USB webcam index 0, dashboard on the LAN)"
  cat > .env <<ENV
DOGGY_CAMERA_INDEX=0
DOGGY_MODEL_PATH=$MODEL
DOGGY_CLIPS_DIR=sounds
DOGGY_WEB_HOST=0.0.0.0
DOGGY_WEB_PORT=8000
DOGGY_LOG_LEVEL=INFO
ENV
fi

echo "==> installing systemd service"
sudo tee /etc/systemd/system/doggy.service >/dev/null <<UNIT
[Unit]
Description=Doggy detector
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$HOME/$REMOTE_DIR
EnvironmentFile=$HOME/$REMOTE_DIR/.env
ExecStart=$(command -v uv) run doggy
# always (not on-failure): the app exits 0 when the camera is missing, so it
# must relaunch until the webcam is present / reconnected. Appliance resilience.
Restart=always
RestartSec=15

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable doggy
sudo systemctl restart doggy
echo "==> service started; recent logs:"
sleep 2 || true
sudo journalctl -u doggy -n 15 --no-pager || true
REMOTE

PI_HOST="${TARGET#*@}"
echo
echo "==> Deployed. Dashboard: http://${PI_HOST}:8000"
echo "    Logs:   ssh $TARGET 'journalctl -u doggy -f'"
echo "    Sound:  add clips to ~/$REMOTE_DIR/sounds and set the USB speaker as"
echo "            the default sink (sudo raspi-config -> System -> Audio)."
