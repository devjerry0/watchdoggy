#!/usr/bin/env bash
# Pull the collected training dataset (raw frames + JSON sidecars) from the Pi
# to this machine for labeling and fine-tuning. Re-runnable: rsync only copies
# what's new. The dataset can be a couple of GB, so it is pulled, not served.
#
# Usage:   ./scripts/pull-dataset.sh <user@host> [dest]
# Example: ./scripts/pull-dataset.sh doggy@doggypi.local ./dataset-pull
set -euo pipefail

TARGET="${1:?usage: pull-dataset.sh <user@host> [dest]}"
DEST="${2:-./dataset-pull}"

mkdir -p "$DEST"
rsync -az "$TARGET:doggy/dataset/" "$DEST/"
COUNT=$(find "$DEST" -name 'sample_*.jpg' | wc -l | tr -d ' ')
echo "==> $COUNT frames in $DEST"
echo "    Each sample_*.jpg has a sample_*.json sidecar with what the detector"
echo "    saw (labels, confidences, boxes) and why the frame was kept."
