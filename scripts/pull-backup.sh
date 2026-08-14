#!/usr/bin/env bash
# Pull an off-device backup of the appliance's irreplaceable state -- the
# labeled training dataset, job history/settings, and deployed models -- to
# ~/watchdoggy-backups/<date>/ on this machine.
#
# The frames must NEVER land inside the repo (it is public); this writes
# only to the home-dir backup folder. Re-runnable: same-day runs update the
# same dated folder. The Pi also keeps its own pre-update state snapshots
# (see doggy-install-code in setup-pi-trainer.sh), and the Modal Volume
# mirrors the dataset per training run -- this is the third, offline copy.
#
# Usage: ./scripts/pull-backup.sh [user@host]
set -euo pipefail

TARGET="${1:-doggy@doggypi.local}"
DEST="$HOME/watchdoggy-backups/$(date +%Y-%m-%d)"
mkdir -p "$DEST"

echo "==> pulling state from $TARGET into $DEST"
rsync -a "$TARGET:doggy/dataset/" "$DEST/dataset/"
rsync -a "$TARGET:doggy/jobs/"    "$DEST/jobs/"
rsync -a "$TARGET:doggy/models/"  "$DEST/models/"
scp -q "$TARGET:doggy/.env" "$DEST/env" 2>/dev/null || true

echo "==> done: $(find "$DEST" -type f | wc -l | tr -d ' ') files, $(du -sh "$DEST" | cut -f1)"
