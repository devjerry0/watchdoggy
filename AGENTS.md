# watchdoggy — agent guide

Raspberry Pi appliance ("Counter Watch") that watches a kitchen counter with a
camera, detects the dog via YOLO/NCNN on-CPU, and plays a deterrent sound
through a Bluetooth/USB speaker. Web dashboard for live view, watch-area,
catch log, and settings. Runs fully offline (nftables egress firewall); only a
sandboxed `trainer` user may reach the internet, for Modal cloud fine-tuning.

This file is the shared brain for all coding agents (Claude Code reads it via
the CLAUDE.md symlink; Codex reads it natively). Capture durable, non-obvious
learnings here (e.g. via /ce-compound). Keep entries terse; cite code paths.

## Ground rules

- **Public repo.** Never commit camera frames, snapshots, or anything from
  `events/`, `dataset-pull/`, or `training-runs/`. Never `git add -A` or
  `git add .` — stage files explicitly by name.
- Python is uv-managed: run things with `uv run doggy`, `uv run pytest tests/`.
  On the Pi, never plain `uv sync` — it removes out-of-band packages and the
  Pi's firewall blocks re-fetching (see docs/ and skills for the deploy flow).
- Package name is `doggy` (src/doggy); entry point `doggy = doggy.main:main`.
- Architecture overview: ARCHITECTURE.md. Systemd units: systemd/.
  Training pipeline: scripts/train_kitchen_model.py.

## Learnings

<!-- Append dated entries below. Terse: what was non-obvious, why, code path. -->
