#!/usr/bin/env python3
"""Entry shim: the trainer daemon lives in scripts/trainer_daemon/.

Kept so the path the Pi's systemd unit invokes (see setup-pi-trainer.sh)
never changes:

    /home/trainer/modal-env/bin/python /home/doggy/doggy/scripts/pi_trainer.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trainer_daemon.daemon import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
