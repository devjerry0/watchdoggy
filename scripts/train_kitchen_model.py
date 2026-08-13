#!/usr/bin/env python3
"""Entry shim: the training pipeline lives in scripts/kitchen_training/.

Kept so the command everyone knows -- and the future dashboard button --
never changes:

    uv run python scripts/train_kitchen_model.py [--sweep --augment ...]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kitchen_training import main  # noqa: E402

if __name__ == "__main__":
    main()
