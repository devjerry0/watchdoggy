"""Entry-point shim: the installed console script resolves doggy.main:main.

The Pi service starts with `uv run --no-sync` (no re-install on deploy), so this
module name must never move again -- the real wiring lives in doggy.app.
"""
from doggy.app import main

if __name__ == "__main__":
    main()
