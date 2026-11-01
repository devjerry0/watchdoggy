from __future__ import annotations

import re
import subprocess
from pathlib import Path

from fastapi import APIRouter

# doggy-netwatch (scripts/netwatch.sh, root) locks the Pi's WiFi profile to the
# mesh node it is on. The web app runs as the app user and can't change
# NetworkManager profiles, so the re-pick button only drops this file (relative
# to the app's working directory, like settings.json); netwatch sees it within
# one check, clears the lock, reconnects to the strongest node and locks there.
REPICK_FILE = Path("wifi-repick.request")


def _link() -> dict | None:
    """The current WiFi association, {} when disconnected, or None when `iw`
    is unavailable (e.g. a dev Mac) so the dashboard can hide the control."""
    try:
        done = subprocess.run(["iw", "dev", "wlan0", "link"],
                              capture_output=True, text=True, timeout=3)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    out = done.stdout
    node = re.search(r"^Connected to (\S+)", out, re.M)
    if not node:
        return {}
    ssid = re.search(r"SSID: (.*)", out)
    freq = re.search(r"freq: ([0-9.]+)", out)
    signal = re.search(r"signal: (-?\d+)", out)
    return {
        "node": node.group(1),
        "ssid": ssid.group(1).strip() if ssid else None,
        "freq_mhz": float(freq.group(1)) if freq else None,
        "signal_dbm": int(signal.group(1)) if signal else None,
    }


def build_router() -> APIRouter:
    router = APIRouter()

    @router.get("/api/wifi")
    def get_wifi() -> dict:
        return {"link": _link(), "repick_pending": REPICK_FILE.exists()}

    @router.post("/api/wifi/repick")
    def repick() -> dict:
        REPICK_FILE.touch()
        return {"ok": True}

    return router
