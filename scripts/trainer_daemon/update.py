"""Self-update from GitHub Releases: the daily check rule and the update
job runner (stage code -> root helper installs + restarts -> health check
-> rollback if the appliance doesn't come back healthy)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import ssl
import tempfile
import time
import urllib.request
from pathlib import Path

from trainer_daemon.env import DOGGY_ROOT, JOBS_DIR, log, settings
from trainer_daemon.github import fetch_source, latest_release

UPDATE_CHECK_INTERVAL = 24 * 3600.0
CHECK_STAMP = JOBS_DIR / "update-check.json"
RELEASE_FILE = DOGGY_ROOT / ".release"
STAGING_CODE = Path.home() / "staging_code"
INSTALL_HELPER = "/usr/local/bin/doggy-install-code"
# What an update replaces. State dirs (dataset, jobs, events, models, sounds,
# soothing, .env) are NEVER part of this list -- see the install helper.
CODE_PATHS = ("src", "scripts", "tests", "pyproject.toml", "uv.lock",
              "README.md", "ARCHITECTURE.md", ".release")
HEALTH_URL = "https://localhost:8443/api/status"
HEALTH_DEADLINE_S = 120.0
HEALTH_POLL_S = 5.0


def installed_release() -> str:
    if not RELEASE_FILE.is_file():
        return "unknown"
    return RELEASE_FILE.read_text().strip() or "unknown"


def _write_check_stamp(latest_tag: str | None) -> None:
    CHECK_STAMP.write_text(json.dumps(
        {"checked_at": time.time(), "latest": latest_tag,
         "installed": installed_release()}))


def _last_checked() -> float:
    try:
        return float(json.loads(CHECK_STAMP.read_text())["checked_at"])
    except (OSError, ValueError, KeyError):
        return 0.0


def update_due(now: float) -> tuple[str, str] | None:
    """Daily: ask GitHub for the latest release; (installed, latest) when an
    update should be queued, None otherwise. Never raises."""
    if not settings().get("auto_update", True):
        return None
    if now - _last_checked() < UPDATE_CHECK_INTERVAL:
        return None
    release = latest_release()
    _write_check_stamp(release["tag"] if release else None)
    if release is None:
        log("update check: GitHub unreachable; will retry tomorrow")
        return None
    current = installed_release()
    if release["tag"] == current:
        return None
    return current, release["tag"]


def _deps_changed(source_root: Path) -> bool:
    ours = DOGGY_ROOT / "uv.lock"
    theirs = source_root / "uv.lock"
    if not ours.is_file() or not theirs.is_file():
        return True  # can't prove deps match -> refuse to self-apply
    return (hashlib.sha256(ours.read_bytes()).digest()
            != hashlib.sha256(theirs.read_bytes()).digest())


def _stage(source_root: Path, tag: str) -> None:
    if STAGING_CODE.exists():
        shutil.rmtree(STAGING_CODE)
    STAGING_CODE.mkdir()
    (source_root / ".release").write_text(tag + "\n")
    for name in CODE_PATHS:
        path = source_root / name
        if not path.exists():
            continue
        if path.is_dir():
            shutil.copytree(path, STAGING_CODE / name)
            continue
        shutil.copyfile(path, STAGING_CODE / name)


def _healthy() -> bool:
    """The detector answered /api/status within the deadline."""
    context = ssl.create_default_context()  # household CA: skip verification
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    deadline = time.time() + HEALTH_DEADLINE_S
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, context=context,
                                        timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(HEALTH_POLL_S)
    return False


def run_update_job(job: dict) -> str:
    release = latest_release()
    _write_check_stamp(release["tag"] if release else None)
    if release is None:
        raise RuntimeError("GitHub unreachable")
    current = installed_release()
    if release["tag"] == current:
        return f"already on {current}"
    with tempfile.TemporaryDirectory() as tmp:
        source = fetch_source(release["tarball_url"], Path(tmp))
        if source is None:
            raise RuntimeError(f"could not fetch/verify {release['tag']}")
        if _deps_changed(source):
            return (f"skipped: {release['tag']} changes dependencies -- "
                    "run scripts/deploy-to-pi.sh from the Mac")
        _stage(source, release["tag"])
    log(f"installing {release['tag']} (was {current})")
    subprocess.run(["sudo", INSTALL_HELPER], check=True)
    if _healthy():
        _write_check_stamp(release["tag"])
        return f"updated {current} -> {release['tag']} (healthy)"
    log(f"{release['tag']} unhealthy after install; rolling back")
    subprocess.run(["sudo", INSTALL_HELPER, "--rollback"], check=True)
    raise RuntimeError(
        f"{release['tag']} unhealthy after install; rolled back to {current}")
