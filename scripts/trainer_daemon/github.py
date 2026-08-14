"""Talking to GitHub Releases (over the trainer's 443 exception): what the
latest release is, and safely fetching + unpacking its source tarball."""
from __future__ import annotations

import json
import tarfile
import urllib.request
from pathlib import Path

REPO = "devjerry0/watchdoggy"
API_TIMEOUT_S = 30
TARBALL_TIMEOUT_S = 300
# GitHub rejects requests without a User-Agent.
_HEADERS = {"User-Agent": "watchdoggy-updater",
            "Accept": "application/vnd.github+json"}


def latest_release() -> dict | None:
    """{"tag": ..., "tarball_url": ...} for the newest release, or None."""
    request = urllib.request.Request(
        f"https://api.github.com/repos/{REPO}/releases/latest",
        headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_S) as resp:
            release = json.load(resp)
    except Exception:
        return None
    tag = release.get("tag_name")
    tarball = release.get("tarball_url")
    if not tag or not tarball:
        return None
    return {"tag": tag, "tarball_url": tarball}


def _safe_members(tar: tarfile.TarFile) -> list[tarfile.TarInfo]:
    """Only plain files/dirs with names that stay inside the extract root."""
    members = []
    for member in tar.getmembers():
        name = member.name
        if name.startswith("/") or ".." in Path(name).parts:
            continue
        if not (member.isfile() or member.isdir()):
            continue  # no symlinks/devices from a network tarball
        members.append(member)
    return members


def fetch_source(tarball_url: str, workdir: Path) -> Path | None:
    """Download + unpack the release tarball; returns the source root (the
    tarball's single top-level directory) or None on any failure."""
    tar_path = workdir / "release.tar.gz"
    request = urllib.request.Request(tarball_url, headers=_HEADERS)
    try:
        with urllib.request.urlopen(request, timeout=TARBALL_TIMEOUT_S) as resp:
            tar_path.write_bytes(resp.read())
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(workdir, members=_safe_members(tar))
    except Exception:
        return None
    roots = [p for p in workdir.iterdir() if p.is_dir()]
    if len(roots) != 1:
        return None
    # Sanity: this really is the appliance's source tree.
    pyproject = roots[0] / "pyproject.toml"
    if not pyproject.is_file() or 'name = "doggy"' not in pyproject.read_text():
        return None
    return roots[0]
