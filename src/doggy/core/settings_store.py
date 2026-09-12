"""Where the live-tunable settings live: settings.json, next to .env.

.env stays structural (camera, ports, paths, TLS: restart-required, set by
the deploy). Everything the dashboard changes live -- 51 sliders, toggles,
schedules, zone polygons -- was being rewritten INTO .env as
DOGGY_*=... lines (lists as JSON-inside-dotenv), with no record of who
changed what. When the certainty slider sat at 0.95 for a day (Sep 2026)
nothing could say when or from where it moved.

Now: settings.json is the source of truth for tunables, written atomically;
every change appends one line per changed key to settings-changes.jsonl.
Precedence at boot: settings.json > the .env-derived tunables (one-time
migration: the first boot without a settings.json writes one from them) >
pydantic defaults. Stale DOGGY_* tunable keys left behind in .env are
harmless: the JSON wins; rollback to older code still finds them.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from doggy.core.tunables import TunableSettings

log = logging.getLogger("doggy")

SETTINGS_FILE = Path("settings.json")
CHANGELOG_FILE = Path("settings-changes.jsonl")
FORMAT_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_tunables(fallback: TunableSettings,
                  path: Path = SETTINGS_FILE) -> TunableSettings:
    """The live tunables. ``fallback`` is the .env/defaults-derived set
    (Settings.tunable()): used, and persisted as the first settings.json,
    when the file is absent; used without persisting when the file is
    unreadable (the bad file is set aside as .bad-<ts> for inspection)."""
    if not path.is_file():
        log.info("settings: no %s yet; migrating tunables from .env/defaults", path)
        save_tunables(fallback, path=path, changed_by="migration", old=None)
        return fallback
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return TunableSettings(**payload["tunables"])
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as exc:
        aside = path.with_name(f"{path.name}.bad-{int(time.time())}")
        log.error("settings: %s unreadable (%s); using .env/defaults and "
                  "setting the file aside as %s", path, exc, aside.name)
        try:
            os.replace(path, aside)
        except OSError:
            pass
        return fallback


def save_tunables(tunable: TunableSettings, *, changed_by: str,
                  old: TunableSettings | None,
                  path: Path = SETTINGS_FILE,
                  changelog: Path | None = None) -> list[dict]:
    """Atomically persist ``tunable`` and append one changelog line per key
    that differs from ``old`` (all keys when old is None). Returns the
    change records written."""
    changelog = changelog if changelog is not None else path.with_name(CHANGELOG_FILE.name)
    payload = {"version": FORMAT_VERSION, "updated_at": _now_iso(),
               "tunables": tunable.model_dump(mode="json")}
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)  # readers never see a torn file

    new = payload["tunables"]
    before = old.model_dump(mode="json") if old is not None else {}
    changes = [{"ts": payload["updated_at"], "key": key,
                "old": before.get(key), "new": value, "by": changed_by}
               for key, value in new.items()
               if old is None or before.get(key) != value]
    if changes:
        with open(changelog, "a", encoding="utf-8") as fh:
            for record in changes:
                fh.write(json.dumps(record) + "\n")
    return changes


def read_changelog(limit: int = 50,
                   changelog: Path = CHANGELOG_FILE) -> list[dict]:
    """The newest ``limit`` change records, newest first."""
    if not changelog.is_file():
        return []
    lines = changelog.read_text(encoding="utf-8").splitlines()
    out: list[dict] = []
    for line in reversed(lines):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


def read_confidence(path: Path = SETTINGS_FILE) -> float | None:
    """The persisted alarm threshold, for out-of-process readers (the trainer
    daemon's deploy gate judges at the appliance's live threshold). None
    when the file is absent or unreadable; callers keep their fallback."""
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))
                     ["tunables"]["confidence"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
