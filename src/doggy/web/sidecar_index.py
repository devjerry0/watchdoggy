"""In-process index of the dataset's parsed sample_*.json sidecars.

Every dataset/training endpoint needs "all sidecars, parsed" -- and at
thousands of frames, re-reading each file per request took ~10s on the Pi
(SD card + a CPU busy with inference) and the polling training page kept
the server permanently mid-scan.

At ~47k sidecars (~95k directory entries, Sep 2026) even a per-request
re-stat of every entry cost 1-3s of Pi CPU, and with the training page
polling that was a near-continuous core stolen from NCNN inference (~1 FPS
while the page was open). So the index is now event-driven:

- In-process writers (labeling/batch endpoints) call ``upsert(stem)`` after
  editing a sidecar in place: one stat + one parse.
- A directory mtime change (a frame captured, a sample pruned) triggers a
  NAMES-ONLY sync: readdir without stat, then stat + parse only new names
  and drop vanished ones (~50ms at 95k entries).
- A full re-stat of everything runs once per ``_FULL_RESCAN_SECONDS`` (daily)
  or on ``invalidate()``: the safety net for anything written from outside
  this process (a restore, a manual rsync).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_SUFFIX = ".json"
_PREFIX = "sample_"
_FULL_RESCAN_SECONDS = 24 * 3600.0

Entry = tuple[tuple[int, int], "dict | None"]


def _parse(path: str) -> dict | None:
    # A sidecar that fails to parse is skipped, not fatal -- capture may be
    # mid-write; the next mtime bump re-parses it.
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


class SidecarIndex:
    """Thread-safe: web workers call snapshot() concurrently."""

    def __init__(self, dataset_dir: Path | str) -> None:
        self._dir = Path(dataset_dir)
        self._lock = threading.Lock()
        self._cache: dict[str, Entry] = {}
        # Bumps whenever the index content changes. Consumers key derived
        # aggregates (chip counts, byte totals) on this, so a warm request
        # does no per-sidecar Python work at all -- that loop cost,
        # amplified by GIL contention with the inference thread, was what
        # kept the pages slow even after parse caching.
        self._generation = 0
        self._snapshot: list[tuple[str, dict]] = []
        self._sizes: dict[str, int] = {}  # every sample_* file name -> bytes
        self._dir_mtime_ns: int | None = None
        self._full_scanned_at = 0.0
        self._dirty = True

    @property
    def generation(self) -> int:
        return self._generation

    def invalidate(self) -> None:
        """Force a full re-stat on the next snapshot() (ops / restores)."""
        with self._lock:
            self._dirty = True

    def upsert(self, stem: str) -> None:
        """An in-process writer edited (or created) ``stem``'s sidecar: refresh
        that one entry. In-place writes don't move the directory mtime, so
        without this the change would wait for the daily rescan."""
        with self._lock:
            if self._refresh_one(stem):
                self._rebuild_snapshot()

    def snapshot(self) -> list[tuple[str, dict]]:
        """(stem, meta) for every parseable sidecar, sorted by stem."""
        with self._lock:
            now = time.monotonic()
            dir_mtime = self._dir_mtime()
            if self._dirty or now - self._full_scanned_at >= _FULL_RESCAN_SECONDS:
                changed = self._full_scan()
                self._full_scanned_at = now
                self._dirty = False
            elif dir_mtime != self._dir_mtime_ns:
                changed = self._sync_names()
            else:
                changed = False
            # Record the mtime seen BEFORE the work: a file landing mid-way
            # moves it again and forces the next call to sync.
            self._dir_mtime_ns = dir_mtime
            if changed:
                self._rebuild_snapshot()
            return self._snapshot

    def sample_bytes(self) -> int:
        """Total size of every sample_* file (frames + sidecars; thumbs live in
        their own subdirectory), maintained incrementally by the scans."""
        self.snapshot()
        return sum(self._sizes.values())

    # -- internals (caller holds the lock) ----------------------------------

    def _dir_mtime(self) -> int | None:
        try:
            return self._dir.stat().st_mtime_ns
        except OSError:
            return None

    def _rebuild_snapshot(self) -> None:
        self._generation += 1
        self._snapshot = [(stem, meta) for stem, (_, meta)
                          in sorted(self._cache.items()) if meta is not None]

    def _refresh_one(self, stem: str) -> bool:
        """Stat + parse one sidecar (or drop it if gone). True when changed."""
        path = self._dir / f"{stem}{_SUFFIX}"
        try:
            stat = path.stat()
        except OSError:
            gone = stem in self._cache
            self._cache.pop(stem, None)
            self._sizes.pop(path.name, None)
            return gone
        key = (stat.st_mtime_ns, stat.st_size)
        self._sizes[path.name] = stat.st_size
        cached = self._cache.get(stem)
        if cached is not None and cached[0] == key:
            return False
        self._cache[stem] = (key, _parse(str(path)))
        return True

    def _full_scan(self) -> bool:
        """Stat every sample_* entry; re-parse sidecars whose (mtime, size)
        moved. The daily safety net and the startup build."""
        if not self._dir.is_dir():
            changed = bool(self._cache)
            self._cache, self._sizes = {}, {}
            return changed
        fresh: dict[str, Entry] = {}
        sizes: dict[str, int] = {}
        changed = False
        with os.scandir(self._dir) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(_PREFIX) or not entry.is_file():
                    continue
                stat = entry.stat()
                sizes[name] = stat.st_size
                if not name.endswith(_SUFFIX):
                    continue
                stem = name[:-len(_SUFFIX)]
                key = (stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(stem)
                if cached is not None and cached[0] == key:
                    fresh[stem] = cached
                    continue
                fresh[stem] = (key, _parse(entry.path))
                changed = True
        changed = changed or len(fresh) != len(self._cache)
        self._cache, self._sizes = fresh, sizes
        return changed

    def _sync_names(self) -> bool:
        """Directory mtime moved (add/delete): readdir WITHOUT stat, then stat
        + parse only names we have never seen and drop names that vanished.
        In-place edits are the writers' job (upsert)."""
        if not self._dir.is_dir():
            changed = bool(self._cache)
            self._cache, self._sizes = {}, {}
            return changed
        seen: set[str] = set()
        changed = False
        with os.scandir(self._dir) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(_PREFIX):
                    continue
                seen.add(name)
                if name in self._sizes:
                    continue  # known file: no stat
                if not entry.is_file():
                    continue
                stat = entry.stat()
                self._sizes[name] = stat.st_size
                if name.endswith(_SUFFIX):
                    stem = name[:-len(_SUFFIX)]
                    self._cache[stem] = ((stat.st_mtime_ns, stat.st_size),
                                         _parse(entry.path))
                    changed = True
        for name in list(self._sizes):
            if name not in seen:
                del self._sizes[name]
                if name.endswith(_SUFFIX):
                    self._cache.pop(name[:-len(_SUFFIX)], None)
                    changed = True
        return changed
