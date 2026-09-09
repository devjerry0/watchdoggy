"""In-process index of the dataset's parsed sample_*.json sidecars.

Every dataset/training endpoint needs "all sidecars, parsed" -- and at
thousands of frames, re-reading each file per request took ~10s on the Pi
(SD card + a CPU busy with inference) and the polling training page kept
the server permanently mid-scan.

At ~52k sidecars (~105k directory entries, Sep 2026) even a per-request
re-stat of every entry cost 1-3s of Pi CPU, and with the training page
polling that was a near-continuous core stolen from NCNN inference (~1 FPS
while the page was open). So the index is event-driven and keeps bulk work
off request threads:

- In-process writers (labeling/batch endpoints) call ``upsert(stem)`` after
  editing a sidecar in place: one stat + one parse, inserted by bisect.
- A directory mtime change (a frame captured, a sample pruned) triggers a
  NAMES-ONLY sync: readdir without stat, then stat + parse only new names
  and drop vanished ones.
- The full re-stat of everything (startup build, then daily, or after
  ``invalidate()``) runs on a background thread at BACKGROUND_NICE when
  ``start()`` was called; request threads never run it, and while it is
  building they simply see the previous (or empty) snapshot.
"""
from __future__ import annotations

import bisect
import json
import os
import threading
import time
from pathlib import Path
# Bound at import: tests that fake ``threading.Thread`` to run serve()'s
# uvicorn threads inline must not run this daemon's loop on their thread.
from threading import Thread as _Thread

from doggy.core.priority import BACKGROUND_NICE, WEB_WORKER_NICE, lower_thread_priority

_SUFFIX = ".json"
_PREFIX = "sample_"
_FULL_RESCAN_SECONDS = 24 * 3600.0
_REFRESH_POLL_SECONDS = 5.0

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
        # Sorted by stem; kept incrementally (bisect), rebuilt only by a
        # full scan. ``_stems`` mirrors ``_snapshot`` for the bisect keys.
        self._snapshot: list[tuple[str, dict]] = []
        self._stems: list[str] = []
        self._sizes: dict[str, int] = {}  # every sample_* file name -> bytes
        self._dir_mtime_ns: int | None = None
        self._full_scanned_at = 0.0
        self._dirty = True
        self._building = False
        self._refresher: _Thread | None = None
        self._stop = threading.Event()

    @property
    def generation(self) -> int:
        return self._generation

    # -- background refresher ---------------------------------------------

    def start(self) -> None:
        """Warm the index and keep the daily full re-stat off request
        threads: a daemon thread at BACKGROUND_NICE does both."""
        if self._refresher is not None:
            return
        self._refresher = _Thread(target=self._refresh_loop,
                                  name="sidecar-index", daemon=True)
        self._refresher.start()

    def stop(self) -> None:
        self._stop.set()

    def _refresh_loop(self) -> None:
        lower_thread_priority(BACKGROUND_NICE)
        while not self._stop.is_set():
            with self._lock:
                due = (self._dirty or time.monotonic() - self._full_scanned_at
                       >= _FULL_RESCAN_SECONDS)
                if due:
                    self._building = True
            if due:
                try:
                    self._run_full_scan()
                finally:
                    with self._lock:
                        self._building = False
            self._stop.wait(_REFRESH_POLL_SECONDS)

    def _run_full_scan(self) -> None:
        # The scan itself holds the lock (single writer, simple invariants);
        # it is only ever slow on the background thread, where the priority
        # is already lowered.
        with self._lock:
            dir_mtime = self._dir_mtime()
            changed = self._full_scan()
            self._full_scanned_at = time.monotonic()
            self._dirty = False
            self._dir_mtime_ns = dir_mtime
            if changed:
                self._rebuild_all()

    # -- public API ----------------------------------------------------------

    def invalidate(self) -> None:
        """Force a full re-stat (ops / restores). Runs on the refresher when
        started, inline on the next snapshot() otherwise."""
        with self._lock:
            self._dirty = True

    def upsert(self, stem: str) -> None:
        """An in-process writer edited (or created) ``stem``'s sidecar: refresh
        that one entry. In-place writes don't move the directory mtime, so
        without this the change would wait for the daily rescan."""
        with self._lock:
            self._refresh_one(stem)

    def snapshot(self) -> list[tuple[str, dict]]:
        """(stem, meta) for every parseable sidecar, sorted by stem."""
        with self._lock:
            if self._building:
                return self._snapshot  # refresher is mid-build: don't double it
            dir_mtime = self._dir_mtime()
            full_due = (self._dirty or time.monotonic() - self._full_scanned_at
                        >= _FULL_RESCAN_SECONDS)
            if full_due and self._refresher is None:
                lower_thread_priority(WEB_WORKER_NICE)
                changed = self._full_scan()
                self._full_scanned_at = time.monotonic()
                self._dirty = False
                if changed:
                    self._rebuild_all()
            elif not full_due and dir_mtime != self._dir_mtime_ns:
                lower_thread_priority(WEB_WORKER_NICE)
                self._sync_names()
            # Record the mtime seen BEFORE the work: a file landing mid-way
            # moves it again and forces the next call to sync.
            self._dir_mtime_ns = dir_mtime
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

    def _rebuild_all(self) -> None:
        self._generation += 1
        items = sorted(self._cache.items())
        self._stems = [stem for stem, (_, meta) in items if meta is not None]
        self._snapshot = [(stem, meta) for stem, (_, meta) in items
                          if meta is not None]

    def _set_entry(self, stem: str, meta: dict | None) -> None:
        """Insert/replace/remove one snapshot row in sorted position."""
        i = bisect.bisect_left(self._stems, stem)
        present = i < len(self._stems) and self._stems[i] == stem
        if meta is None:
            if present:
                del self._stems[i]
                del self._snapshot[i]
        elif present:
            self._snapshot[i] = (stem, meta)
        else:
            self._stems.insert(i, stem)
            self._snapshot.insert(i, (stem, meta))
        self._generation += 1

    def _refresh_one(self, stem: str) -> bool:
        """Stat + parse one sidecar (or drop it if gone). True when changed."""
        path = self._dir / f"{stem}{_SUFFIX}"
        try:
            stat = path.stat()
        except OSError:
            gone = stem in self._cache
            self._cache.pop(stem, None)
            self._sizes.pop(path.name, None)
            if gone:
                self._set_entry(stem, None)
            return gone
        key = (stat.st_mtime_ns, stat.st_size)
        self._sizes[path.name] = stat.st_size
        cached = self._cache.get(stem)
        if cached is not None and cached[0] == key:
            return False
        meta = _parse(str(path))
        self._cache[stem] = (key, meta)
        self._set_entry(stem, meta)
        return True

    def _full_scan(self) -> bool:
        """Stat every sample_* entry; re-parse sidecars whose (mtime, size)
        moved. The startup build and the daily safety net."""
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
            if changed:
                self._rebuild_all()
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
                    meta = _parse(entry.path)
                    self._cache[stem] = ((stat.st_mtime_ns, stat.st_size), meta)
                    self._set_entry(stem, meta)
                    changed = True
        for name in list(self._sizes):
            if name not in seen:
                del self._sizes[name]
                if name.endswith(_SUFFIX):
                    stem = name[:-len(_SUFFIX)]
                    self._cache.pop(stem, None)
                    self._set_entry(stem, None)
                    changed = True
        return changed
