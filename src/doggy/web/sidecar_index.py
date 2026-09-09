"""In-process index of the dataset's parsed sample_*.json sidecars.

Every dataset/training endpoint needs "all sidecars, parsed" -- and at
thousands of frames, re-reading each file per request took ~10s on the Pi
(SD card + a CPU busy with inference) and the polling training page kept
the server permanently mid-scan. The index re-parses only files whose
(mtime_ns, size) changed -- normally none.

At ~47k sidecars (~95k directory entries, Sep 2026) even the re-stat pass
per request cost 1-3s of CPU on the Pi, and with the training page polling
that was a near-continuous core stolen from NCNN inference (~1 FPS while
the page was open). So a scan now runs only when the directory's own mtime
moved (a sidecar was added or deleted), when an in-process writer called
``invalidate()`` after editing a sidecar in place (all writers are in this
process: the labeling/batch endpoints and the capture stage), or as a
safety net every ``_MAX_STALE_SECONDS``."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_SUFFIX = ".json"
_PREFIX = "sample_"
_MAX_STALE_SECONDS = 300.0

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
        # Bumps whenever a scan sees any add/change/delete. Consumers key
        # derived aggregates (chip counts, byte totals) on this, so a warm
        # request does no per-sidecar Python work at all -- that loop cost,
        # amplified by GIL contention with the inference thread, was what
        # kept the pages slow even after parse caching.
        self._generation = 0
        self._snapshot: list[tuple[str, dict]] = []
        self._dir_mtime_ns: int | None = None
        self._scanned_at = 0.0
        self._dirty = True
        self._bytes = 0

    @property
    def generation(self) -> int:
        return self._generation

    def invalidate(self) -> None:
        """An in-process writer edited a sidecar in place (which does not move
        the directory mtime): the next snapshot() must rescan."""
        with self._lock:
            self._dirty = True

    def snapshot(self) -> list[tuple[str, dict]]:
        """(stem, meta) for every parseable sidecar, sorted by stem."""
        with self._lock:
            if self._scan_due():
                fresh, changed = self._scan()
                self._cache = fresh
                if changed:
                    self._generation += 1
                    self._snapshot = [(stem, meta) for stem, (_, meta)
                                      in sorted(fresh.items()) if meta is not None]
            return self._snapshot

    def _scan_due(self) -> bool:
        try:
            dir_mtime = self._dir.stat().st_mtime_ns
        except OSError:
            dir_mtime = None
        now = time.monotonic()
        if (self._dirty or dir_mtime != self._dir_mtime_ns
                or now - self._scanned_at >= _MAX_STALE_SECONDS):
            # Record the mtime seen BEFORE scanning: a file landing mid-scan
            # moves it again and forces the next call to rescan.
            self._dir_mtime_ns = dir_mtime
            self._scanned_at = now
            self._dirty = False
            return True
        return False

    def _scan(self) -> tuple[dict[str, Entry], bool]:
        if not self._dir.is_dir():
            self._bytes = 0
            return {}, bool(self._cache)
        fresh: dict[str, Entry] = {}
        changed = False
        total = 0
        with os.scandir(self._dir) as entries:
            for entry in entries:
                name = entry.name
                if not name.startswith(_PREFIX) or not entry.is_file():
                    continue
                stat = entry.stat()
                total += stat.st_size
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
        self._bytes = total
        return fresh, changed or len(fresh) != len(self._cache)

    def sample_bytes(self) -> int:
        """Total size of every sample_* file (frames + sidecars; thumbs live in
        their own subdirectory). Accumulated by the last scan, so it costs no
        second directory pass."""
        self.snapshot()
        return self._bytes
