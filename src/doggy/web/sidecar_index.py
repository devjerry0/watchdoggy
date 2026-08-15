"""In-process index of the dataset's parsed sample_*.json sidecars.

Every dataset/training endpoint needs "all sidecars, parsed" -- and at
thousands of frames, re-reading each file per request took ~10s on the Pi
(SD card + a CPU busy with inference) and the polling training page kept
the server permanently mid-scan. The index re-stats the directory per
request (dentry-cached: milliseconds) and re-parses only files whose
(mtime_ns, size) changed -- normally none. Writers keep writing sidecars
directly; the changed mtime is what invalidates an entry."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_SUFFIX = ".json"
_PREFIX = "sample_"

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

    @property
    def generation(self) -> int:
        return self._generation

    def snapshot(self) -> list[tuple[str, dict]]:
        """(stem, meta) for every parseable sidecar, sorted by stem."""
        with self._lock:
            fresh, changed = self._scan()
            self._cache = fresh
            if changed:
                self._generation += 1
                self._snapshot = [(stem, meta) for stem, (_, meta)
                                  in sorted(fresh.items()) if meta is not None]
            return self._snapshot

    def _scan(self) -> tuple[dict[str, Entry], bool]:
        if not self._dir.is_dir():
            return {}, bool(self._cache)
        fresh: dict[str, Entry] = {}
        changed = False
        with os.scandir(self._dir) as entries:
            for entry in entries:
                name = entry.name
                if not (name.startswith(_PREFIX) and name.endswith(_SUFFIX)):
                    continue
                stem = name[:-len(_SUFFIX)]
                stat = entry.stat()
                key = (stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(stem)
                if cached is not None and cached[0] == key:
                    fresh[stem] = cached
                    continue
                fresh[stem] = (key, _parse(entry.path))
                changed = True
        return fresh, changed or len(fresh) != len(self._cache)

    def sample_bytes(self) -> int:
        """Total size of every sample_* file (frames + sidecars + thumbs are
        excluded: thumbs live in their own subdirectory)."""
        if not self._dir.is_dir():
            return 0
        total = 0
        with os.scandir(self._dir) as entries:
            for entry in entries:
                if entry.name.startswith(_PREFIX) and entry.is_file():
                    total += entry.stat().st_size
        return total
