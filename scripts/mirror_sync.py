"""Client-side mirror sync: delta uploads plus the light index.

Standalone on purpose: this runs in the trainer venv, which has modal and
nothing else -- importing it must never drag in the ML stack (the
kitchen_training package __init__ eagerly imports cv2 and friends).

Runs on the machine that owns the frames (the Pi's trainer, or a Mac).
Two jobs: ship only new/changed sample_* files to the shared Volume mirror
(short retried chunks, manifest checkpoint after every chunk so any attempt
resumes exactly where the last stopped), and ship ONE small index of what
every sidecar says -- on the Volume's network filesystem, rediscovering
"which frames are labeled" by reading tens of thousands of sidecars took
the better part of an hour per run. The uploader already knows; it says so
in a file."""
from __future__ import annotations

import json
import os
import posixpath
import time
from pathlib import Path

MIRROR = "pipeline/mirror"
# Requests are sized by bytes: ~80 jpgs or up to 2,000 small sidecars.
# A flat 100 files per request made a 39k-sidecar resync (after a big
# labeling pass) take ~20 min of per-request overhead and manifest rewrites.
UPLOAD_CHUNK_BYTES = 8 << 20
UPLOAD_CHUNK_MAX_FILES = 2000
PROGRESS_EVERY_S = 10
CHUNK_RETRIES = 5
CHUNK_RETRY_WAIT_S = 20


def shard_of(name: str) -> str:
    """The day folder a mirror file lives in on the cloud volume: the first
    5 digits of its sample_ millisecond timestamp (~1.16 days per folder, a
    few thousand files each). One flat folder of ~192k files made the v1
    volume ~100x slower, and v2 caps a folder at 262,144 files. Non-sample
    files (the index) stay at the mirror root ("")."""
    if not name.startswith("sample_"):
        return ""
    digits = name[len("sample_"):][:5]
    return digits if len(digits) == 5 and digits.isdigit() else ""


def mirror_key(name: str) -> str:
    """Volume path of a mirror file."""
    return posixpath.join(MIRROR, shard_of(name), name)


def upload_chunks(names: list[str], sizes: dict[str, int]):
    """Consecutive runs of names under UPLOAD_CHUNK_BYTES (a single bigger
    file goes alone) and UPLOAD_CHUNK_MAX_FILES."""
    chunk, total = [], 0
    for name in names:
        if chunk and (total + sizes[name] > UPLOAD_CHUNK_BYTES
                      or len(chunk) >= UPLOAD_CHUNK_MAX_FILES):
            yield chunk
            chunk, total = [], 0
        chunk.append(name)
        total += sizes[name]
    if chunk:
        yield chunk


def _put_chunk(volume, dataset_dir: Path, chunk: list[str]) -> None:
    for attempt in range(CHUNK_RETRIES):
        try:
            with volume.batch_upload(force=True) as up:
                for name in chunk:
                    up.put_file(str(dataset_dir / name), mirror_key(name))
            return
        except Exception as exc:
            if attempt == CHUNK_RETRIES - 1:
                raise
            print(f"[pipeline]   chunk failed ({type(exc).__name__}); "
                  f"retry {attempt + 2}/{CHUNK_RETRIES} in "
                  f"{CHUNK_RETRY_WAIT_S}s", flush=True)
            time.sleep(CHUNK_RETRY_WAIT_S)


def _write_index(volume, dataset_dir: Path) -> None:
    """{stem: [human_label, auto_verdict, settled, disputed, has_boxes]}
    for every frame -- everything the cloud's build, jury prefilters, and
    box re-send need without opening a single sidecar over the network
    filesystem."""
    index: dict = {}
    for side in dataset_dir.glob("sample_*.json"):
        try:
            meta = json.loads(side.read_text())
        except (OSError, ValueError):
            continue
        index[side.stem] = [
            meta.get("human_label"),
            (meta.get("auto_label") or {}).get("verdict"),
            1 if meta.get("dispute_settled_at") else 0,
            1 if meta.get("disputed") else 0,
            1 if "prelabels" in meta else 0,
        ]
    local = _state_dir(dataset_dir) / ".labeled-index.json"
    local.write_text(json.dumps(index))
    with volume.batch_upload(force=True) as up:
        up.put_file(str(local), mirror_key(".labeled-index.json"))
    print(f"[pipeline] index shipped: {len(index)} frames", flush=True)


def _state_dir(dataset_dir: Path) -> Path:
    """Where the manifest and index live. On the appliance the trainer
    cannot create files in doggy's dataset dir (755 doggy:doggy); the
    shared jobs dir (2775 doggy:trainer) is the two-user meeting point by
    construction. A workstation owns its dataset copy and keeps state
    beside it."""
    if os.access(dataset_dir, os.W_OK):
        return dataset_dir
    return dataset_dir.parent / "jobs"


def sync_mirror(volume, dataset_dir: Path) -> None:
    started = time.monotonic()
    state = _state_dir(dataset_dir)
    manifest_file = state / ".upload-manifest.json"
    # Legacy location fallback: earlier versions kept state beside the
    # frames; honor an existing manifest there so migration re-uploads
    # nothing.
    source = manifest_file if manifest_file.is_file() \
        else dataset_dir / ".upload-manifest.json"
    try:
        old = json.loads(source.read_text())
    except (OSError, ValueError):
        old = {}
    current = {}
    for p in sorted(dataset_dir.iterdir()):
        if not p.name.startswith("sample_") or not p.is_file():
            continue
        st = p.stat()
        current[p.name] = [st.st_mtime_ns, st.st_size]
    changed = [n for n, sig in current.items() if old.get(n) != sig]
    deleted = [n for n in old if n not in current]
    print(f"[pipeline] mirror sync: {len(changed)} to upload, "
          f"{len(deleted)} to delete, {len(current) - len(changed)} unchanged "
          f"(scanned in {time.monotonic() - started:.0f}s)", flush=True)
    done, last_print, upload_started = 0, time.monotonic(), time.monotonic()
    total_bytes = sum(current[n][1] for n in changed)
    for chunk in upload_chunks(changed, {n: current[n][1] for n in changed}):
        _put_chunk(volume, dataset_dir, chunk)
        for name in chunk:
            old[name] = current[name]
        manifest_file.write_text(json.dumps(old))  # resume point
        done += len(chunk)
        if done == len(changed) or time.monotonic() - last_print >= PROGRESS_EVERY_S:
            print(f"[pipeline]   uploaded {done}/{len(changed)}", flush=True)
            last_print = time.monotonic()
    if changed:
        took = max(time.monotonic() - upload_started, 1e-6)
        print(f"[pipeline]   upload took {took:.0f}s: "
              f"{len(changed) / took:.0f} files/s, "
              f"{total_bytes / took / 1e6:.1f} MB/s", flush=True)
    for name in deleted:
        try:
            volume.remove_file(mirror_key(name))
        except Exception:
            pass  # already gone is fine; the mirror only needs to converge
    manifest_file.write_text(json.dumps(current))
    index_started = time.monotonic()
    _write_index(volume, dataset_dir)
    print(f"[pipeline] sync took {time.monotonic() - started:.0f}s "
          f"(index {time.monotonic() - index_started:.0f}s)", flush=True)
