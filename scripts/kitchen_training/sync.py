"""Client-side mirror sync: delta uploads plus the light index.

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
import time
from pathlib import Path

MIRROR = "pipeline/mirror"
UPLOAD_CHUNK_FILES = 100
CHUNK_RETRIES = 5
CHUNK_RETRY_WAIT_S = 20


def _put_chunk(volume, dataset_dir: Path, chunk: list[str]) -> None:
    for attempt in range(CHUNK_RETRIES):
        try:
            with volume.batch_upload(force=True) as up:
                for name in chunk:
                    up.put_file(str(dataset_dir / name), f"{MIRROR}/{name}")
            return
        except Exception as exc:
            if attempt == CHUNK_RETRIES - 1:
                raise
            print(f"[pipeline]   chunk failed ({type(exc).__name__}); "
                  f"retry {attempt + 2}/{CHUNK_RETRIES} in "
                  f"{CHUNK_RETRY_WAIT_S}s", flush=True)
            time.sleep(CHUNK_RETRY_WAIT_S)


def _write_index(volume, dataset_dir: Path) -> None:
    """{stem: [human_label, auto_verdict, settled, disputed]} for every
    frame -- everything the cloud's build and jury prefilters need without
    opening a single sidecar over the network filesystem."""
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
        ]
    local = dataset_dir / ".labeled-index.json"
    local.write_text(json.dumps(index))
    with volume.batch_upload(force=True) as up:
        up.put_file(str(local), f"{MIRROR}/.labeled-index.json")
    print(f"[pipeline] index shipped: {len(index)} frames", flush=True)


def sync_mirror(volume, dataset_dir: Path) -> None:
    manifest_file = dataset_dir / ".upload-manifest.json"
    try:
        old = json.loads(manifest_file.read_text())
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
          f"{len(deleted)} to delete, {len(current) - len(changed)} unchanged",
          flush=True)
    for start in range(0, len(changed), UPLOAD_CHUNK_FILES):
        chunk = changed[start:start + UPLOAD_CHUNK_FILES]
        _put_chunk(volume, dataset_dir, chunk)
        for name in chunk:
            old[name] = current[name]
        manifest_file.write_text(json.dumps(old))  # resume point
        print(f"[pipeline]   uploaded "
              f"{min(start + UPLOAD_CHUNK_FILES, len(changed))}/{len(changed)}",
              flush=True)
    for name in deleted:
        try:
            volume.remove_file(f"{MIRROR}/{name}")
        except Exception:
            pass  # already gone is fine; the mirror only needs to converge
    manifest_file.write_text(json.dumps(current))
    _write_index(volume, dataset_dir)
