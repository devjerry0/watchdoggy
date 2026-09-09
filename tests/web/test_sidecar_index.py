import json
import os

from doggy.web.sidecar_index import SidecarIndex


def _write(d, stem, meta):
    (d / f"{stem}.json").write_text(json.dumps(meta))


def test_snapshot_tracks_new_changed_and_deleted(tmp_path):
    index = SidecarIndex(tmp_path)
    assert index.snapshot() == []
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    _write(tmp_path, "sample_2", {"reasons": ["periodic"]})
    assert [stem for stem, _ in index.snapshot()] == ["sample_1", "sample_2"]
    # A changed file is re-read once the (in-process) writer invalidates:
    # an in-place edit does not move the directory mtime the scan keys on...
    _write(tmp_path, "sample_1", {"reasons": ["fire"], "human_label": "dog"})
    index.upsert("sample_1")
    assert index.snapshot()[0][1]["human_label"] == "dog"
    # ...and a deleted one drops out.
    (tmp_path / "sample_2.json").unlink()
    assert [stem for stem, _ in index.snapshot()] == ["sample_1"]


def test_unchanged_files_are_not_reparsed(tmp_path):
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    first = index.snapshot()[0][1]
    # Same parsed object comes back while (mtime, size) is unchanged: the
    # cache hit is what makes thousands of sidecars per request affordable.
    assert index.snapshot()[0][1] is first


def test_torn_sidecar_is_skipped_then_recovers(tmp_path):
    (tmp_path / "sample_1.json").write_text('{"reasons": ["fi')  # mid-write
    index = SidecarIndex(tmp_path)
    assert index.snapshot() == []
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    # Force a different mtime even on coarse-timestamp filesystems.
    os.utime(tmp_path / "sample_1.json", ns=(1, 1))
    # The completed rewrite is in place (no dir mtime move): it is picked up
    # on the next invalidate() or dir change (the next capture), or the
    # staleness net -- never silently lost.
    index.invalidate()
    assert [stem for stem, _ in index.snapshot()] == ["sample_1"]


def test_generation_moves_only_on_change(tmp_path):
    index = SidecarIndex(tmp_path)
    index.snapshot()
    start = index.generation
    # Quiet scans don't move the generation -- derived-aggregate memos
    # (chip counts, dataset tallies) rely on this to skip recomputing.
    index.snapshot()
    assert index.generation == start
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index.snapshot()
    bumped = index.generation
    assert bumped > start
    index.snapshot()
    assert index.generation == bumped
    (tmp_path / "sample_1.json").unlink()
    index.snapshot()
    assert index.generation > bumped


def test_non_sidecar_files_ignored_and_bytes_counted(tmp_path):
    _write(tmp_path, "sample_1", {"reasons": []})
    (tmp_path / "sample_1.jpg").write_bytes(b"x" * 100)
    (tmp_path / "notes.txt").write_text("not a sample")
    (tmp_path / "thumbs").mkdir()
    (tmp_path / "thumbs" / "sample_1.jpg").write_bytes(b"y" * 50)
    index = SidecarIndex(tmp_path)
    assert len(index.snapshot()) == 1
    # bytes: the frame + its sidecar, but not thumbs or foreign files.
    sidecar_size = (tmp_path / "sample_1.json").stat().st_size
    assert index.sample_bytes() == 100 + sidecar_size


def test_quiet_directory_is_not_rescanned(tmp_path, monkeypatch):
    # At ~95k entries the per-request stat pass cost 1-3s of Pi CPU and,
    # with the training page polling, ~1 FPS. No change -> no scan.
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    index.snapshot()
    scans = []
    monkeypatch.setattr(index, "_full_scan", lambda: scans.append("full") or False)
    monkeypatch.setattr(index, "_sync_names", lambda: scans.append("names") or False)
    for _ in range(5):
        index.snapshot()
    assert scans == []


def test_added_file_rescans_without_invalidate(tmp_path):
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    assert len(index.snapshot()) == 1
    _write(tmp_path, "sample_2", {"reasons": ["periodic"]})  # dir mtime moves
    assert len(index.snapshot()) == 2


def test_in_place_edit_needs_invalidate_or_staleness(tmp_path, monkeypatch):
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    index.snapshot()
    _write(tmp_path, "sample_1", {"reasons": ["fire"], "human_label": "dog"})
    assert "human_label" not in index.snapshot()[0][1]   # not yet: dir mtime same
    index.upsert("sample_1")                               # what the writer does
    assert index.snapshot()[0][1]["human_label"] == "dog"
    # ...and the safety net: a stale index re-stats everything on its own.
    _write(tmp_path, "sample_1", {"reasons": ["fire"], "human_label": "empty"})
    index._full_scanned_at -= 100_000
    assert index.snapshot()[0][1]["human_label"] == "empty"


def test_sample_bytes_comes_from_the_scan(tmp_path):
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    (tmp_path / "sample_1.jpg").write_bytes(b"x" * 100)
    (tmp_path / "other.txt").write_bytes(b"y" * 50)
    index = SidecarIndex(tmp_path)
    expected = (tmp_path / "sample_1.json").stat().st_size + 100
    assert index.sample_bytes() == expected


def test_added_file_uses_names_sync_not_full_restat(tmp_path, monkeypatch):
    # A captured frame moves the dir mtime: only the NEW name gets a stat +
    # parse; the other 95k entries are not re-stat'ed.
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    index.snapshot()
    full = []
    monkeypatch.setattr(index, "_full_scan", lambda: full.append(1) or False)
    _write(tmp_path, "sample_2", {"reasons": ["periodic"]})
    assert [s for s, _ in index.snapshot()] == ["sample_1", "sample_2"]
    (tmp_path / "sample_1.json").unlink()
    assert [s for s, _ in index.snapshot()] == ["sample_2"]
    assert full == []


def test_upsert_of_a_deleted_sidecar_drops_it(tmp_path):
    _write(tmp_path, "sample_1", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    assert len(index.snapshot()) == 1
    (tmp_path / "sample_1.json").unlink()
    index.upsert("sample_1")
    assert index.snapshot() == []


def test_background_refresher_builds_the_index_off_the_request_path(tmp_path):
    import time
    for i in range(3):
        _write(tmp_path, f"sample_{i}", {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    index.start()
    deadline = time.monotonic() + 10
    while index.generation == 0 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert index.generation > 0
    assert [s for s, _ in index.snapshot()] == ["sample_0", "sample_1", "sample_2"]
    # A later capture is still picked up by the request-path names sync.
    _write(tmp_path, "sample_3", {"reasons": ["periodic"]})
    assert [s for s, _ in index.snapshot()][-1] == "sample_3"


def test_incremental_snapshot_keeps_sorted_order(tmp_path):
    for stem in ("sample_5", "sample_1", "sample_9"):
        _write(tmp_path, stem, {"reasons": ["fire"]})
    index = SidecarIndex(tmp_path)
    assert [s for s, _ in index.snapshot()] == ["sample_1", "sample_5", "sample_9"]
    _write(tmp_path, "sample_3", {"reasons": ["fire"]})
    index.upsert("sample_3")                            # bisect insert, no re-sort
    _write(tmp_path, "sample_5", {"reasons": ["fire"], "human_label": "dog"})
    index.upsert("sample_5")                            # replace in place
    (tmp_path / "sample_9.json").unlink()
    index.upsert("sample_9")                            # drop
    snap = index.snapshot()
    assert [s for s, _ in snap] == ["sample_1", "sample_3", "sample_5"]
    assert dict(snap)["sample_5"]["human_label"] == "dog"


def test_lower_thread_priority_is_a_safe_noop_off_linux(monkeypatch):
    from doggy.core import priority
    monkeypatch.setattr(priority.sys, "platform", "darwin")
    priority.lower_thread_priority(15)                  # must not raise
