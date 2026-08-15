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
    # A changed file is re-read...
    _write(tmp_path, "sample_1", {"reasons": ["fire"], "human_label": "dog"})
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
    assert [stem for stem, _ in index.snapshot()] == ["sample_1"]


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
