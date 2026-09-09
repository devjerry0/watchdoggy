import cv2
import numpy as np
from PIL import Image

from doggy.reaction.clips import ClipBuffer, encode_clip
from doggy.events.store import EventStore


def _img():
    return np.zeros((8, 8, 3), np.uint8)


def _jpeg(color: int, size: tuple[int, int] = (16, 16)) -> bytes:
    img = np.full((size[1], size[0], 3), color, np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_buffer_drops_old_frames():
    b = ClipBuffer(window_seconds=2.0)
    b.push(0.0, b"a")
    b.push(1.0, b"b")
    b.push(3.0, b"c")  # 0.0 is now older than 2s behind the newest -> dropped
    assert b.slice(0.0, 3.0) == [b"b", b"c"]


def test_slice_timed_keeps_timestamps():
    b = ClipBuffer(window_seconds=100.0)
    b.push(1.0, b"a")
    b.push(2.0, b"b")
    assert b.slice_timed(1.0, 2.0) == [(1.0, b"a"), (2.0, b"b")]


def test_slice_returns_inclusive_range_in_order():
    b = ClipBuffer(window_seconds=100.0)
    for i in range(5):
        b.push(float(i), bytes([i]))
    assert b.slice(1.0, 3.0) == [bytes([1]), bytes([2]), bytes([3])]
    assert b.slice(2.5, 10.0) == [bytes([3]), bytes([4])]
    assert b.slice(10.0, 20.0) == []


def test_encode_writes_animated_webp(tmp_path):
    # Animated WebP is the only format every browser can play in the
    # dashboard; OpenCV's mp4v output looks like video but no browser
    # decodes it.
    frames = [_jpeg(c) for c in (10, 100, 200, 50)]
    out = tmp_path / "clip.webp"
    path = encode_clip(frames, fps=6, out_path=out)
    assert path == out
    assert path.exists() and path.stat().st_size > 0
    with Image.open(path) as im:
        assert im.format == "WEBP"
        assert getattr(im, "n_frames", 1) == len(frames)


def test_encode_normalizes_suffix_to_webp(tmp_path):
    # A caller passing a legacy .mp4 path still gets a playable webp,
    # and no mp4 is ever produced.
    frames = [_jpeg(c) for c in (10, 100, 200)]
    path = encode_clip(frames, fps=6, out_path=tmp_path / "clip.mp4")
    assert path.suffix == ".webp"
    assert path.exists() and path.stat().st_size > 0
    assert not (tmp_path / "clip.mp4").exists()


def test_prune_caps_clips_to_retention(tmp_path):
    # keep only the newest 2 clips; older clip files are deleted and the field cleared.
    store = EventStore(tmp_path, max_events=100, max_age_days=0, clip_retention=2)
    recs = []
    for i in range(4):
        r = store.add(_img(), 0.5, None, float(i + 1), float(i + 1))
        clip_name = f"{r.id}.mp4"
        (tmp_path / clip_name).write_bytes(b"video-bytes")
        store.attach_clip(r.id, clip_name)
        recs.append((r.id, clip_name))

    store.prune()

    survivors = {e.id: e for e in store.list()}
    # oldest two clips cleared + files gone
    for eid, clip_name in recs[:2]:
        assert survivors[eid].clip is None
        assert not (tmp_path / clip_name).exists()
    # newest two clips kept
    for eid, clip_name in recs[2:]:
        assert survivors[eid].clip == clip_name
        assert (tmp_path / clip_name).exists()

    # persisted to events.jsonl
    reloaded = EventStore(tmp_path, 100, 0, clip_retention=2)
    rl = {e.id: e for e in reloaded.list()}
    assert rl[recs[0][0]].clip is None
    assert rl[recs[3][0]].clip == recs[3][1]


def test_finalize_encodes_off_the_detect_thread(tmp_path):
    # Encoding ~70 JPEGs to animated WebP takes 5-15s on a Pi; inline it froze
    # the detect loop right after a catch. finalize_due must return without
    # encoding, and the clip attaches once the worker finishes (drain()).
    from doggy.core.config import Settings
    from doggy.core.runtime import RuntimeSettings
    from doggy.reaction.clips import ClipService
    from doggy.reaction.hub import DogCaught

    settings = Settings(clips_enabled=True, clip_window_seconds=10,
                        clip_preroll_seconds=1.0, clip_postroll_seconds=1.0,
                        clip_fps=6)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, max_events=10, max_age_days=0)
    buffer = ClipBuffer(settings.clip_window_seconds)
    for i, color in enumerate((10, 60, 120, 180, 240)):
        buffer.push(float(i), _jpeg(color))
    service = ClipService(store, tmp_path, buffer, runtime)

    record = store.add(_img(), 0.9, None, 2.0, 2.0)
    service.on_dog_caught(DogCaught(record, _img(), 2.0))
    cfg = runtime.get()
    service.finalize_due(2.5, cfg)          # post-roll not elapsed: still pending
    assert service._pending
    service.finalize_due(3.5, cfg)          # elapsed: handed to a worker
    assert not service._pending
    service.drain(timeout=10)
    rec = store.list()[0]
    assert rec.clip == f"{record.id}.webp"
    assert (tmp_path / rec.clip).is_file()
