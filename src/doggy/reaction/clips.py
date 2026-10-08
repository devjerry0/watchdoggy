from __future__ import annotations

import logging
import threading
from collections import deque
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.events.store import EventStore
from doggy.reaction.hub import DogCaught

log = logging.getLogger("doggy")


class ClipBuffer:
    """Rolling in-memory ring of ``(mono_ts, jpeg)`` frames.

    Holds only the most recent ``window_seconds`` of frames so a catch can be
    turned into a short clip WITHOUT any continuous SD writes -- nothing touches
    the card until a fire actually asks for a slice.
    """

    def __init__(self, window_seconds: float) -> None:
        self._window = float(window_seconds)
        self._frames: deque[tuple[float, bytes]] = deque()

    def push(self, mono_ts: float, jpeg: bytes) -> None:
        self._frames.append((mono_ts, jpeg))
        # Drop everything older than ``window_seconds`` behind the newest frame.
        cutoff = mono_ts - self._window
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()

    def slice(self, start: float, end: float) -> list[bytes]:
        """JPEGs whose timestamp is in ``[start, end]``, oldest -> newest."""
        return [jpeg for ts, jpeg in self._frames if start <= ts <= end]

    def slice_timed(self, start: float, end: float) -> list[tuple[float, bytes]]:
        """Like ``slice`` but keeps the timestamps (for spaced sampling)."""
        return [(ts, jpeg) for ts, jpeg in self._frames if start <= ts <= end]


def encode_clip(frames: list[bytes], fps: int, out_path: Path) -> Path:
    """Encode JPEG ``frames`` into an animated WebP, returning the path written.

    Animated WebP is the one format every browser plays that this appliance can
    always produce: OpenCV's only universal writer is mp4v (MPEG-4 Part 2),
    which no browser decodes, and shipping a real H.264 encoder would need
    ffmpeg on the device. Any suffix on ``out_path`` is normalized to .webp.
    """
    pil_frames = []
    for jpeg in frames:
        try:
            pil_frames.append(Image.open(BytesIO(jpeg)).convert("RGB"))
        except OSError:
            continue
    if not pil_frames:
        raise ValueError("encode_clip: no decodable frames")

    webp_path = Path(out_path).with_suffix(".webp")
    pil_frames[0].save(
        webp_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
    )
    return webp_path


class ClipService:
    """Per-frame bufferer + pending-clip finalizer + Reaction on catches.

    Owns the rolling in-memory JPEG buffer and the deferred-encode bookkeeping
    that used to live in the pipeline. As a per-frame stage it buffers annotated
    frames; as a hub ``Reaction`` it registers a pending clip on each catch; and
    ``finalize_due`` cuts + encodes clips once their post-roll has elapsed.
    """

    def __init__(self, store: EventStore, event_dir: Path, buffer: ClipBuffer,
                 runtime: RuntimeSettings) -> None:
        self._store = store
        self._event_dir = event_dir
        self._buffer = buffer
        # ``runtime`` supplies the clips-enabled decision at the fire moment,
        # exactly where the pipeline read cfg on the fire frame.
        self._runtime = runtime
        self._pending: list[dict] = []
        self._workers: list[threading.Thread] = []

    def on_frame(self, annotated: np.ndarray, now: float, cfg: TunableSettings) -> None:
        if cfg.clips_enabled:
            # Buffer the ANNOTATED frame so any resulting clip shows the boxes.
            ok, buf = cv2.imencode(".jpg", annotated)
            if ok:
                self._buffer.push(now, buf.tobytes())

    def on_dog_caught(self, event: DogCaught) -> None:
        cfg = self._runtime.get()
        if cfg.clips_enabled:
            # Defer encoding until post-roll has elapsed so the clip captures
            # a few seconds after the catch as well as the pre-roll.
            self._pending.append(
                {"id": event.record.id, "fire_ts": event.mono_ts,
                 "end": event.mono_ts + cfg.clip_postroll_seconds})

    def finalize_due(self, now: float, cfg: TunableSettings) -> None:
        """Hand any pending clip whose post-roll window has elapsed to a
        background encode. Encoding ~70 JPEGs into an animated WebP takes
        5-15s on a Pi 4; done inline it froze the detect loop (FPS read
        0.1) for exactly the seconds after a catch when the dog is still
        there. The frame slice is a private copy, and EventStore.attach_clip
        takes the store lock, so the worker needs no further coordination."""
        if not self._pending:
            return
        still_pending = []
        for p in self._pending:
            if p["end"] > now:
                still_pending.append(p)
                continue
            frames = self._buffer.slice(p["fire_ts"] - cfg.clip_preroll_seconds, p["end"])
            if frames:
                worker = threading.Thread(
                    target=self._encode_and_attach,
                    args=(p["id"], frames, cfg.clip_fps),
                    name=f"clip-{p['id']}", daemon=True,
                )
                worker.start()
                self._workers.append(worker)
        self._pending = still_pending
        self._workers = [w for w in self._workers if w.is_alive()]

    def drain(self, timeout: float | None = None) -> None:
        """Wait for in-flight encodes (shutdown, tests)."""
        for worker in self._workers:
            worker.join(timeout)
        self._workers = [w for w in self._workers if w.is_alive()]

    def _encode_and_attach(self, event_id: str, frames: list[bytes],
                           fps: int) -> None:
        try:
            path = encode_clip(frames, fps, self._event_dir / f"{event_id}.webp")
            self._store.attach_clip(event_id, path.name)
        except Exception:
            log.exception("failed to encode clip for %s", event_id)
