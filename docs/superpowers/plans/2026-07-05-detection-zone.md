# Detection Zone + Detection Interval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Only dogs inside a user-drawn polygon zone trigger the alert, and cap inference rate for thermal relief.

**Architecture:** New `ZoneFilter` rasterizes a normalized polygon to a cached frame-sized mask and keeps detections whose box overlaps it; new `Pacer` throttles the detect loop. `Pipeline.run_once` filters detections through the zone before the trigger; `annotate` draws the polygon + red (in-zone) / grey (ignored) boxes. The frontend gets a canvas overlay to click-place polygon vertices. All new config lives in `TunableSettings` (env + live + `.env`).

**Tech Stack:** Python 3.11, pydantic / pydantic-settings, OpenCV (`cv2`), NumPy, FastAPI, vanilla-JS static frontend, pytest.

## Global Constraints

- All params env-configurable via `TunableSettings` (single source of truth), prefix `DOGGY_`.
- Polygon vertices are **normalized [0,1]** floats; a zone needs **≥3 points** (fewer = disabled = alert anywhere).
- `zone_enabled=False` default → **no behavior change** for existing deployments.
- Keep `imgsz=640` (do NOT reduce model resolution); thermal relief comes from `detect_interval_seconds` only.
- `run_once` must call `self.clock()` exactly once (existing tests supply exact-length clock iterators).
- Tests live flat in `tests/`; run with `uv run pytest`.

---

### Task 1: Config fields + `.env` JSON round-trip

**Files:**
- Modify: `src/doggy/config.py` (add fields to `TunableSettings`)
- Modify: `src/doggy/web.py:29-49` (`_write_env` JSON-encode non-scalars)
- Test: `tests/test_config.py`, `tests/test_web.py`

**Interfaces:**
- Produces: `TunableSettings.zone_enabled: bool`, `TunableSettings.zone_points: list[tuple[float,float]]`, `TunableSettings.detect_interval_seconds: float`.

- [ ] **Step 1: Write failing tests**

In `tests/test_config.py`:
```python
def test_zone_defaults_disabled():
    from doggy.config import Settings
    s = Settings()
    assert s.zone_enabled is False
    assert s.zone_points == []
    assert s.detect_interval_seconds == 0.7

def test_zone_points_parsed_from_env(monkeypatch):
    from doggy.config import Settings
    monkeypatch.setenv("DOGGY_ZONE_ENABLED", "true")
    monkeypatch.setenv("DOGGY_ZONE_POINTS", "[[0.1,0.2],[0.3,0.4],[0.5,0.1]]")
    s = Settings()
    assert s.zone_enabled is True
    assert s.zone_points == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]
```

In `tests/test_web.py` (add near the other `_write_env` tests):
```python
def test_write_env_roundtrips_zone_points(tmp_path, monkeypatch):
    from doggy.web import _write_env
    from doggy.config import Settings, TunableSettings
    env = tmp_path / ".env"
    env.write_text("DOGGY_CAMERA_INDEX=0\n")
    _write_env(TunableSettings(zone_enabled=True,
                               zone_points=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]), env)
    text = env.read_text()
    assert "DOGGY_ZONE_POINTS=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.1]]" in text
    assert "DOGGY_CAMERA_INDEX=0" in text            # structural key preserved
    # and it re-parses:
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=str(env))
    assert s.zone_points == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_config.py::test_zone_defaults_disabled tests/test_web.py::test_write_env_roundtrips_zone_points -q`
Expected: FAIL (fields/format missing).

- [ ] **Step 3: Add config fields**

In `src/doggy/config.py`, inside `TunableSettings` (after `log_level`), add:
```python
    zone_enabled: bool = False
    zone_points: list[tuple[float, float]] = Field(default_factory=list)
    detect_interval_seconds: float = Field(0.7, ge=0.0)
```
(`Field` is already imported.)

- [ ] **Step 4: Fix `_write_env` to JSON-encode non-scalars**

In `src/doggy/web.py`, add `import json` at the top, and replace the `updates = {...}` line in `_write_env` (line ~33) with:
```python
    def _fmt(v: object) -> str:
        if isinstance(v, (str, int, float, bool)) or isinstance(v, Path):
            return str(v)
        return json.dumps(v)   # lists/tuples -> JSON so pydantic-settings can re-parse
    updates = {f"DOGGY_{k.upper()}": _fmt(v) for k, v in tunable.model_dump().items()}
```
(`Path` is already imported.) Note: `json.dumps` of a list-of-tuples emits a JSON array-of-arrays.

- [ ] **Step 5: Run tests to verify pass**

Run: `uv run pytest tests/test_config.py tests/test_web.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/doggy/config.py src/doggy/web.py tests/test_config.py tests/test_web.py
git commit -m "feat: zone + detect-interval config fields (.env JSON round-trip)"
```

---

### Task 2: `ZoneFilter` (polygon mask + box overlap)

**Files:**
- Create: `src/doggy/zone.py`
- Test: `tests/test_zone.py`

**Interfaces:**
- Consumes: `doggy.detection.Detection` (`.box` = `(x1,y1,x2,y2)`).
- Produces: `ZoneFilter().filter(detections: list[Detection], points: list[tuple[float,float]], shape: tuple) -> list[Detection]` and `.in_zone(box, points, shape) -> bool`. `points` are normalized [0,1]; `<3` points → pass-through (returns all). `shape` is `frame.shape` (`(h, w, ...)`).

- [ ] **Step 1: Write failing tests** — `tests/test_zone.py`:
```python
import numpy as np
from doggy.detection import Detection
from doggy.zone import ZoneFilter

# A triangle covering the top-left area of a 100x100 frame.
TRI = [(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)]
SHAPE = (100, 100, 3)

def test_box_inside_zone_is_kept():
    z = ZoneFilter()
    assert z.in_zone((5, 5, 15, 15), TRI, SHAPE) is True

def test_box_outside_zone_is_dropped():
    z = ZoneFilter()
    assert z.in_zone((80, 80, 95, 95), TRI, SHAPE) is False

def test_box_straddling_boundary_overlaps():
    z = ZoneFilter()
    # box spans the diagonal edge -> partial overlap -> True
    assert z.in_zone((25, 25, 45, 45), TRI, SHAPE) is True

def test_filter_keeps_only_in_zone():
    z = ZoneFilter()
    inside = Detection("dog", 0.9, (5, 5, 15, 15))
    outside = Detection("dog", 0.9, (80, 80, 95, 95))
    assert z.filter([inside, outside], TRI, SHAPE) == [inside]

def test_fewer_than_three_points_passes_through():
    z = ZoneFilter()
    d = Detection("dog", 0.9, (80, 80, 95, 95))
    assert z.filter([d], [(0.1, 0.1)], SHAPE) == [d]

def test_mask_rebuilds_on_shape_change():
    z = ZoneFilter()
    assert z.in_zone((5, 5, 15, 15), TRI, (100, 100, 3)) is True
    # different shape must not reuse the old mask (would index out of range)
    assert z.in_zone((5, 5, 15, 15), TRI, (50, 50, 3)) is True
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_zone.py -q`
Expected: FAIL (`No module named doggy.zone`).

- [ ] **Step 3: Implement** — `src/doggy/zone.py`:
```python
from __future__ import annotations

import cv2
import numpy as np

from doggy.detection import Detection

_MIN_POLYGON_POINTS = 3


class ZoneFilter:
    """Keep only detections whose box overlaps a normalized polygon zone.

    The polygon (points in [0,1]) is rasterized to a frame-sized 0/1 mask and
    cached; the mask is rebuilt only when the points or the frame shape change.
    Fewer than 3 points means "no zone" -> every detection passes through.
    """

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None
        self._key: tuple | None = None

    def _ensure_mask(self, points: list[tuple[float, float]], shape: tuple) -> None:
        h, w = shape[0], shape[1]
        key = (tuple(points), (h, w))
        if key == self._key:
            return
        mask = np.zeros((h, w), np.uint8)
        pts = np.array([[int(x * w), int(y * h)] for x, y in points], np.int32)
        cv2.fillPoly(mask, [pts], 1)
        self._mask, self._key = mask, key

    def in_zone(self, box: tuple[int, int, int, int],
                points: list[tuple[float, float]], shape: tuple) -> bool:
        if len(points) < _MIN_POLYGON_POINTS:
            return True
        self._ensure_mask(points, shape)
        assert self._mask is not None
        h, w = shape[0], shape[1]
        x1, y1, x2, y2 = box
        x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
        y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            return False
        return bool(self._mask[y1:y2, x1:x2].any())

    def filter(self, detections: list[Detection],
               points: list[tuple[float, float]], shape: tuple) -> list[Detection]:
        if len(points) < _MIN_POLYGON_POINTS:
            return list(detections)
        return [d for d in detections if self.in_zone(d.box, points, shape)]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_zone.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doggy/zone.py tests/test_zone.py
git commit -m "feat: ZoneFilter — polygon mask + box-overlap detection filtering"
```

---

### Task 3: `Pacer` (detect-interval throttle)

**Files:**
- Create: `src/doggy/pacer.py`
- Test: `tests/test_pacer.py`

**Interfaces:**
- Produces: `Pacer(clock=time.monotonic, sleep=time.sleep)`, `.wait(interval: float) -> None` sleeps only the remainder to keep consecutive calls `interval` apart. First call never sleeps.

- [ ] **Step 1: Write failing tests** — `tests/test_pacer.py`:
```python
from doggy.pacer import Pacer

def make(clock_values):
    it = iter(clock_values)
    slept = []
    p = Pacer(clock=lambda: next(it), sleep=slept.append)
    return p, slept

def test_first_call_never_sleeps():
    p, slept = make([0.0])
    p.wait(1.0)
    assert slept == []

def test_sleeps_remainder_when_called_too_soon():
    p, slept = make([0.0, 0.3, 1.0])   # last=0.0; now=0.3 -> sleep 0.7; last=1.0
    p.wait(1.0)
    p.wait(1.0)
    assert slept == [0.7]

def test_no_sleep_when_interval_already_elapsed():
    p, slept = make([0.0, 2.0, 2.0])
    p.wait(1.0)
    p.wait(1.0)
    assert slept == []
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_pacer.py -q`
Expected: FAIL (`No module named doggy.pacer`).

- [ ] **Step 3: Implement** — `src/doggy/pacer.py`:
```python
from __future__ import annotations

import time
from typing import Callable


class Pacer:
    """Throttle a loop: `wait(interval)` sleeps only the time still needed to
    keep consecutive calls at least `interval` seconds apart. First call is free.
    Clock and sleep are injected for testing."""

    def __init__(self, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self, interval: float) -> None:
        if self._last is not None:
            remaining = interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `uv run pytest tests/test_pacer.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doggy/pacer.py tests/test_pacer.py
git commit -m "feat: Pacer — inject-clock detect-interval throttle"
```

---

### Task 4: Pipeline + annotate integration

**Files:**
- Modify: `src/doggy/pipeline.py` (`annotate`, `Pipeline.__init__`, `run_once`, `run`)
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `ZoneFilter` (Task 2), `Pacer` (Task 3), `TunableSettings.zone_enabled/zone_points/detect_interval_seconds` (Task 1).
- Produces: `annotate(frame, detections, in_zone=None, zone_points=None)` — `in_zone=None` colours all boxes as active (back-compat). Only in-zone detections reach the trigger and the `dogs` count.

- [ ] **Step 1: Write failing tests** — add to `tests/test_pipeline.py`:
```python
def test_pipeline_ignores_dogs_outside_zone(tmp_path):
    # zone = top-left triangle; a dog only in the bottom-right must NOT count/fire
    settings = Settings(zone_enabled=True,
                        zone_points=[(0.0, 0.0), (0.5, 0.0), (0.0, 0.5)],
                        confirm_seconds=0.0, window_m=1, window_n=1)
    runtime = RuntimeSettings(settings.tunable())
    outside = [Detection("dog", 0.9, (80, 80, 95, 95))]
    status = StatusStore()
    pipe = Pipeline(
        settings=settings, detector=StubDetector([outside]),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        alerter=FakeAlerter(), runtime=runtime, status=status,
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, tmp_path), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    fired = pipe.run_once(np.zeros((100, 100, 3), np.uint8))
    assert fired is False
    assert status.snapshot().dogs == 0

def test_pipeline_fires_for_dog_inside_zone(tmp_path):
    settings = Settings(zone_enabled=True,
                        zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)],
                        confirm_seconds=0.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    inside = [Detection("dog", 0.9, (5, 5, 20, 20))]
    alerter = FakeAlerter()
    pipe = Pipeline(
        settings=settings, detector=StubDetector([inside]),
        camera=FakeCamera([np.zeros((100, 100, 3), np.uint8)], loop=True),
        alerter=alerter, runtime=runtime, status=StatusStore(),
        raw_buffer=FrameBuffer(), annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, tmp_path), clock=lambda: 0.0,
        rng=random.Random(0),
    )
    assert pipe.run_once(np.zeros((100, 100, 3), np.uint8)) is True
    assert alerter.calls == 1

def test_annotate_draws_zone_polygon():
    from doggy.pipeline import annotate
    frame = np.zeros((100, 100, 3), np.uint8)
    out = annotate(frame, [], in_zone=[], zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    assert (out != 0).any()   # the polygon outline/fill was drawn
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_pipeline.py -q`
Expected: FAIL (zone not filtered; `annotate` signature).

- [ ] **Step 3: Update `annotate`** in `src/doggy/pipeline.py`. Add colour/zone constants near the existing overlay constants:
```python
_DOG_ACTIVE_COLOR = (0, 0, 255)     # red BGR — in-zone / will trigger
_DOG_IGNORED_COLOR = (150, 150, 150)  # grey — outside zone, ignored
_ZONE_COLOR = (0, 165, 255)         # orange BGR
_ZONE_ALPHA = 0.25
```
Replace `annotate` with:
```python
def annotate(frame, detections, in_zone=None, zone_points=None):
    out = frame.copy()
    h, w = frame.shape[0], frame.shape[1]
    if zone_points and len(zone_points) >= 3:
        pts = np.array([[int(x * w), int(y * h)] for x, y in zone_points], np.int32)
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], _ZONE_COLOR)
        cv2.addWeighted(overlay, _ZONE_ALPHA, out, 1 - _ZONE_ALPHA, 0, out)
        cv2.polylines(out, [pts], True, _ZONE_COLOR, _BOX_THICKNESS)
    active = detections if in_zone is None else in_zone
    for d in detections:
        color = _DOG_ACTIVE_COLOR if d in active else _DOG_IGNORED_COLOR
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, _BOX_THICKNESS)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - _LABEL_Y_OFFSET)),
                    cv2.FONT_HERSHEY_SIMPLEX, _LABEL_FONT_SCALE, color, _LABEL_THICKNESS)
    return out
```

- [ ] **Step 4: Wire `ZoneFilter` + `Pacer` into `Pipeline`.** In `pipeline.py` add imports:
```python
from doggy.zone import ZoneFilter
from doggy.pacer import Pacer
```
In `Pipeline.__init__`, after `self.trigger = ...`:
```python
        self.zone = ZoneFilter()
        self.pacer = Pacer(clock=clock)
```
Replace `run_once` body with (keeps exactly one `self.clock()` call):
```python
    def run_once(self, frame: np.ndarray) -> bool:
        now = self.clock()
        cfg = self.runtime.get()
        detections = self.detector.detect(frame)
        points = cfg.zone_points if cfg.zone_enabled else []
        in_zone = self.zone.filter(detections, points, frame.shape)
        self.annotated_buffer.set(annotate(frame, detections, in_zone, points))
        top = max((d.confidence for d in in_zone), default=0.0)
        fired = self.trigger.update(in_zone, now)
        muted = not self.safety.allow_fire(now)
        if fired and not muted:
            self.alerter.alert()
            event = self.safety.record_fire(frame, top, now)
            self.status.add_event(event)
            self.status.update(last_fire_ts=event["ts"], last_fire_thumb=event["thumb"])
        self.status.update(state=self.trigger.state.value, confidence=round(top, CONFIDENCE_DECIMALS),
                           dogs=len(in_zone),
                           fires_this_hour=self.safety.fires_last_hour(now), muted=muted)
        return fired and not muted
```
In `run`, add pacing just before `self.run_once(frame)`:
```python
            self.pacer.wait(self.runtime.get().detect_interval_seconds)
            self.run_once(frame)
```

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (including the pre-existing `test_pipeline_counts_multiple_dogs` and `test_pipeline_fires_after_confirmation`, which call `annotate`/`run_once` unchanged-compatibly).

- [ ] **Step 6: Commit**

```bash
git add src/doggy/pipeline.py tests/test_pipeline.py
git commit -m "feat: pipeline filters detections by zone + paces inference; annotate draws zone"
```

---

### Task 5: Frontend — draw zone + interval slider

**Files:**
- Modify: `src/doggy/static/index.html`
- Test: `tests/test_web.py` (smoke: page serves the new controls)

**Interfaces:**
- Consumes: `PATCH /api/settings` with `{zone_enabled, zone_points}` and `{detect_interval_seconds}`; `GET /api/status` `settings.zone_points`.

- [ ] **Step 1: Write failing smoke test** — add to `tests/test_web.py`:
```python
def test_index_has_zone_controls():
    from fastapi.testclient import TestClient
    from doggy.web import create_app
    from doggy.config import Settings
    from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
    from doggy.alerter import FakeAlerter
    s = Settings()
    app = create_app(s, RuntimeSettings(s.tunable()), FrameBuffer(), StatusStore(), FakeAlerter())
    html = TestClient(app).get("/").text
    assert "Finish zone" in html and "Clear zone" in html
    assert "detect_interval_seconds" in html
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_web.py::test_index_has_zone_controls -q`
Expected: FAIL.

- [ ] **Step 3: Implement frontend.** In `src/doggy/static/index.html`:

(a) Wrap the video and add an overlay canvas + buttons — replace the video panel `<div class="panel"><img .../></div>` with:
```html
  <div class="panel">
    <div id="vidwrap" style="position:relative; display:inline-block;">
      <img id="live" src="/stream.mjpg" alt="live" />
      <canvas id="zonecanvas" style="position:absolute; left:0; top:0; cursor:crosshair;"></canvas>
    </div>
    <div>
      <button onclick="finishZone()">Finish zone</button>
      <button onclick="clearZone()">Clear zone</button>
      <span id="zonehint" style="font-size:.8rem;color:#aaa;">click to place zone points</span>
    </div>
  </div>
```

(b) Add the interval slider to the knobs panel (after the `max_fires_per_hour` slider):
```html
    <label>detect_interval_seconds <span id="detect_interval_seconds_v"></span></label>
    <input id="detect_interval_seconds" type="range" min="0" max="3" step="0.1" />
```

(c) Add `"detect_interval_seconds"` to the `KNOBS` array.

(d) Add zone-drawing JS before `setInterval(poll, 500);`:
```javascript
const img = document.getElementById("live");
const canvas = document.getElementById("zonecanvas");
const ctx = canvas.getContext("2d");
let drawing = [];         // in-progress points, normalized [0,1]
function sizeCanvas(){ canvas.width = img.clientWidth; canvas.height = img.clientHeight; }
img.addEventListener("load", sizeCanvas);
window.addEventListener("resize", () => { sizeCanvas(); });
canvas.addEventListener("click", e => {
  const r = img.getBoundingClientRect();
  drawing.push([(e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height]);
  drawZone(drawing, true);
});
function drawZone(points, inProgress){
  sizeCanvas();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!points.length) return;
  ctx.strokeStyle = inProgress ? "#0af" : "#fa0";
  ctx.lineWidth = 2; ctx.beginPath();
  points.forEach((p, i) => {
    const x = p[0] * canvas.width, y = p[1] * canvas.height;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    ctx.fillStyle = "#0af"; ctx.fillRect(x - 3, y - 3, 6, 6);
  });
  if (!inProgress) ctx.closePath();
  ctx.stroke();
}
async function finishZone(){
  if (drawing.length < 3){ alert("place at least 3 points"); return; }
  await patch({zone_enabled: true, zone_points: drawing});
  drawing = [];
}
async function clearZone(){
  drawing = [];
  await patch({zone_enabled: false, zone_points: []});
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}
```

(e) In `poll()`, after updating knobs, redraw the saved zone only when not mid-draw (the MJPEG already shows it server-side, but this gives the canvas parity):
```javascript
  if (drawing.length === 0 && s.settings.zone_enabled){ drawZone(s.settings.zone_points, false); }
```

- [ ] **Step 4: Run smoke test + full suite**

Run: `uv run pytest tests/test_web.py::test_index_has_zone_controls -q && uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/doggy/static/index.html tests/test_web.py
git commit -m "feat: dashboard — click-draw polygon zone + detect-interval slider"
```

---

## Self-Review

**Spec coverage:** zone_enabled/zone_points/detect_interval_seconds (Task 1) ✓; ZoneFilter mask + box overlap (Task 2) ✓; Pacer (Task 3) ✓; pipeline filter + dogs count + pacing + annotate polygon/colors (Task 4) ✓; frontend draw/clear/slider + persistence via existing Save (Task 5) ✓; `_write_env` JSON round-trip (Task 1) ✓. Tests cover in/out/edge overlap, pass-through, mask rebuild, pacer branches, pipeline in/out-of-zone, annotate draw, env round-trip, page controls.

**Placeholders:** none — full code in every step.

**Type consistency:** `ZoneFilter.filter(detections, points, shape)` and `.in_zone(box, points, shape)` used identically in Task 4; `annotate(frame, detections, in_zone=None, zone_points=None)` matches its Task-4 call and the back-compat calls in the existing tests; `Pacer.wait(interval)` matches. `zone_points` is `list[tuple[float,float]]` throughout; normalized [0,1]; ≥3 rule enforced in `ZoneFilter` and `finishZone`.

## Deployment (after merge, run from repo root)

The Pi is hardened (no internet) but this needs no new packages. Redeploy code + restart:
```bash
rsync -az --exclude .venv --exclude .git --exclude models --exclude events --exclude .env \
  ./ doggy@192.168.50.151:doggy/
ssh doggy@192.168.50.151 'sudo systemctl restart doggy'
```
Then open `http://192.168.50.151:8000`, click the counter polygon, **Finish zone**, **Save to .env**.
