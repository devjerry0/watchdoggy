# Doggy Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local dog-presence detector that watches a USB webcam, plays a deterrent sound clip when a dog is confirmed in view, and exposes a stupid-simple localhost dashboard for live viewing + live knob tuning.

**Architecture:** A threaded Python pipeline — a capture thread keeps only the newest webcam frame, a detect thread runs YOLO26n and drives a time-based trigger state machine guarded by a safety governor, and the alerter plays clips fire-and-forget. Config is pydantic-settings from `DOGGY_*` env vars; the live-tunable subset is a shared thread-safe object the FastAPI dashboard patches at runtime.

**Tech Stack:** Python 3.11+, `uv`, Ultralytics YOLO26n, OpenCV, `sounddevice`+`soundfile`, `pydantic-settings`, FastAPI+uvicorn, pytest.

**Spec:** `docs/superpowers/specs/2026-07-05-doggy-detector-design.md`

## Global Constraints

- **Inference is local only** — no cloud / OpenRouter.
- **v1 camera = USB webcam** via `cv2.VideoCapture` (no Pi CSI camera). Dev camera is a Logitech C922; `DOGGY_CAMERA_INDEX` selects it.
- **Env vars are the single config source of truth**, `DOGGY_` prefix, loaded by `pydantic-settings` (+ `.env`). No YAML/JSON, no `requirements.txt`.
- **Dependency manager is `uv`** with `pyproject.toml`.
- **License: AGPL-3.0** (repo relicensed from Apache-2.0 to match Ultralytics).
- **Trigger is time-based**, not frame-count (frame-rate independent); M-of-N window for flicker tolerance; jittered cooldown.
- **Safety envelope required:** rate limit → auto-mute, master off switch, event log with thumbnails, volume cap.
- **Audio:** `sounddevice`+`soundfile` (CoreAudio on Mac / ALSA on Pi).
- **Device auto-select** (MPS on Mac, CPU/NCNN on Pi) — never hardcode `mps`.
- **Web UI:** localhost-bound (`127.0.0.1`) by default, no auth; MJPEG encoded only when a client is connected, throttled/downscaled so it never starves detection. Frontend is one static `index.html`, vanilla JS, no build step.
- **Package layout:** `src/doggy/`, tests in `tests/`.

---

### Task 1: Project scaffold, tooling & license

**Files:**
- Create: `pyproject.toml`, `src/doggy/__init__.py`, `tests/__init__.py`, `tests/test_smoke.py`, `.gitignore`, `.python-version`
- Replace: `LICENSE` (Apache-2.0 → AGPL-3.0)

**Interfaces:**
- Consumes: nothing.
- Produces: importable package `doggy` (exposes `__version__: str`); `uv run pytest` works.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "doggy"
version = "0.1.0"
description = "Local dog-presence detector with a deterrent speaker and a localhost dashboard"
requires-python = ">=3.11"
license = "AGPL-3.0-or-later"
dependencies = [
    "ultralytics>=8.3.0",
    "torch>=2.2",
    "torchvision>=0.17",
    "opencv-python>=4.10",
    "numpy>=1.26",
    "sounddevice>=0.4.7",
    "soundfile>=0.12",
    "pydantic-settings>=2.4",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "httpx>=0.27",
]

# CPU-only PyTorch. On Linux (x86_64 CI or ARM), the default PyPI `torch` wheel
# bundles CUDA (~2.5GB) which this project never uses — pin torch/torchvision to
# the CPU index there. macOS arm64 keeps the default PyPI wheel (already CPU+MPS,
# no CUDA variant exists), so the CPU index is gated to Linux only.
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
torchvision = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/doggy"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = ["slow: needs model weights / real hardware (deselect with -m 'not slow')"]
```

- [ ] **Step 2: Create `.python-version` and `.gitignore`**

`.python-version`:
```
3.11
```

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
.env
models/
events/
*.mp4
.pytest_cache/
```

- [ ] **Step 3: Create the package and a smoke test**

`src/doggy/__init__.py`:
```python
__version__ = "0.1.0"
```

`tests/__init__.py`: (empty file)

`tests/test_smoke.py`:
```python
import doggy


def test_package_imports_and_has_version():
    assert isinstance(doggy.__version__, str)
    assert doggy.__version__
```

- [ ] **Step 4: Sync the environment and run the smoke test**

Run:
```bash
uv sync
uv run pytest tests/test_smoke.py -v
```
Expected: PASS (1 passed). `uv sync` creates `.venv` and `uv.lock`.

- [ ] **Step 5: Relicense the repo to AGPL-3.0**

Replace the contents of `LICENSE` with the canonical GNU AGPL-3.0 text (the standard ~34KB license from https://www.gnu.org/licenses/agpl-3.0.txt — copy it verbatim). Do not paraphrase; it must be the exact license text.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src/doggy/__init__.py tests/__init__.py tests/test_smoke.py LICENSE
git commit -m "chore: scaffold uv project, smoke test, relicense to AGPL-3.0"
```

---

### Task 2: Configuration (`pydantic-settings`)

**Files:**
- Create: `src/doggy/config.py`, `.env.example`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `TunableSettings(BaseModel)` — frozen; fields `confidence: float`, `confirm_seconds: float`, `window_m: int`, `window_n: int`, `cooldown_min_seconds: float`, `cooldown_max_seconds: float`, `max_volume: float`, `safety_enabled: bool`, `max_fires_per_hour: int`, `clips_dir: Path`, `log_level: str`. Validators: `window_m <= window_n`, `cooldown_min_seconds <= cooldown_max_seconds`.
  - `Settings(TunableSettings, BaseSettings)` — adds structural fields `camera_backend: str`, `camera_index: int`, `camera_path: Path | None`, `model_path: Path`, `confidence`… (inherited), `alerter_backend: str`, `audio_device: str | None`, `event_log_dir: Path`, `web_enabled: bool`, `web_host: str`, `web_port: int`. `model_config` uses `env_prefix="DOGGY_"`, `.env` file, case-insensitive. Method `tunable() -> TunableSettings`.
  - `load_settings() -> Settings`.

- [ ] **Step 1: Write failing tests**

`tests/test_config.py`:
```python
import pytest
from pydantic import ValidationError

from doggy.config import Settings, TunableSettings, load_settings


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # avoid picking up a real .env
    s = load_settings()
    assert s.confidence == 0.55
    assert s.window_m == 4 and s.window_n == 6
    assert s.camera_index == 0
    assert s.web_host == "127.0.0.1"
    assert s.web_port == 8000


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOGGY_CONFIDENCE", "0.7")
    monkeypatch.setenv("DOGGY_CAMERA_INDEX", "1")
    s = load_settings()
    assert s.confidence == 0.7
    assert s.camera_index == 1


def test_window_validation():
    with pytest.raises(ValidationError):
        TunableSettings(window_m=7, window_n=6)


def test_cooldown_validation():
    with pytest.raises(ValidationError):
        TunableSettings(cooldown_min_seconds=30, cooldown_max_seconds=10)


def test_confidence_range():
    with pytest.raises(ValidationError):
        TunableSettings(confidence=1.5)


def test_tunable_subset_extracted():
    s = Settings(confidence=0.6)
    t = s.tunable()
    assert isinstance(t, TunableSettings)
    assert t.confidence == 0.6
    assert not hasattr(t, "camera_index")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.config'`.

- [ ] **Step 3: Implement `config.py`**

```python
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TunableSettings(BaseModel):
    """The subset of config that can be changed live via the web UI."""

    model_config = {"frozen": True}

    confidence: float = Field(0.55, ge=0.0, le=1.0)
    confirm_seconds: float = Field(1.2, ge=0.0)
    window_m: int = Field(4, ge=1)
    window_n: int = Field(6, ge=1)
    cooldown_min_seconds: float = Field(12.0, ge=0.0)
    cooldown_max_seconds: float = Field(20.0, ge=0.0)
    max_volume: float = Field(0.8, ge=0.0, le=1.0)
    safety_enabled: bool = True
    max_fires_per_hour: int = Field(6, ge=0)
    clips_dir: Path = Path("sounds")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _check_ranges(self) -> "TunableSettings":
        if self.window_m > self.window_n:
            raise ValueError("window_m must be <= window_n")
        if self.cooldown_min_seconds > self.cooldown_max_seconds:
            raise ValueError("cooldown_min_seconds must be <= cooldown_max_seconds")
        return self


class Settings(TunableSettings, BaseSettings):
    """Full config: structural (restart-required) fields + the tunable subset."""

    model_config = SettingsConfigDict(
        env_prefix="DOGGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    camera_backend: str = "opencv"  # opencv | file
    camera_index: int = 0
    camera_path: Path | None = None
    model_path: Path = Path("models/yolo26n.pt")
    alerter_backend: str = "sounddevice"  # sounddevice | command | log
    audio_device: str | None = None
    event_log_dir: Path = Path("events")
    web_enabled: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    def tunable(self) -> TunableSettings:
        fields = TunableSettings.model_fields
        return TunableSettings(**{name: getattr(self, name) for name in fields})


def load_settings() -> Settings:
    return Settings()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Create `.env.example`**

```dotenv
# Copy to .env for local dev. All settings are optional; defaults shown.
# Camera
DOGGY_CAMERA_BACKEND=opencv
DOGGY_CAMERA_INDEX=1            # C922 is usually 1 (FaceTime built-in is 0)
# DOGGY_CAMERA_PATH=fixtures/dog_walk.mp4   # only for camera_backend=file
# Detector
DOGGY_MODEL_PATH=models/yolo26n.pt
DOGGY_CONFIDENCE=0.55
# Trigger
DOGGY_CONFIRM_SECONDS=1.2
DOGGY_WINDOW_M=4
DOGGY_WINDOW_N=6
DOGGY_COOLDOWN_MIN_SECONDS=12
DOGGY_COOLDOWN_MAX_SECONDS=20
# Alerter
DOGGY_ALERTER_BACKEND=sounddevice
DOGGY_CLIPS_DIR=sounds
# DOGGY_AUDIO_DEVICE=
DOGGY_MAX_VOLUME=0.8
# Safety
DOGGY_SAFETY_ENABLED=true
DOGGY_MAX_FIRES_PER_HOUR=6
DOGGY_EVENT_LOG_DIR=events
# Web
DOGGY_WEB_ENABLED=true
DOGGY_WEB_HOST=127.0.0.1
DOGGY_WEB_PORT=8000
DOGGY_LOG_LEVEL=INFO
```

- [ ] **Step 6: Commit**

```bash
git add src/doggy/config.py tests/test_config.py .env.example
git commit -m "feat: pydantic-settings config with tunable subset"
```

---

### Task 3: Core types & shared thread-safe state

**Files:**
- Create: `src/doggy/detection.py`, `src/doggy/state.py`
- Test: `tests/test_state.py`

**Interfaces:**
- Consumes: `TunableSettings` (Task 2).
- Produces:
  - `Detection` frozen dataclass: `label: str`, `confidence: float`, `box: tuple[int, int, int, int]`.
  - `RuntimeSettings` — thread-safe holder: `__init__(tunable: TunableSettings)`, `get() -> TunableSettings`, `update(tunable: TunableSettings) -> None`.
  - `FrameBuffer` — latest-only slot: `set(frame) -> None`, `get() -> np.ndarray | None`.
  - `Status` dataclass + `StatusStore` — `update(**kwargs) -> None`, `snapshot() -> Status`, `add_event(event: dict) -> None`, `events() -> list[dict]`.

- [ ] **Step 1: Write failing tests**

`tests/test_state.py`:
```python
import numpy as np

from doggy.config import TunableSettings
from doggy.detection import Detection
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore


def test_detection_is_frozen():
    d = Detection(label="dog", confidence=0.9, box=(1, 2, 3, 4))
    assert d.box == (1, 2, 3, 4)
    try:
        d.confidence = 0.1  # type: ignore[misc]
        assert False, "should be frozen"
    except Exception:
        pass


def test_runtime_settings_atomic_swap():
    rs = RuntimeSettings(TunableSettings(confidence=0.5))
    assert rs.get().confidence == 0.5
    rs.update(TunableSettings(confidence=0.9))
    assert rs.get().confidence == 0.9


def test_frame_buffer_keeps_latest():
    fb = FrameBuffer()
    assert fb.get() is None
    fb.set(np.zeros((2, 2), dtype=np.uint8))
    fb.set(np.ones((2, 2), dtype=np.uint8))
    assert fb.get().sum() == 4  # the newest frame won


def test_status_store_update_and_events():
    ss = StatusStore()
    ss.update(state="CONFIRMING", fps=5.0)
    snap = ss.snapshot()
    assert snap.state == "CONFIRMING"
    assert snap.fps == 5.0
    ss.add_event({"ts": 1.0, "confidence": 0.9})
    assert ss.events()[-1]["confidence"] == 0.9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.detection'`.

- [ ] **Step 3: Implement `detection.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
```

- [ ] **Step 4: Implement `state.py`**

```python
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from doggy.config import TunableSettings


class RuntimeSettings:
    """Thread-safe holder for the live-tunable settings, swapped atomically."""

    def __init__(self, tunable: TunableSettings) -> None:
        self._lock = threading.Lock()
        self._tunable = tunable

    def get(self) -> TunableSettings:
        with self._lock:
            return self._tunable

    def update(self, tunable: TunableSettings) -> None:
        with self._lock:
            self._tunable = tunable


class FrameBuffer:
    """Holds only the most recent frame; setters overwrite (drop-oldest)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame

    def get(self) -> np.ndarray | None:
        with self._lock:
            return self._frame


@dataclass
class Status:
    state: str = "IDLE"
    fps: float = 0.0
    confidence: float = 0.0
    fires_this_hour: int = 0
    last_fire_ts: float | None = None
    last_fire_thumb: str | None = None
    muted: bool = False


class StatusStore:
    def __init__(self, max_events: int = 50) -> None:
        self._lock = threading.Lock()
        self._status = Status()
        self._events: deque[dict] = deque(maxlen=max_events)

    def update(self, **kwargs) -> None:
        with self._lock:
            self._status = replace(self._status, **kwargs)

    def snapshot(self) -> Status:
        with self._lock:
            return self._status

    def add_event(self, event: dict) -> None:
        with self._lock:
            self._events.append(event)

    def events(self) -> list[dict]:
        with self._lock:
            return list(self._events)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: PASS (4 passed).

- [ ] **Step 6: Commit**

```bash
git add src/doggy/detection.py src/doggy/state.py tests/test_state.py
git commit -m "feat: Detection type and thread-safe shared state"
```

---

### Task 4: Trigger state machine

**Files:**
- Create: `src/doggy/trigger.py`
- Test: `tests/test_trigger.py`

**Interfaces:**
- Consumes: `Detection` (Task 3), `RuntimeSettings` (Task 3), `TunableSettings` (Task 2).
- Produces:
  - `TriggerState` enum: `IDLE`, `CONFIRMING`, `COOLDOWN`.
  - `TriggerLogic(runtime: RuntimeSettings, rng=random.Random())`:
    - `update(detections: list[Detection], now: float) -> bool` — returns `True` exactly on the fire edge.
    - `state: TriggerState` attribute (read for status).

- [ ] **Step 1: Write failing tests**

`tests/test_trigger.py`:
```python
import random

from doggy.config import TunableSettings
from doggy.detection import Detection
from doggy.state import RuntimeSettings
from doggy.trigger import TriggerLogic, TriggerState

DOG = [Detection(label="dog", confidence=0.9, box=(0, 0, 10, 10))]
NONE: list[Detection] = []


def make(**over):
    base = dict(confirm_seconds=1.0, window_m=2, window_n=3,
               cooldown_min_seconds=10, cooldown_max_seconds=10, confidence=0.5)
    base.update(over)
    return TriggerLogic(RuntimeSettings(TunableSettings(**base)),
                        rng=random.Random(0))


def test_single_frame_does_not_fire():
    t = make()
    assert t.update(DOG, now=0.0) is False
    assert t.state is TriggerState.CONFIRMING


def test_fires_after_confirm_seconds():
    t = make()
    assert t.update(DOG, now=0.0) is False
    assert t.update(DOG, now=0.5) is False
    fired = t.update(DOG, now=1.0)  # 1.0s >= confirm_seconds
    assert fired is True
    assert t.state is TriggerState.COOLDOWN


def test_low_confidence_ignored():
    t = make(confidence=0.8)
    low = [Detection(label="dog", confidence=0.6, box=(0, 0, 1, 1))]
    assert t.update(low, now=0.0) is False
    assert t.state is TriggerState.IDLE


def test_lost_dog_resets_to_idle():
    t = make()
    t.update(DOG, now=0.0)
    t.update(NONE, now=0.1)
    t.update(NONE, now=0.2)  # window no longer M-of-N
    assert t.state is TriggerState.IDLE


def test_flicker_tolerated_by_m_of_n():
    t = make()  # window_m=2, window_n=3
    assert t.update(DOG, now=0.0) is False
    assert t.update(NONE, now=0.5) is False   # one dropped frame
    fired = t.update(DOG, now=1.0)            # 2 of last 3 had a dog, 1.0s elapsed
    assert fired is True


def test_cooldown_blocks_refire():
    t = make()
    t.update(DOG, now=0.0)
    assert t.update(DOG, now=1.0) is True     # fires, cooldown=10s
    assert t.update(DOG, now=2.0) is False    # still cooling down
    assert t.state is TriggerState.COOLDOWN


def test_refires_after_cooldown_with_fresh_confirm():
    t = make()
    t.update(DOG, now=0.0)
    assert t.update(DOG, now=1.0) is True
    assert t.update(DOG, now=12.0) is False   # cooldown expired -> fresh CONFIRMING
    assert t.update(DOG, now=13.0) is True     # confirmed again
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_trigger.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.trigger'`.

- [ ] **Step 3: Implement `trigger.py`**

```python
from __future__ import annotations

import random
from collections import deque
from enum import Enum

from doggy.detection import Detection
from doggy.state import RuntimeSettings


class TriggerState(str, Enum):
    IDLE = "IDLE"
    CONFIRMING = "CONFIRMING"
    COOLDOWN = "COOLDOWN"


class TriggerLogic:
    """Time-based confirmation + M-of-N window + jittered cooldown.

    update() returns True exactly on the transition into COOLDOWN (the fire edge).
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()
        self.state = TriggerState.IDLE
        self._confirm_start: float = 0.0
        self._cooldown_until: float = 0.0
        self._window: deque[bool] = deque()

    def update(self, detections: list[Detection], now: float) -> bool:
        cfg = self._runtime.get()
        has_dog = any(
            d.label == "dog" and d.confidence >= cfg.confidence for d in detections
        )

        self._window.append(has_dog)
        while len(self._window) > cfg.window_n:
            self._window.popleft()
        m_of_n = sum(self._window) >= cfg.window_m

        if self.state is TriggerState.COOLDOWN:
            if now >= self._cooldown_until:
                self.state = TriggerState.IDLE
            else:
                return False

        if self.state is TriggerState.IDLE:
            if has_dog:
                self.state = TriggerState.CONFIRMING
                self._confirm_start = now
            return False

        # CONFIRMING
        if not m_of_n:
            self.state = TriggerState.IDLE
            return False
        if now - self._confirm_start >= cfg.confirm_seconds:
            self._cooldown_until = now + self._rng.uniform(
                cfg.cooldown_min_seconds, cfg.cooldown_max_seconds
            )
            self.state = TriggerState.COOLDOWN
            return True
        return False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_trigger.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add src/doggy/trigger.py tests/test_trigger.py
git commit -m "feat: time-based trigger state machine with M-of-N window"
```

---

### Task 5: Safety governor

**Files:**
- Create: `src/doggy/safety.py`
- Test: `tests/test_safety.py`

**Interfaces:**
- Consumes: `RuntimeSettings` (Task 3).
- Produces:
  - `SafetyGovernor(runtime: RuntimeSettings, event_log_dir: Path)`:
    - `allow_fire(now: float) -> bool` — False if disabled or rate limit hit.
    - `record_fire(frame: np.ndarray, confidence: float, now: float) -> dict` — writes a thumbnail + appends a jsonl line; returns the event dict.
    - `fires_last_hour(now: float) -> int`.

- [ ] **Step 1: Write failing tests**

`tests/test_safety.py`:
```python
import numpy as np

from doggy.config import TunableSettings
from doggy.safety import SafetyGovernor
from doggy.state import RuntimeSettings

FRAME = np.zeros((16, 16, 3), dtype=np.uint8)


def gov(tmp_path, **over):
    base = dict(safety_enabled=True, max_fires_per_hour=2)
    base.update(over)
    rs = RuntimeSettings(TunableSettings(**base))
    return SafetyGovernor(rs, event_log_dir=tmp_path)


def test_allows_when_enabled_and_under_limit(tmp_path):
    g = gov(tmp_path)
    assert g.allow_fire(now=0.0) is True


def test_master_off_switch_blocks(tmp_path):
    g = gov(tmp_path, safety_enabled=False)
    assert g.allow_fire(now=0.0) is False


def test_rate_limit_blocks_after_max(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, now=0.0)
    g.record_fire(FRAME, 0.9, now=10.0)
    assert g.allow_fire(now=20.0) is False


def test_rate_limit_window_rolls_off(tmp_path):
    g = gov(tmp_path, max_fires_per_hour=2)
    g.record_fire(FRAME, 0.9, now=0.0)
    g.record_fire(FRAME, 0.9, now=10.0)
    assert g.allow_fire(now=3601.0) is True  # first fire aged out of the hour


def test_record_fire_writes_thumbnail_and_log(tmp_path):
    g = gov(tmp_path)
    event = g.record_fire(FRAME, 0.87, now=123.0)
    thumb = tmp_path / event["thumb"]
    assert thumb.exists()
    assert (tmp_path / "events.jsonl").exists()
    assert event["confidence"] == 0.87
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_safety.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.safety'`.

- [ ] **Step 3: Implement `safety.py`**

```python
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from doggy.state import RuntimeSettings

_HOUR = 3600.0


class SafetyGovernor:
    def __init__(self, runtime: RuntimeSettings, event_log_dir: Path) -> None:
        self._runtime = runtime
        self._dir = Path(event_log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._fires: deque[float] = deque()

    def _prune(self, now: float) -> None:
        while self._fires and now - self._fires[0] >= _HOUR:
            self._fires.popleft()

    def fires_last_hour(self, now: float) -> int:
        self._prune(now)
        return len(self._fires)

    def allow_fire(self, now: float) -> bool:
        cfg = self._runtime.get()
        if not cfg.safety_enabled:
            return False
        return self.fires_last_hour(now) < cfg.max_fires_per_hour

    def record_fire(self, frame: np.ndarray, confidence: float, now: float) -> dict:
        self._fires.append(now)
        thumb_name = f"fire_{now:.3f}.jpg"
        cv2.imwrite(str(self._dir / thumb_name), frame)
        event = {"ts": now, "confidence": round(float(confidence), 3), "thumb": thumb_name}
        with (self._dir / "events.jsonl").open("a") as fh:
            fh.write(json.dumps(event) + "\n")
        return event
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_safety.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/doggy/safety.py tests/test_safety.py
git commit -m "feat: safety governor with rate limit and event log"
```

---

### Task 6: Camera abstraction

**Files:**
- Create: `src/doggy/camera.py`
- Test: `tests/test_camera.py`

**Interfaces:**
- Consumes: `Settings` (Task 2).
- Produces:
  - `Camera` protocol: `frames() -> Iterator[np.ndarray]`, `close() -> None`.
  - `FakeCamera(frames: list[np.ndarray], loop: bool = False)`.
  - `OpenCVCamera(index: int, max_reconnects: int = 5)`.
  - `build_camera(settings: Settings) -> Camera`.

- [ ] **Step 1: Write failing tests**

`tests/test_camera.py`:
```python
import numpy as np

from doggy.camera import FakeCamera, build_camera
from doggy.config import Settings


def test_fake_camera_yields_given_frames():
    frames = [np.full((4, 4), i, dtype=np.uint8) for i in range(3)]
    cam = FakeCamera(frames)
    out = list(cam.frames())
    assert len(out) == 3
    assert out[1][0, 0] == 1
    cam.close()


def test_build_camera_file_backend_uses_fake(tmp_path):
    # camera_backend=file with no path still returns a Camera object
    s = Settings(camera_backend="file", camera_path=None)
    cam = build_camera(s)
    assert hasattr(cam, "frames")
    cam.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_camera.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.camera'`.

- [ ] **Step 3: Implement `camera.py`**

```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Protocol

import cv2
import numpy as np

from doggy.config import Settings


class Camera(Protocol):
    def frames(self) -> Iterator[np.ndarray]: ...
    def close(self) -> None: ...


class FakeCamera:
    """Yields a fixed list of frames (in-memory) or, via from_video, a file."""

    def __init__(self, frames: list[np.ndarray], loop: bool = False) -> None:
        self._frames = frames
        self._loop = loop

    @classmethod
    def from_video(cls, path: Path, loop: bool = False) -> "FakeCamera":
        cap = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return cls(frames, loop=loop)

    def frames(self) -> Iterator[np.ndarray]:
        while True:
            for f in self._frames:
                yield f
            if not self._loop:
                return

    def close(self) -> None:
        self._frames = []


class OpenCVCamera:
    """USB webcam via cv2.VideoCapture; reconnects on transient read failures."""

    def __init__(self, index: int, max_reconnects: int = 5) -> None:
        self._index = index
        self._max_reconnects = max_reconnects
        self._cap = cv2.VideoCapture(index)

    def frames(self) -> Iterator[np.ndarray]:
        failures = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                failures += 1
                if failures > self._max_reconnects:
                    raise RuntimeError(f"camera {self._index} lost after {failures} failures")
                self._cap.release()
                time.sleep(0.5)
                self._cap = cv2.VideoCapture(self._index)
                continue
            failures = 0
            yield frame

    def close(self) -> None:
        self._cap.release()


def build_camera(settings: Settings) -> Camera:
    if settings.camera_backend == "file":
        if settings.camera_path:
            return FakeCamera.from_video(settings.camera_path, loop=True)
        return FakeCamera([], loop=False)
    return OpenCVCamera(settings.camera_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_camera.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/doggy/camera.py tests/test_camera.py
git commit -m "feat: camera abstraction (OpenCV webcam + fake)"
```

---

### Task 7: Alerter

**Files:**
- Create: `src/doggy/alerter.py`
- Test: `tests/test_alerter.py`

**Interfaces:**
- Consumes: `Settings` (Task 2), `RuntimeSettings` (Task 3).
- Produces:
  - `Alerter` protocol: `alert() -> None`.
  - `pick_clip(clips_dir: Path, rng: random.Random) -> Path | None`.
  - `FakeAlerter` — records calls in `.calls: int`.
  - `SoundDeviceAlerter(runtime, rng)` — non-blocking playback of a random clip.
  - `CommandAlerter(runtime)` — shells `afplay`/`aplay`.
  - `build_alerter(settings, runtime) -> Alerter`.

- [ ] **Step 1: Write failing tests**

`tests/test_alerter.py`:
```python
import random

from doggy.alerter import FakeAlerter, pick_clip


def test_fake_alerter_counts_calls():
    a = FakeAlerter()
    a.alert()
    a.alert()
    assert a.calls == 2


def test_pick_clip_none_when_empty(tmp_path):
    assert pick_clip(tmp_path, random.Random(0)) is None


def test_pick_clip_is_deterministic_with_seed(tmp_path):
    for name in ["a.wav", "b.wav", "c.wav"]:
        (tmp_path / name).write_bytes(b"RIFF")
    chosen = pick_clip(tmp_path, random.Random(0))
    assert chosen.suffix == ".wav"
    assert pick_clip(tmp_path, random.Random(0)) == chosen  # same seed -> same pick
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_alerter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.alerter'`.

- [ ] **Step 3: Implement `alerter.py`**

```python
from __future__ import annotations

import random
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from doggy.config import Settings
from doggy.state import RuntimeSettings

_CLIP_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def pick_clip(clips_dir: Path, rng: random.Random) -> Path | None:
    clips = sorted(p for p in Path(clips_dir).glob("*") if p.suffix.lower() in _CLIP_EXTS)
    if not clips:
        return None
    return rng.choice(clips)


class Alerter(Protocol):
    def alert(self) -> None: ...


class FakeAlerter:
    def __init__(self) -> None:
        self.calls = 0

    def alert(self) -> None:
        self.calls += 1


class SoundDeviceAlerter:
    """Plays a random clip on a background thread (fire-and-forget)."""

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()

    def alert(self) -> None:
        cfg = self._runtime.get()
        clip = pick_clip(cfg.clips_dir, self._rng)
        if clip is None:
            return
        threading.Thread(target=self._play, args=(clip, cfg.max_volume), daemon=True).start()

    def _play(self, clip: Path, volume: float) -> None:
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(str(clip), dtype="float32")
        sd.play(data * max(0.0, min(1.0, volume)), samplerate)
        sd.wait()


class CommandAlerter:
    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()

    def alert(self) -> None:
        cfg = self._runtime.get()
        clip = pick_clip(cfg.clips_dir, self._rng)
        if clip is None:
            return
        cmd = "afplay" if sys.platform == "darwin" else "aplay"
        subprocess.Popen([cmd, str(clip)])  # non-blocking


def build_alerter(settings: Settings, runtime: RuntimeSettings) -> Alerter:
    if settings.alerter_backend == "log":
        return FakeAlerter()
    if settings.alerter_backend == "command":
        return CommandAlerter(runtime)
    return SoundDeviceAlerter(runtime)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_alerter.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/doggy/alerter.py tests/test_alerter.py
git commit -m "feat: alerter with random clip selection and portable audio"
```

---

### Task 8: Detector (YOLO26n wrapper + stub)

**Files:**
- Create: `src/doggy/detector.py`
- Test: `tests/test_detector.py`

**Interfaces:**
- Consumes: `Settings` (Task 2), `Detection` (Task 3).
- Produces:
  - `Detector` protocol: `detect(frame: np.ndarray) -> list[Detection]`.
  - `StubDetector(scripted: list[list[Detection]])` — pops one list per call (for pipeline/web tests).
  - `YoloDetector(model_path, confidence, device=None)` — wraps Ultralytics, filters to `dog`.
  - `select_device() -> str` — `"mps"` on Apple Silicon, else `"cpu"`.
  - `build_detector(settings: Settings) -> Detector`.

- [ ] **Step 1: Write failing tests**

`tests/test_detector.py`:
```python
import numpy as np
import pytest

from doggy.detection import Detection
from doggy.detector import StubDetector, select_device


def test_stub_detector_scripts_returns():
    stub = StubDetector([[Detection("dog", 0.9, (0, 0, 1, 1))], []])
    assert stub.detect(np.zeros((2, 2))) == [Detection("dog", 0.9, (0, 0, 1, 1))]
    assert stub.detect(np.zeros((2, 2))) == []
    assert stub.detect(np.zeros((2, 2))) == []  # exhausted -> empty


def test_select_device_returns_known_value():
    assert select_device() in {"mps", "cpu", "cuda"}


@pytest.mark.slow
def test_yolo_detects_dog_and_ignores_empty_room():
    from pathlib import Path
    from doggy.detector import YoloDetector
    import cv2

    det = YoloDetector(Path("models/yolo26n.pt"), confidence=0.4)
    dog = cv2.imread("tests/fixtures/dog.jpg")
    empty = cv2.imread("tests/fixtures/empty_room.jpg")
    assert any(d.label == "dog" for d in det.detect(dog))
    assert not any(d.label == "dog" for d in det.detect(empty))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_detector.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.detector'`.

- [ ] **Step 3: Implement `detector.py`**

```python
from __future__ import annotations

import platform
from pathlib import Path
from typing import Protocol

import numpy as np

from doggy.config import Settings
from doggy.detection import Detection


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class StubDetector:
    """Returns scripted detections; used by pipeline/web tests (no model)."""

    def __init__(self, scripted: list[list[Detection]]) -> None:
        self._scripted = list(scripted)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self._scripted:
            return self._scripted.pop(0)
        return []


def select_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if platform.machine() == "arm64" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    def __init__(self, model_path: Path, confidence: float, device: str | None = None) -> None:
        from ultralytics import YOLO

        self._model = YOLO(str(model_path))
        self._confidence = confidence
        self._device = device or select_device()

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results = self._model.predict(
            frame, conf=self._confidence, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label != "dog":
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return out


def build_detector(settings: Settings) -> Detector:
    return YoloDetector(settings.model_path, settings.confidence)
```

- [ ] **Step 4: Run non-slow tests to verify they pass**

Run: `uv run pytest tests/test_detector.py -v -m "not slow"`
Expected: PASS (2 passed, 1 deselected).

- [ ] **Step 5: Commit**

```bash
git add src/doggy/detector.py tests/test_detector.py
git commit -m "feat: YOLO26n detector wrapper with dog filter and device select"
```

Note: the `slow` test needs `models/yolo26n.pt` and fixture images. Download the model once with `uv run yolo export model=yolo26n.pt format=ncnn` (also produces the Pi NCNN dir) and add two fixture images. Run it manually with `uv run pytest -m slow` when hardware/weights are present; it is intentionally excluded from the default suite.

---

### Task 9: Pipeline wiring & entrypoint

**Files:**
- Create: `src/doggy/pipeline.py`, `src/doggy/main.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: all prior modules.
- Produces:
  - `annotate(frame, detections) -> np.ndarray` (draws boxes/labels).
  - `Pipeline(settings, detector, camera, alerter, runtime, status, raw_buffer, annotated_buffer, clock=time.monotonic, rng=random.Random())` with `run_once() -> bool` (process one available frame; returns whether it fired) and `run(stop: threading.Event)` (capture + detect threads).
  - `main() -> None` entrypoint (loads settings, builds real components, installs SIGINT/SIGTERM, optionally starts web).

- [ ] **Step 1: Write failing test (end-to-end with fakes, deterministic clock)**

`tests/test_pipeline.py`:
```python
import random
import threading

import numpy as np

from doggy.alerter import FakeAlerter
from doggy.camera import FakeCamera
from doggy.config import Settings
from doggy.detection import Detection
from doggy.detector import StubDetector
from doggy.pipeline import Pipeline
from doggy.safety import SafetyGovernor
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
from doggy.trigger import TriggerLogic


def test_pipeline_fires_after_confirmation(tmp_path):
    settings = Settings(confirm_seconds=1.0, window_m=1, window_n=1,
                        cooldown_min_seconds=5, cooldown_max_seconds=5)
    runtime = RuntimeSettings(settings.tunable())
    dog = [Detection("dog", 0.9, (0, 0, 10, 10))]
    detector = StubDetector([dog, dog, dog, dog])
    alerter = FakeAlerter()
    clock = iter([0.0, 0.5, 1.0, 1.5])
    pipe = Pipeline(
        settings=settings,
        detector=detector,
        camera=FakeCamera([np.zeros((16, 16, 3), np.uint8)], loop=True),
        alerter=alerter,
        runtime=runtime,
        status=StatusStore(),
        raw_buffer=FrameBuffer(),
        annotated_buffer=FrameBuffer(),
        safety=SafetyGovernor(runtime, tmp_path),
        clock=lambda: next(clock),
        rng=random.Random(0),
    )
    fired = [pipe.run_once() for _ in range(4)]
    assert any(fired)
    assert alerter.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.pipeline'`.

- [ ] **Step 3: Implement `pipeline.py`**

```python
from __future__ import annotations

import logging
import random
import threading
import time
from typing import Callable

import cv2
import numpy as np

from doggy.alerter import Alerter
from doggy.camera import Camera
from doggy.config import Settings
from doggy.detection import Detection
from doggy.detector import Detector
from doggy.safety import SafetyGovernor
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
from doggy.trigger import TriggerLogic

log = logging.getLogger("doggy")


def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d.box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


class Pipeline:
    def __init__(self, *, settings: Settings, detector: Detector, camera: Camera,
                 alerter: Alerter, runtime: RuntimeSettings, status: StatusStore,
                 raw_buffer: FrameBuffer, annotated_buffer: FrameBuffer,
                 safety: SafetyGovernor, clock: Callable[[], float] = time.monotonic,
                 rng: random.Random | None = None) -> None:
        self.settings = settings
        self.detector = detector
        self.camera = camera
        self.alerter = alerter
        self.runtime = runtime
        self.status = status
        self.raw_buffer = raw_buffer
        self.annotated_buffer = annotated_buffer
        self.safety = safety
        self.clock = clock
        self.trigger = TriggerLogic(runtime, rng=rng or random.Random())

    def run_once(self) -> bool:
        frame = self.raw_buffer.get()
        if frame is None:
            # In tests the capture thread isn't running; pull one frame directly.
            frame = next(self.camera.frames(), None)
            if frame is None:
                return False
        now = self.clock()
        detections = self.detector.detect(frame)
        self.annotated_buffer.set(annotate(frame, detections))
        top = max((d.confidence for d in detections), default=0.0)
        fired = self.trigger.update(detections, now)
        muted = not self.safety.allow_fire(now)
        if fired and not muted:
            self.alerter.alert()
            event = self.safety.record_fire(frame, top, now)
            self.status.add_event(event)
            self.status.update(last_fire_ts=event["ts"], last_fire_thumb=event["thumb"])
        self.status.update(state=self.trigger.state.value, confidence=round(top, 3),
                           fires_this_hour=self.safety.fires_last_hour(now), muted=muted)
        return fired and not muted

    def _capture_loop(self, stop: threading.Event) -> None:
        for frame in self.camera.frames():
            if stop.is_set():
                return
            self.raw_buffer.set(frame)

    def run(self, stop: threading.Event) -> None:
        cap = threading.Thread(target=self._capture_loop, args=(stop,), daemon=True)
        cap.start()
        last = self.clock()
        while not stop.is_set():
            if self.raw_buffer.get() is None:
                time.sleep(0.01)
                continue
            self.run_once()
            now = self.clock()
            dt = now - last
            if dt > 0:
                self.status.update(fps=round(1.0 / dt, 1))
            last = now
        self.camera.close()
```

- [ ] **Step 4: Implement `main.py`**

```python
from __future__ import annotations

import logging
import signal
import threading

from doggy.alerter import build_alerter
from doggy.camera import build_camera
from doggy.config import load_settings
from doggy.detector import build_detector
from doggy.pipeline import Pipeline
from doggy.safety import SafetyGovernor
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore


def main() -> None:
    settings = load_settings()
    logging.basicConfig(level=settings.log_level,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("doggy")

    runtime = RuntimeSettings(settings.tunable())
    status = StatusStore()
    raw_buffer = FrameBuffer()
    annotated_buffer = FrameBuffer()
    safety = SafetyGovernor(runtime, settings.event_log_dir)

    detector = build_detector(settings)   # loads model now (fail fast)
    camera = build_camera(settings)
    alerter = build_alerter(settings, runtime)

    pipeline = Pipeline(
        settings=settings, detector=detector, camera=camera, alerter=alerter,
        runtime=runtime, status=status, raw_buffer=raw_buffer,
        annotated_buffer=annotated_buffer, safety=safety,
    )

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    web_thread = None
    if settings.web_enabled:
        from doggy.web import serve
        web_thread = threading.Thread(
            target=serve,
            args=(settings, runtime, annotated_buffer, status, alerter),
            daemon=True,
        )
        web_thread.start()
        log.info("dashboard at http://%s:%s", settings.web_host, settings.web_port)

    log.info("doggy starting")
    pipeline.run(stop)
    log.info("doggy stopped")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the pipeline test to verify it passes**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS (1 passed). (Note: `main.py`/`web` import is lazy, so this test does not require Task 10.)

- [ ] **Step 6: Register the console entrypoint**

Add to `pyproject.toml` under `[project]`:
```toml
[project.scripts]
doggy = "doggy.main:main"
```

- [ ] **Step 7: Commit**

```bash
git add src/doggy/pipeline.py src/doggy/main.py tests/test_pipeline.py pyproject.toml
git commit -m "feat: pipeline wiring, capture/detect threads, entrypoint"
```

---

### Task 10: Web dashboard

**Files:**
- Create: `src/doggy/web.py`, `src/doggy/static/index.html`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `Settings`, `RuntimeSettings`, `FrameBuffer` (annotated), `StatusStore`, `Alerter`, `TunableSettings`.
- Produces:
  - `create_app(settings, runtime, annotated_buffer, status, alerter, save_env=...) -> FastAPI`.
  - `serve(settings, runtime, annotated_buffer, status, alerter) -> None` (runs uvicorn).
  - Endpoints: `GET /`, `GET /stream.mjpg`, `GET /api/status`, `PATCH /api/settings`, `POST /api/test-sound`, `POST /api/settings/save`.

- [ ] **Step 1: Write failing tests**

`tests/test_web.py`:
```python
from fastapi.testclient import TestClient

from doggy.alerter import FakeAlerter
from doggy.config import Settings
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
from doggy.web import create_app


def client(saved=None):
    settings = Settings()
    runtime = RuntimeSettings(settings.tunable())
    alerter = FakeAlerter()
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), alerter,
                     save_env=lambda t: saved.update(t.model_dump()) if saved is not None else None)
    return TestClient(app), runtime, alerter


def test_status_returns_settings_and_state():
    c, _, _ = client()
    r = c.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "IDLE"
    assert body["settings"]["confidence"] == 0.55


def test_patch_updates_runtime():
    c, runtime, _ = client()
    r = c.patch("/api/settings", json={"confidence": 0.8})
    assert r.status_code == 200
    assert runtime.get().confidence == 0.8
    assert c.get("/api/status").json()["settings"]["confidence"] == 0.8


def test_patch_rejects_invalid():
    c, _, _ = client()
    r = c.patch("/api/settings", json={"window_m": 9, "window_n": 3})
    assert r.status_code == 422


def test_test_sound_triggers_alerter():
    c, _, alerter = client()
    assert c.post("/api/test-sound").status_code == 200
    assert alerter.calls == 1


def test_save_persists(tmp_path):
    saved = {}
    c, _, _ = client(saved=saved)
    c.patch("/api/settings", json={"confidence": 0.65})
    assert c.post("/api/settings/save").status_code == 200
    assert saved["confidence"] == 0.65
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_web.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'doggy.web'`.

- [ ] **Step 3: Implement `web.py`**

```python
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse

from doggy.alerter import Alerter
from doggy.config import Settings, TunableSettings
from doggy.state import FrameBuffer, RuntimeSettings, StatusStore

_STATIC = Path(__file__).parent / "static"


def _write_env(tunable: TunableSettings, path: Path = Path(".env")) -> None:
    lines = [f"DOGGY_{k.upper()}={v}" for k, v in tunable.model_dump().items()]
    path.write_text("\n".join(lines) + "\n")


def create_app(settings: Settings, runtime: RuntimeSettings,
               annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter,
               save_env: Callable[[TunableSettings], None] = _write_env) -> FastAPI:
    app = FastAPI(title="doggy")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(_STATIC / "index.html")

    @app.get("/api/status")
    def api_status() -> dict:
        snap = status.snapshot()
        return {
            "state": snap.state, "fps": snap.fps, "confidence": snap.confidence,
            "fires_this_hour": snap.fires_this_hour, "muted": snap.muted,
            "last_fire_ts": snap.last_fire_ts, "last_fire_thumb": snap.last_fire_thumb,
            "settings": runtime.get().model_dump(mode="json"),
            "events": status.events(),
        }

    @app.patch("/api/settings")
    def api_patch(patch: dict) -> dict:
        merged = {**runtime.get().model_dump(), **patch}
        updated = TunableSettings(**merged)  # raises 422 on invalid
        runtime.update(updated)
        return updated.model_dump(mode="json")

    @app.post("/api/test-sound")
    def api_test_sound() -> dict:
        alerter.alert()
        return {"ok": True}

    @app.post("/api/settings/save")
    def api_save() -> dict:
        save_env(runtime.get())
        return {"ok": True}

    @app.get("/stream.mjpg")
    def stream() -> StreamingResponse:
        def gen():
            while True:
                frame = annotated_buffer.get()
                if frame is not None:
                    ok, buf = cv2.imencode(".jpg", frame)
                    if ok:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                               + buf.tobytes() + b"\r\n")
                time.sleep(0.1)  # throttle to ~10 FPS so it never starves detection

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def serve(settings: Settings, runtime: RuntimeSettings,
          annotated_buffer: FrameBuffer, status: StatusStore, alerter: Alerter) -> None:
    import uvicorn

    app = create_app(settings, runtime, annotated_buffer, status, alerter)
    uvicorn.run(app, host=settings.web_host, port=settings.web_port, log_level="warning")
```

Note: FastAPI raises `RequestValidationError` (HTTP 422) when `TunableSettings(**merged)` fails, satisfying `test_patch_rejects_invalid`.

- [ ] **Step 4: Implement `static/index.html`**

```html
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>doggy</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #111; color: #eee; }
  h1 { font-size: 1.3rem; } img { max-width: 100%; border: 2px solid #333; border-radius: 8px; }
  .row { display: flex; gap: 2rem; flex-wrap: wrap; }
  .panel { flex: 1; min-width: 280px; }
  label { display: block; margin: .6rem 0 .2rem; font-size: .85rem; color: #aaa; }
  input[type=range] { width: 100%; }
  .stat { font-size: 1.5rem; } .muted { color: #f66; }
  button { margin-top: 1rem; padding: .5rem 1rem; border: 0; border-radius: 6px;
           background: #2a7; color: #041; font-weight: 600; cursor: pointer; }
  #events { font-size: .8rem; color: #999; max-height: 8rem; overflow: auto; }
</style>
</head>
<body>
<h1>🐕 doggy dashboard</h1>
<div class="row">
  <div class="panel"><img src="/stream.mjpg" alt="live" /></div>
  <div class="panel">
    <div class="stat">State: <span id="state">–</span></div>
    <div>FPS: <span id="fps">–</span> · conf: <span id="conf">–</span>
         · fires/hr: <span id="fires">–</span> <span id="muted"></span></div>
    <label>confidence <span id="confidence_v"></span></label>
    <input id="confidence" type="range" min="0" max="1" step="0.01" />
    <label>confirm_seconds <span id="confirm_seconds_v"></span></label>
    <input id="confirm_seconds" type="range" min="0" max="5" step="0.1" />
    <label>cooldown_min_seconds <span id="cooldown_min_seconds_v"></span></label>
    <input id="cooldown_min_seconds" type="range" min="0" max="60" step="1" />
    <label>cooldown_max_seconds <span id="cooldown_max_seconds_v"></span></label>
    <input id="cooldown_max_seconds" type="range" min="0" max="120" step="1" />
    <label>max_fires_per_hour <span id="max_fires_per_hour_v"></span></label>
    <input id="max_fires_per_hour" type="range" min="0" max="60" step="1" />
    <label><input id="safety_enabled" type="checkbox" /> safety enabled</label>
    <div>
      <button onclick="testSound()">Test sound</button>
      <button onclick="save()">Save to .env</button>
    </div>
    <h3>Recent fires</h3>
    <div id="events"></div>
  </div>
</div>
<script>
const KNOBS = ["confidence","confirm_seconds","cooldown_min_seconds",
               "cooldown_max_seconds","max_fires_per_hour"];
async function patch(body){
  await fetch("/api/settings",{method:"PATCH",headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)});
}
KNOBS.forEach(k => {
  const el = document.getElementById(k);
  el.addEventListener("change", () => patch({[k]: parseFloat(el.value)}));
});
document.getElementById("safety_enabled").addEventListener("change", e =>
  patch({safety_enabled: e.target.checked}));
async function testSound(){ await fetch("/api/test-sound",{method:"POST"}); }
async function save(){ await fetch("/api/settings/save",{method:"POST"}); }
async function poll(){
  const s = await (await fetch("/api/status")).json();
  document.getElementById("state").textContent = s.state;
  document.getElementById("fps").textContent = s.fps;
  document.getElementById("conf").textContent = s.confidence;
  document.getElementById("fires").textContent = s.fires_this_hour;
  document.getElementById("muted").innerHTML = s.muted ? '<span class="muted">MUTED</span>' : '';
  for (const k of KNOBS){
    const el = document.getElementById(k);
    if (document.activeElement !== el){ el.value = s.settings[k]; }
    document.getElementById(k+"_v").textContent = s.settings[k];
  }
  document.getElementById("safety_enabled").checked = s.settings.safety_enabled;
  document.getElementById("events").innerHTML =
    s.events.slice(-10).reverse().map(e => `${e.ts.toFixed(1)}s · conf ${e.confidence}`).join("<br>");
}
setInterval(poll, 500); poll();
</script>
</body>
</html>
```

- [ ] **Step 5: Ensure static files ship in the package**

Add to `pyproject.toml`:
```toml
[tool.hatch.build.targets.wheel.force-include]
"src/doggy/static" = "doggy/static"
```

- [ ] **Step 6: Run web tests to verify they pass**

Run: `uv run pytest tests/test_web.py -v`
Expected: PASS (5 passed).

- [ ] **Step 7: Commit**

```bash
git add src/doggy/web.py src/doggy/static/index.html tests/test_web.py pyproject.toml
git commit -m "feat: localhost dashboard (MJPEG, live knobs, test sound, save)"
```

---

### Task 11: Deployment glue, docs & full-suite green

**Files:**
- Create: `README.md`, `systemd/doggy.service`, `sounds/README.md`
- Test: run the whole suite.

**Interfaces:**
- Consumes: everything.
- Produces: run instructions, a systemd unit, a place for clips.

- [ ] **Step 1: Create `systemd/doggy.service`**

```ini
[Unit]
Description=Doggy detector
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/pi/doggy
EnvironmentFile=/home/pi/doggy/.env
ExecStart=/home/pi/.local/bin/uv run doggy
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Create `sounds/README.md`**

```markdown
Drop deterrent clips here (.wav/.flac/.ogg/.mp3). One is chosen at random per fire
(anti-habituation). Point elsewhere with DOGGY_CLIPS_DIR.
```

- [ ] **Step 3: Create `README.md`**

```markdown
# 🐕 doggy

Local dog-presence detector: watches a USB webcam, plays a deterrent clip when a
dog is confirmed, and serves a localhost dashboard for live view + live tuning.

## Quick start (Mac dev)

    uv sync
    cp .env.example .env          # set DOGGY_CAMERA_INDEX (C922 is usually 1)
    uv run yolo export model=yolo26n.pt format=ncnn   # downloads yolo26n.pt
    # add at least one clip to sounds/
    uv run doggy                  # dashboard at http://127.0.0.1:8000

Grant your terminal camera permission (System Settings → Privacy → Camera) or
OpenCV returns empty frames silently.

## Config

Everything is set via `DOGGY_*` env vars (see `.env.example`). Live-tunable
params (confidence, confirm/cooldown, volume, safety) can also be changed from the
dashboard; structural params (camera, model, audio backend) need a restart.

## Raspberry Pi 5

- Use a USB webcam (the CSI ribbon camera is not supported in v1).
- Set a USB speaker as the default audio sink (Pi 5 has no 3.5mm jack).
- Export the model to NCNN for speed; install and run with `uv`.
- Run as a service: copy `systemd/doggy.service`, `systemctl enable --now doggy`.

## Tests

    uv run pytest -m "not slow"    # fast suite, no hardware/weights
    uv run pytest -m slow          # detector test (needs model + fixtures)

## License

AGPL-3.0-or-later (YOLO26n is AGPL; the whole project matches).
```

- [ ] **Step 4: Run the full fast suite**

Run: `uv run pytest -m "not slow" -v`
Expected: PASS (all tests from Tasks 1–10 green).

- [ ] **Step 5: Manual end-to-end smoke (optional, needs webcam + a clip)**

Run: `uv run doggy`, open `http://127.0.0.1:8000`, confirm the live video shows, move the confidence slider and watch `/api/status` reflect it, click **Test sound**. Ctrl-C exits cleanly (camera released).

- [ ] **Step 6: Commit**

```bash
git add README.md systemd/doggy.service sounds/README.md
git commit -m "docs: readme, systemd unit, sounds folder"
```

---

## Self-Review

**Spec coverage** (spec §→task):
- §2 scope (any-dog presence, sound clip, web UI): Tasks 4, 7, 10 ✓
- §3 model YOLO26n + swappable interface: Task 8 (`Detector` protocol) ✓
- §4 architecture (capture/detect threads, Detection dataclass, shared state): Tasks 3, 9 ✓; alert is fire-and-forget inside the alerter (Task 7) rather than a dedicated loop thread — noted deviation, satisfies "non-blocking playback".
- §5 safety envelope (rate limit, off switch, event log + thumbnail, volume cap): Task 5 + volume in Task 7 ✓
- §6 env config (pydantic-settings, DOGGY_ prefix, live-tunable vs restart, save-to-.env): Tasks 2, 10 ✓
- §7 web UI (endpoints, MJPEG throttled, static html, localhost): Task 10 ✓
- §8 portability (USB webcam, sounddevice, device auto-select, macOS perm note): Tasks 6, 7, 8, README ✓
- §9 tooling (uv, pyproject, pydantic-settings, fastapi/uvicorn): Task 1 ✓
- §10 testing (pure units, detector integration slow, e2e via FakeCamera, web TestClient): Tasks 2–10 ✓
- §11 deployment (systemd, Pi notes): Task 11 ✓
- §3 license relicense to AGPL: Task 1 ✓

**Placeholder scan:** No TODO/TBD; every code step has complete code. The AGPL text and fixture images/model weights are external assets fetched by explicit commands, not code placeholders.

**Type consistency:** `Detection(label, confidence, box)`, `RuntimeSettings.get()/update()`, `TunableSettings`, `Detector.detect()`, `Alerter.alert()`, `TriggerLogic.update(detections, now)`, `SafetyGovernor.allow_fire(now)/record_fire(frame, confidence, now)`, `FrameBuffer.get()/set()`, `StatusStore.update()/snapshot()/events()/add_event()` are used consistently across tasks 3–10. `Pipeline` constructor keyword args match `main.py` and `test_pipeline.py`.
