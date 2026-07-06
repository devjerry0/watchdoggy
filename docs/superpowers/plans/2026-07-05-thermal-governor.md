# Thermal Governor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The detector auto-scales its inference interval by CPU temperature so a bare-board Pi 4 hovers at a safe temp instead of throttling, without ever stopping detection.

**Architecture:** A new stateless `ThermalGovernor` reads CPU temp from sysfs and maps it (proportional ramp) to an effective detect interval; the pipeline `run` loop paces with that instead of the fixed interval and publishes temp + effective interval to the status store; the dashboard shows a 🌡️ readout + a COOLING badge.

**Tech Stack:** Python 3.11, pydantic/pydantic-settings, NumPy/OpenCV (existing), FastAPI, vanilla-JS frontend, pytest.

## Global Constraints

- All params env-configurable via `TunableSettings` (prefix `DOGGY_`).
- Governor is **stateless** and **inert** when temp is unavailable (non-Pi) or `thermal_enabled=False` → returns the normal `detect_interval_seconds`.
- Control law: `t≤target`→normal; `t≥max`→cooldown; between→linear ramp; and NEVER faster than normal → `max(detect_interval_seconds, ramped)`.
- Defaults: `thermal_target_c=74`, `thermal_max_c=82`, `thermal_cooldown_interval_seconds=1.5`.
- `run_once` still calls `self.clock()` exactly once (unchanged — the governor lives in `run`, not `run_once`).
- Tests live flat in `tests/`; run `uv run pytest`.

---

### Task 1: Config fields + validator

**Files:**
- Modify: `src/doggy/config.py` (`TunableSettings`)
- Test: `tests/test_config.py`

**Interfaces produced:** `TunableSettings.thermal_enabled: bool`, `thermal_target_c: float`, `thermal_max_c: float`, `thermal_cooldown_interval_seconds: float`.

- [ ] **Step 1: Failing tests** — add to `tests/test_config.py`:
```python
def test_thermal_defaults():
    from doggy.config import Settings
    s = Settings()
    assert s.thermal_enabled is True
    assert s.thermal_target_c == 74.0
    assert s.thermal_max_c == 82.0
    assert s.thermal_cooldown_interval_seconds == 1.5

def test_thermal_target_must_be_le_max():
    import pytest
    from pydantic import ValidationError
    from doggy.config import TunableSettings
    with pytest.raises(ValidationError):
        TunableSettings(thermal_target_c=90.0, thermal_max_c=80.0)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_config.py::test_thermal_defaults tests/test_config.py::test_thermal_target_must_be_le_max -q` → FAIL.

- [ ] **Step 3: Add fields + validator.** In `src/doggy/config.py`, inside `TunableSettings` (after `detect_interval_seconds`):
```python
    thermal_enabled: bool = True
    thermal_target_c: float = Field(74.0, ge=0.0)
    thermal_max_c: float = Field(82.0, ge=0.0)
    thermal_cooldown_interval_seconds: float = Field(1.5, ge=0.0)
```
And extend the existing `_check_ranges` `model_validator` — add before its `return self`:
```python
        if self.thermal_target_c > self.thermal_max_c:
            raise ValueError("thermal_target_c must be <= thermal_max_c")
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_config.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/doggy/config.py tests/test_config.py
git commit -m "feat: thermal governor config fields + target<=max validator"
```

---

### Task 2: `ThermalGovernor`

**Files:**
- Create: `src/doggy/thermal.py`
- Test: `tests/test_thermal.py`

**Interfaces produced:** `ThermalGovernor(temp_path: str = <sysfs>)`; `.read_temp_c() -> float | None`; `.effective_interval(temp_c: float | None, cfg: TunableSettings) -> float`.

- [ ] **Step 1: Failing tests** — `tests/test_thermal.py`:
```python
from doggy.config import TunableSettings
from doggy.thermal import ThermalGovernor

CFG = TunableSettings(detect_interval_seconds=0.5, thermal_enabled=True,
                      thermal_target_c=74.0, thermal_max_c=82.0,
                      thermal_cooldown_interval_seconds=1.5)

def test_read_temp_c_parses_millidegrees(tmp_path):
    f = tmp_path / "temp"; f.write_text("78123\n")
    assert ThermalGovernor(str(f)).read_temp_c() == 78.123

def test_read_temp_c_missing_file_returns_none(tmp_path):
    assert ThermalGovernor(str(tmp_path / "nope")).read_temp_c() is None

def test_interval_none_temp_is_normal():
    assert ThermalGovernor().effective_interval(None, CFG) == 0.5

def test_interval_below_target_is_normal():
    assert ThermalGovernor().effective_interval(70.0, CFG) == 0.5

def test_interval_at_or_above_max_is_cooldown():
    assert ThermalGovernor().effective_interval(82.0, CFG) == 1.5
    assert ThermalGovernor().effective_interval(90.0, CFG) == 1.5

def test_interval_ramps_linearly_between():
    # midpoint 78 of [74,82] -> halfway between 0.5 and 1.5 = 1.0
    assert ThermalGovernor().effective_interval(78.0, CFG) == 1.0

def test_interval_disabled_is_normal():
    cfg = CFG.model_copy(update={"thermal_enabled": False})
    assert ThermalGovernor().effective_interval(90.0, cfg) == 0.5

def test_interval_never_faster_than_normal():
    # cooldown accidentally set below normal -> guard keeps it at normal
    cfg = CFG.model_copy(update={"detect_interval_seconds": 2.0,
                                 "thermal_cooldown_interval_seconds": 1.5})
    assert ThermalGovernor().effective_interval(90.0, cfg) == 2.0
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_thermal.py -q` → FAIL (`No module named doggy.thermal`).

- [ ] **Step 3: Implement** — `src/doggy/thermal.py`:
```python
from __future__ import annotations

from doggy.config import TunableSettings

_SYSFS_TEMP = "/sys/class/thermal/thermal_zone0/temp"


class ThermalGovernor:
    """Map CPU temperature to a detect interval (stateless, proportional).

    Cool → normal interval; hotter → longer interval (less load) up to a cap;
    unreadable temp or disabled → normal interval (inert). Never returns an
    interval faster than the configured normal one.
    """

    def __init__(self, temp_path: str = _SYSFS_TEMP) -> None:
        self._temp_path = temp_path

    def read_temp_c(self) -> float | None:
        try:
            with open(self._temp_path) as fh:
                return int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def effective_interval(self, temp_c: float | None, cfg: TunableSettings) -> float:
        normal = cfg.detect_interval_seconds
        if temp_c is None or not cfg.thermal_enabled:
            return normal
        if temp_c <= cfg.thermal_target_c:
            return normal
        if temp_c >= cfg.thermal_max_c:
            ramped = cfg.thermal_cooldown_interval_seconds
        else:
            span = cfg.thermal_max_c - cfg.thermal_target_c
            frac = (temp_c - cfg.thermal_target_c) / span if span > 0 else 1.0
            ramped = normal + frac * (cfg.thermal_cooldown_interval_seconds - normal)
        return max(normal, ramped)
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_thermal.py -q` → PASS.

- [ ] **Step 5: Commit**
```bash
git add src/doggy/thermal.py tests/test_thermal.py
git commit -m "feat: ThermalGovernor — proportional temp->interval control"
```

---

### Task 3: Status fields + pipeline wiring

**Files:**
- Modify: `src/doggy/state.py` (`Status`)
- Modify: `src/doggy/pipeline.py` (`Pipeline.__init__`, `run`)
- Test: `tests/test_state.py`, `tests/test_thermal.py` (integration)

**Interfaces:** consumes `ThermalGovernor` (Task 2) + config fields (Task 1). `Status` gains `temp_c: float | None = None`, `detect_interval_effective: float = 0.0`.

- [ ] **Step 1: Failing tests** — add to `tests/test_state.py`:
```python
def test_status_has_thermal_fields():
    from doggy.state import Status, StatusStore
    assert Status().temp_c is None
    assert Status().detect_interval_effective == 0.0
    s = StatusStore(); s.update(temp_c=76.5, detect_interval_effective=1.0)
    assert s.snapshot().temp_c == 76.5
    assert s.snapshot().detect_interval_effective == 1.0
```
And add to `tests/test_thermal.py` an integration check that the pipeline paces via the governor (construct a governor with a fake hot temp file, assert the chosen interval):
```python
def test_governor_picks_cooldown_when_hot(tmp_path):
    from doggy.config import TunableSettings
    from doggy.thermal import ThermalGovernor
    f = tmp_path / "temp"; f.write_text("83000\n")
    g = ThermalGovernor(str(f))
    cfg = TunableSettings(detect_interval_seconds=0.5, thermal_cooldown_interval_seconds=1.5)
    assert g.effective_interval(g.read_temp_c(), cfg) == 1.5
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_state.py::test_status_has_thermal_fields tests/test_thermal.py::test_governor_picks_cooldown_when_hot -q` → FAIL.

- [ ] **Step 3: Add Status fields.** In `src/doggy/state.py`, in the `Status` dataclass (after `muted: bool = False`):
```python
    temp_c: float | None = None
    detect_interval_effective: float = 0.0
```

- [ ] **Step 4: Wire the governor into the pipeline.** In `src/doggy/pipeline.py`:
  - Add import: `from doggy.thermal import ThermalGovernor`
  - In `Pipeline.__init__`, after `self.pacer = Pacer(clock=clock)`:
    ```python
        self.governor = ThermalGovernor()
    ```
  - In `run`, REPLACE the existing pacing line
    `self.pacer.wait(self.runtime.get().detect_interval_seconds)`
    with:
    ```python
            cfg = self.runtime.get()
            temp = self.governor.read_temp_c()
            interval = self.governor.effective_interval(temp, cfg)
            self.status.update(temp_c=temp, detect_interval_effective=interval)
            self.pacer.wait(interval)
    ```

- [ ] **Step 5: Run tests** — `uv run pytest -q` → PASS (except the known pre-existing `test_yolo_detects_dog_and_ignores_empty_room` fixture failure).

- [ ] **Step 6: Commit**
```bash
git add src/doggy/state.py src/doggy/pipeline.py tests/test_state.py tests/test_thermal.py
git commit -m "feat: pipeline paces via ThermalGovernor; publish temp_c + effective interval"
```

---

### Task 4: Dashboard temp readout + COOLING badge

**Files:**
- Modify: `src/doggy/static/index.html`
- Test: `tests/test_web.py` (smoke)

**Interfaces:** consumes `GET /api/status` fields `temp_c`, `detect_interval_effective`, and `settings.detect_interval_seconds`.

- [ ] **Step 1: Failing smoke test** — add to `tests/test_web.py`:
```python
def test_index_has_temp_readout():
    from fastapi.testclient import TestClient
    from doggy.web import create_app
    from doggy.config import Settings
    from doggy.state import FrameBuffer, RuntimeSettings, StatusStore
    from doggy.alerter import FakeAlerter
    s = Settings()
    app = create_app(s, RuntimeSettings(s.tunable()), FrameBuffer(), StatusStore(), FakeAlerter())
    html = TestClient(app).get("/").text
    assert 'id="temp"' in html
    assert "COOLING" in html
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_web.py::test_index_has_temp_readout -q` → FAIL.

- [ ] **Step 3: Implement.** In `src/doggy/static/index.html`:
  (a) In the status line div (the one showing FPS/dogs/conf/fires), append a temp span + cooling badge — change:
  ```html
       · fires/hr: <span id="fires">–</span> <span id="muted"></span></div>
  ```
  to:
  ```html
       · fires/hr: <span id="fires">–</span>
       · 🌡️ <span id="temp">–</span>°C <span id="cooling"></span> <span id="muted"></span></div>
  ```
  (b) In `poll()`, after the `muted` line, add:
  ```javascript
    document.getElementById("temp").textContent = s.temp_c == null ? "–" : s.temp_c.toFixed(1);
    const cooling = s.detect_interval_effective > s.settings.detect_interval_seconds;
    document.getElementById("cooling").innerHTML = cooling ? '<span style="color:#fa0">COOLING</span>' : '';
  ```

- [ ] **Step 4: Run tests** — `uv run pytest tests/test_web.py -q && uv run pytest -q` → PASS (except the known pre-existing fixture failure).

- [ ] **Step 5: Commit**
```bash
git add src/doggy/static/index.html tests/test_web.py
git commit -m "feat: dashboard shows CPU temp + COOLING badge when the governor backs off"
```

---

## Self-Review

**Spec coverage:** config fields + validator (Task 1) ✓; `ThermalGovernor.read_temp_c`/`effective_interval` control law incl. `max(normal,…)` guard + inert cases (Task 2) ✓; Status fields + pipeline `run`-loop pacing via governor (Task 3) ✓; dashboard temp + COOLING (Task 4) ✓. Tests cover parse/missing, all control-law branches, disabled, guard, status fields, hot-picks-cooldown, page controls.

**Placeholders:** none — full code in every step.

**Type consistency:** `ThermalGovernor(temp_path=...)`, `.read_temp_c() -> float|None`, `.effective_interval(temp_c, cfg) -> float` used identically in Tasks 2–3; `Status.temp_c: float|None`, `detect_interval_effective: float` match between Task 3 definition and Task 4 consumption; config field names match spec and are read via `cfg.<name>` in the governor.
