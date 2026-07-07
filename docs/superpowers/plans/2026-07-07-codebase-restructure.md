# Codebase Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize flat `src/doggy/` into domain packages with seven deliberate patterns (Observer+Decorator, Chain-of-Responsibility with Adapter links, Strategy registries, State FSM, Template Method, Facade, composition root) — zero behavior change.

**Architecture:** Move modules with `git mv` (history preserved), one domain per task, updating every importer in the same task so the full suite is green at every task boundary. New pattern code (filters, analyzer, hub, gate, recorder, state classes, alerter template) is introduced alongside the moves it belongs with; `pipeline.py` shrinks task by task and is finalized as a thin Facade near the end.

**Tech Stack:** unchanged — Python 3.11, FastAPI, pydantic-settings, OpenCV, uv/pytest/ruff. No new dependencies.

## Global Constraints

- **Zero behavior change.** All ~132 existing tests keep passing with assertions unweakened (moves/renames of test files allowed; expected totals per task are stated below and grow only from NEW seam tests).
- **Frozen external contract:** every `DOGGY_*` env var name; every HTTP route path/method/status/JSON shape; `.env` round-trip format; `events.jsonl` schema + `events/` layout; `uv run doggy` console script; deploy scripts; systemd drop-ins.
- **`src/doggy/main.py` must survive as a shim** (`from doggy.app import main` + `__main__` guard). The Pi's installed entry-point metadata resolves `doggy.main:main` and the service runs `uv run --no-sync`, so removing it bricks the appliance. `pyproject.toml` `[project.scripts]` stays `doggy = "doggy.main:main"` — do not edit pyproject.
- **All file moves via `git mv`** (then edit), never delete+recreate.
- Per task: `uv run pytest -m "not slow"` green AND `uv run ruff check` clean on every touched file, then commit.
- After each task, the old flat module(s) it covers must be gone: `grep -rn "from doggy.<old> import\|doggy\.<old>\." src/ tests/` returns nothing (checked per task below).
- No new dependencies; no async; no pyproject changes.

## File Structure

Final tree is specified in the spec (`docs/superpowers/specs/2026-07-07-codebase-restructure-design.md` — read it once before Task 1). Every task below names its exact moves.

---

### Task 1: `core/` package (config, runtime, status, pacer)

**Files:**
- `git mv src/doggy/config.py src/doggy/core/config.py`
- Split `src/doggy/state.py` → Create `src/doggy/core/runtime.py` (RuntimeSettings) and `src/doggy/core/status.py` (CONFIDENCE_DECIMALS, FrameBuffer, Status, StatusStore); delete `state.py` (use `git mv state.py core/status.py`, then extract RuntimeSettings into runtime.py)
- `git mv src/doggy/pacer.py src/doggy/core/pacer.py`; create `src/doggy/core/__init__.py` (empty)
- Tests: `git mv tests/test_config.py tests/core/test_config.py`; `git mv tests/test_state.py tests/core/test_status.py`; `git mv tests/test_pacer.py tests/core/test_pacer.py`; create `tests/core/__init__.py`

**Interfaces (produced, used by all later tasks):** `from doggy.core.config import Settings, TunableSettings, load_settings` · `from doggy.core.runtime import RuntimeSettings` · `from doggy.core.status import CONFIDENCE_DECIMALS, FrameBuffer, Status, StatusStore` · `from doggy.core.pacer import Pacer`

- [ ] **Step 1:** Perform the moves above. Class/function bodies unchanged (pure relocation; `runtime.py` needs only `threading` + `TunableSettings` imports; `status.py` keeps the rest of state.py's content minus RuntimeSettings).
- [ ] **Step 2:** Update every importer repo-wide. Find them: `grep -rln "doggy.config\|doggy.state\|doggy.pacer" src/ tests/`. Rewrite: `doggy.config → doggy.core.config`; `doggy.state import RuntimeSettings → doggy.core.runtime import RuntimeSettings`; other `doggy.state` names → `doggy.core.status`; `doggy.pacer → doggy.core.pacer`.
- [ ] **Step 3:** Verify: `uv run pytest -m "not slow"` → **132 passed**; `grep -rn "doggy\.state\|doggy\.pacer\b" src/ tests/` → empty; `grep -rn "from doggy.config" src/ tests/` → empty; ruff clean on touched files.
- [ ] **Step 4:** Commit: `refactor: extract core/ package (config, runtime, status, pacer)`

---

### Task 2: `events/` + `hardware/` packages

**Files:**
- `git mv src/doggy/events.py src/doggy/events/store.py` (create `events/__init__.py`)
- `git mv src/doggy/thermal.py src/doggy/hardware/thermal.py`; `git mv src/doggy/power.py src/doggy/hardware/power.py` (create `hardware/__init__.py`)
- Tests: `git mv tests/test_events.py tests/events/test_store.py`; `git mv tests/test_thermal.py tests/hardware/test_thermal.py`; `git mv tests/test_power.py tests/hardware/test_power.py` (+ `__init__.py` files)

**Interfaces:** `from doggy.events.store import EventStore, EventRecord, EVENTS_FILE` · `from doggy.hardware.thermal import ThermalGovernor` · `from doggy.hardware.power import PowerMonitor, PowerStatus`

- [ ] **Step 1:** Moves above; update importers (`grep -rln "doggy.events import\|doggy.thermal\|doggy.power" src/ tests/` → safety.py, web.py, main.py, pipeline.py, tests). Note `doggy.events` becomes a package: the old `from doggy.events import EventStore` becomes `from doggy.events.store import EventStore`.
- [ ] **Step 2:** Verify: suite → **132 passed**; `grep -rn "doggy\.thermal\|doggy\.power\b" src/ tests/` → empty; ruff clean.
- [ ] **Step 3:** Commit: `refactor: extract events/ and hardware/ packages`

---

### Task 3: `vision/` package (detection, camera, detector, annotate) + Strategy registries

**Files:**
- `git mv` detection.py, camera.py, detector.py → `src/doggy/vision/` (+ `__init__.py`)
- Create `src/doggy/vision/annotate.py`: move `annotate()`, `_draw_box()`, and the color/drawing constants (`_BOX_THICKNESS`, `_LABEL_*`, `_DOG_ACTIVE_COLOR`, `_DOG_IGNORED_COLOR`, `_PERSON_COLOR`, `_ZONE_COLOR`, `_ZONE_ALPHA`) out of `pipeline.py`, unchanged. `pipeline.py` imports `annotate` from here.
- In `vision/camera.py`: replace the `if/elif` in `build_camera` with a registry: `_BACKENDS: dict[str, Callable[[Settings], Camera]] = {"opencv": ..., "file": ...}`; unknown backend raises `ValueError` with the same message style as today. Same construction behavior.
- In `vision/detector.py`: keep `build_detector(settings, runtime)` (single strategy + StubDetector for tests) — relocation only.
- Tests: `git mv tests/test_camera.py tests/vision/test_camera.py`; `git mv tests/test_detector.py tests/vision/test_detector.py` (+ `__init__.py`)

**Interfaces:** `from doggy.vision.detection import Detection, TARGET_LABEL, PERSON_LABEL` · `from doggy.vision.camera import Camera, FakeCamera, build_camera` · `from doggy.vision.detector import Detector, StubDetector, YoloDetector, build_detector` · `from doggy.vision.annotate import annotate`

- [ ] **Step 1:** Moves + registry edit + annotate extraction; update importers (people.py, zone.py, trigger.py, pipeline.py, main.py, tests).
- [ ] **Step 2:** Verify: suite → **132 passed**; `grep -rn "from doggy.detection\|from doggy.camera\|from doggy.detector" src/ tests/` → empty; `grep -n "annotate\|_draw_box\|_ZONE_COLOR" src/doggy/pipeline.py` shows only the import + call sites; ruff clean.
- [ ] **Step 3:** Commit: `refactor: extract vision/ package with strategy registry for camera backends`

---

### Task 4: `vision/filters/` (Adapter links) + `vision/analysis.py`; pipeline adopts the analyzer

**Files:**
- Create `src/doggy/vision/analysis.py`, `src/doggy/vision/filters/{__init__,base,person,zone}.py`
- Absorb and delete `src/doggy/people.py` (logic → `filters/person.py`; `iou` + `suppress_dogs_overlapping_people` move verbatim) and `src/doggy/zone.py` (mask-cache logic → `filters/zone.py`)
- Modify `src/doggy/pipeline.py`: `run_once` detection section becomes `analysis = self.analyzer.analyze(frame, cfg)` + annotate call
- Tests: create `tests/vision/filters/test_person.py` and `test_zone.py` by `git mv` from `tests/test_people.py` / `tests/test_zone.py`, retargeting the same behavioral assertions at the filter classes; create `tests/vision/test_analysis.py`

**Interfaces (produced):**

```python
# vision/analysis.py
@dataclass
class FrameAnalysis:
    shape: tuple[int, ...]
    people: list[Detection]
    dogs: list[Detection]          # post-suppression (drawn)
    candidates: list[Detection]    # post-zone (may trigger)

class DetectionAnalyzer:
    def __init__(self, detector: Detector, chain: FilterChain) -> None: ...
    def analyze(self, frame: np.ndarray, cfg: TunableSettings) -> FrameAnalysis:
        # detect -> split by label -> seed analysis(dogs=candidates=dog-labeled,
        # people=person-labeled) -> chain.run(analysis, cfg) -> return

# vision/filters/base.py
class DetectionFilter(Protocol):
    def apply(self, analysis: FrameAnalysis, cfg: TunableSettings) -> None: ...

class FilterChain:
    def __init__(self, filters: Sequence[DetectionFilter]) -> None: ...
    def run(self, analysis: FrameAnalysis, cfg: TunableSettings) -> None: ...  # in order
```

- `filters/person.py` `PersonSuppressionFilter.apply`: no-op unless `cfg.person_suppression_enabled` and `analysis.people`; else `analysis.dogs = suppress_dogs_overlapping_people(analysis.dogs, analysis.people, cfg.person_iou_threshold)`; `analysis.candidates = list(analysis.dogs)`.
- `filters/zone.py` `ZoneInclusionFilter.apply`: no-op unless `cfg.zone_enabled` and `len(cfg.zone_points) >= 3`; else narrow `analysis.candidates` with the existing cached-mask test (cache keyed `(tuple(points), shape[:2])` exactly as today).

**Behavior locks (pipeline after adoption):** annotate receives `analysis.dogs`, `in_zone=analysis.candidates`, `zone_points=cfg.zone_points if cfg.zone_enabled else []`, `people=analysis.people if cfg.person_suppression_enabled else None`; status `dogs=len(analysis.candidates)`, `people=len(analysis.people) if cfg.person_suppression_enabled else 0`; trigger sees `analysis.candidates`; `top = max(candidates confidence, default 0.0)`.

- [ ] **Step 1 (failing tests first):** `tests/vision/test_analysis.py`:

```python
def _cfg(**over):
    from doggy.core.config import TunableSettings
    base = dict(person_suppression_enabled=True, person_iou_threshold=0.85,
                zone_enabled=True, zone_points=[(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)])
    base.update(over)
    return TunableSettings(**base)

def test_analyzer_splits_suppresses_and_zones():
    dog_in = Detection("dog", 0.9, (5, 5, 20, 20))
    dog_out = Detection("dog", 0.9, (80, 80, 95, 95))
    fake_person = Detection("person", 0.9, (30, 30, 60, 90))
    fake_dog = Detection("dog", 0.9, (31, 31, 59, 89))   # coincident with person
    det = StubDetector([[dog_in, dog_out, fake_person, fake_dog]])
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    a = DetectionAnalyzer(det, chain).analyze(np.zeros((100, 100, 3), np.uint8), _cfg())
    assert a.people == [fake_person]
    assert fake_dog not in a.dogs                 # suppressed
    assert a.candidates == [dog_in]               # zone kept only the in-zone dog
    assert dog_out in a.dogs                      # still drawn, just not a candidate

def test_chain_respects_enable_flags():
    fake_person = Detection("person", 0.9, (30, 30, 60, 90))
    fake_dog = Detection("dog", 0.9, (31, 31, 59, 89))
    det = StubDetector([[fake_person, fake_dog]])
    chain = FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])
    cfg = _cfg(person_suppression_enabled=False, zone_enabled=False)
    a = DetectionAnalyzer(det, chain).analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert a.dogs == [fake_dog] and a.candidates == [fake_dog]

def test_chain_runs_filters_in_order():
    calls = []
    class F:
        def __init__(self, tag): self.tag = tag
        def apply(self, analysis, cfg): calls.append(self.tag)
    FilterChain([F("a"), F("b")]).run(
        FrameAnalysis((1, 1, 3), [], [], []), _cfg())
    assert calls == ["a", "b"]
```

- [ ] **Step 2:** Run → fails (modules missing). Implement, adopt in pipeline, delete people.py/zone.py, retarget moved filter tests.
- [ ] **Step 3:** Verify: suite → **135 passed** (132 + 3 new; the retargeted person/zone tests replace their old versions 1:1); `grep -rn "doggy\.people\|doggy\.zone" src/ tests/` → empty; ruff clean.
- [ ] **Step 4:** Commit: `refactor: detection filter chain (adapter links) + FrameAnalysis; pipeline uses analyzer`

---

### Task 5: `decision/trigger.py` with the State pattern

**Files:**
- `git mv src/doggy/trigger.py src/doggy/decision/trigger.py` (+ `decision/__init__.py`); rewrite internals to State classes
- Tests: `git mv tests/test_trigger.py tests/decision/test_trigger.py` (+ `__init__.py`) — **assertions unchanged**; add one transition-wiring test

**Interfaces (frozen public API):** `TriggerLogic(runtime, rng=None)`; `update(detections, now) -> bool`; attributes `state` (exposing `.value` in {"IDLE","CONFIRMING","COOLDOWN"} exactly as today — keep the `TriggerState` str-Enum as the public face), `fire_confidence`, `fire_latency`.

**Implementation shape:** internal state objects `_Idle`, `_Confirming`, `_Cooldown`, each with `handle(ctx, has_dog, frame_max, m_of_n, window_full, now) -> next-state-object`; `update()` computes `frame_confs/has_dog/frame_max`, maintains the window deque, delegates to the current state object, maps state-object → public enum. All timing/window/cooldown/confidence/latency behavior byte-identical — the moved test file passing UNMODIFIED is the acceptance proof. Cooldown expiry must preserve today's semantics exactly: on expiry the frame falls through to Idle handling in the same `update()` call (a dog present that frame enters CONFIRMING).

- [ ] **Step 1 (new test, add to moved file):**

```python
def test_state_objects_round_trip():
    t = make()  # window_m=2, n=3, confirm=1.0, cooldown 10
    assert t.state is TriggerState.IDLE
    t.update(DOG, now=0.0);  assert t.state is TriggerState.CONFIRMING
    t.update(DOG, now=1.0);  assert t.state is TriggerState.COOLDOWN
    t.update(NONE, now=20.0); assert t.state is TriggerState.IDLE   # expired, no dog
```

- [ ] **Step 2:** Run moved suite → import errors; implement move + State rewrite; run `uv run pytest tests/decision/test_trigger.py -q` → all pass (old assertions untouched).
- [ ] **Step 3:** Verify: full suite → **136 passed**; `grep -rn "from doggy.trigger" src/ tests/` → empty; ruff clean.
- [ ] **Step 4:** Commit: `refactor: trigger FSM via State pattern (behavior locked by existing suite)`

---

### Task 6: `reaction/sound.py` — alerter move + Template Method

**Files:**
- `git mv src/doggy/alerter.py src/doggy/reaction/sound.py` (+ `reaction/__init__.py`); restructure to `BaseAlerter` Template Method
- Tests: `git mv tests/test_alerter.py tests/reaction/test_sound.py` (+ `__init__.py`)

**Interfaces:** `from doggy.reaction.sound import Alerter, BaseAlerter, FakeAlerter, build_alerter` (keep `Alerter` Protocol and `FakeAlerter` names — tests and web depend on them).

**Implementation shape:** `BaseAlerter.alert()` owns the skeleton — `cfg = self._runtime.get()`; `clip = self._resolve_clip(cfg)` (selected_sound if set+exists else random from clips_dir; None if no clips — exactly today's selection incl. missing-file→random fallback); clamp volume once; spawn daemon thread → `self._play(clip, volume)`. Subclasses (`SounddeviceAlerter`, `CommandAlerter`, `LogAlerter`) implement `_play` only; `_volume_args` stays with `CommandAlerter`. `build_alerter` becomes a registry `{"sounddevice": ..., "command": ..., "log": ...}`. All existing alerter tests pass (update imports only; if a test monkeypatches internals, adapt the patch target, not the assertion).

- [ ] **Step 1:** Move + restructure; update importers (main.py, web.py, pipeline.py, tests).
- [ ] **Step 2:** Verify: suite → **136 passed**; `grep -rn "doggy\.alerter" src/ tests/` → empty; ruff clean.
- [ ] **Step 3:** Commit: `refactor: sound reactions with Template Method backends + registry`

---

### Task 7: Fire path — `decision/gate.py`, `reaction/hub.py`, `reaction/recorder.py`; pipeline adopts the sequence

**Files:**
- Create `src/doggy/reaction/hub.py` (DogCaught, Reaction, ReactionHub, SafeReaction — exact shapes from the spec §Pattern 2)
- Create `src/doggy/reaction/recorder.py`:

```python
class Recorder:
    def __init__(self, store: EventStore) -> None: ...
    def record(self, frame, confidence: float, latency_s: float,
               wall_time: float, mono_ts: float) -> EventRecord:
        return self._store.add(frame, confidence, latency_s, wall_time, mono_ts=mono_ts)
```

- `git mv src/doggy/safety.py src/doggy/decision/gate.py`; rename `SafetyGovernor → FireGate`; drop the `event_store` param and `record_fire`; split into `allow(now)` (same checks, same order: safety_enabled → snooze → hourly cap) and `note_fire(now)` (deque append); keep `fires_last_hour`, `snooze`, `cancel_snooze`, `snooze_remaining`.
- Modify `src/doggy/pipeline.py` fire branch to the spec sequence: `muted = not gate.allow(now)`; on fired-and-not-muted → `record = recorder.record(frame, trigger.fire_confidence, trigger.fire_latency, time.time(), now)` → `gate.note_fire(now)` → `hub.publish(DogCaught(record, frame, now))` → status last_fire updates. `SoundReaction` (in `reaction/sound.py`: `on_dog_caught` → `self._alerter.alert()`) registered via hub; pipeline no longer calls `alerter.alert()` directly (alerter object still constructed in app wiring for `/api/test-sound`).
- Modify `src/doggy/main.py` wiring + `src/doggy/web.py` (snooze router param type is now FireGate; `/api/status` fires_this_hour source unchanged via gate).
- Tests: `git mv tests/test_safety.py tests/decision/test_gate.py` (retarget constructor: gate has no store; the record-delegation test moves to `tests/reaction/test_recorder.py`); create `tests/reaction/test_hub.py`.

**New seam tests (write first):**

```python
# tests/reaction/test_hub.py
def test_hub_publishes_in_registration_order():
    calls = []
    class R:
        def __init__(self, tag): self.tag = tag
        def on_dog_caught(self, event): calls.append(self.tag)
    hub = ReactionHub([R("sound"), R("clip")])
    hub.publish(_event())          # helper builds a DogCaught with a dummy record
    assert calls == ["sound", "clip"]

def test_safe_reaction_swallows_and_logs(caplog):
    class Boom:
        def on_dog_caught(self, event): raise RuntimeError("kaput")
    ok = []
    class Fine:
        def on_dog_caught(self, event): ok.append(1)
    hub = ReactionHub([SafeReaction(Boom()), SafeReaction(Fine())])
    hub.publish(_event())          # must not raise
    assert ok == [1]
    assert any("kaput" in r.message or "dog_caught" in r.message for r in caplog.records)

# tests/decision/test_gate.py (adapted move — semantics identical to old allow_fire/record_fire)
def test_gate_allow_note_fire_preserves_hourly_cap():
    gate = FireGate(RuntimeSettings(Settings(max_fires_per_hour=2).tunable()))
    assert gate.allow(now=0.0)
    gate.note_fire(0.0); gate.note_fire(10.0)
    assert gate.allow(now=20.0) is False          # cap hit
    assert gate.allow(now=3610.0) is True         # oldest aged out
```

**Behavior locks:** existing pipeline tests (fires-after-confirmation, conf-not-fire-edge, zone, suppression, clip e2e) pass with only constructor updates (Pipeline now takes gate/recorder/hub — update test wiring in `tests/test_pipeline.py` mechanically, assertions untouched). Snooze web tests unchanged.

- [ ] **Step 1:** Write the seam tests; run → fail. 
- [ ] **Step 2:** Implement (hub, recorder, gate split, pipeline sequence, wiring, importer updates).
- [ ] **Step 3:** Verify: full suite → **139 passed** (136 + hub×2 + gate-cap×1; the moved delegation test stays 1:1); `grep -rn "doggy\.safety\|SafetyGovernor" src/ tests/` → empty; ruff clean.
- [ ] **Step 4:** Commit: `refactor: fire path via FireGate + Recorder + ReactionHub with SafeReaction isolation`

---

### Task 8: `reaction/clips.py` — ClipService absorbs the pipeline's clip wiring

**Files:**
- `git mv src/doggy/clips.py src/doggy/reaction/clips.py`; keep `ClipBuffer`, `encode_clip` as-is; add `ClipService`:

```python
class ClipService:
    """Per-frame bufferer + pending-clip finalizer + Reaction on catches."""
    def __init__(self, store: EventStore, event_dir: Path, buffer: ClipBuffer) -> None: ...
    def on_frame(self, annotated: np.ndarray, now: float, cfg: TunableSettings) -> None:
        # if cfg.clips_enabled: imencode('.jpg', annotated) -> buffer.push(now, bytes)
    def on_dog_caught(self, event: DogCaught) -> None:
        # if clips enabled at fire time: append pending {id, fire_ts, end}
    def finalize_due(self, now: float, cfg: TunableSettings) -> None:
        # encode + attach_clip for pendings with end <= now (exact logic moved
        # from pipeline._finalize_clips / pending list, unchanged)
```

  (`on_dog_caught` needs cfg: read via the runtime injected at construction OR carry `clips_enabled` decision in the pipeline exactly as today — preserve today's toggle semantics: pending registered only when `cfg.clips_enabled` at the fire frame; finalization runs regardless. Inject `runtime: RuntimeSettings` and read `runtime.get()` where today's pipeline read `cfg`.)
- Modify `src/doggy/pipeline.py`: delete `_clip_buffer`/`_pending_clips`/finalize internals; call `clip_service.on_frame(annotated, now, cfg)` and `clip_service.finalize_due(now, cfg)`; hub already publishes to it (registered as a reaction in wiring).
- Tests: `git mv tests/test_clips.py tests/reaction/test_clips.py` (retarget imports; buffer/encode tests unchanged); `tests/test_pipeline.py::test_pipeline_finalizes_clip_after_postroll` must pass with only constructor-wiring updates.

- [ ] **Step 1:** Move + extract; update wiring (main.py) and pipeline; retarget tests.
- [ ] **Step 2:** Verify: suite → **139 passed**; `grep -rn "doggy\.clips import\|_pending_clips" src/doggy/pipeline.py` → empty; ruff clean.
- [ ] **Step 3:** Commit: `refactor: ClipService owns clip capture; pipeline delegates`

---

### Task 9: Facade finalization + composition root (`app.py`, `main.py` shim)

**Files:**
- `git mv src/doggy/main.py src/doggy/app.py`; then create new `src/doggy/main.py`:

```python
"""Entry-point shim: the installed console script resolves doggy.main:main.

The Pi service starts with `uv run --no-sync` (no re-install on deploy), so this
module name must never move again -- the real wiring lives in doggy.app.
"""
from doggy.app import main

if __name__ == "__main__":
    main()
```

- `app.py`: same wiring as before plus construction of the new collaborators in dependency order (store → gate → recorder → alerter → SoundReaction → ClipService → `ReactionHub([SafeReaction(SoundReaction(alerter)), SafeReaction(clip_service)])` → camera/detector/`FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()])`/`DetectionAnalyzer` → `Pipeline(...)`).
- `src/doggy/pipeline.py`: final tidy to the spec's ~60-line Facade shape — `run_once` = clock-once → analyze → annotate+buffer → clip on_frame/finalize → trigger → gate → fire sequence → status; `run()` = capture thread + thermal/power/pacer loop, both unchanged in behavior. No dead code, no unused imports.
- Tests: none new — `tests/test_smoke.py` and the pipeline suite are the lock. Add one import-contract test to `tests/test_smoke.py`:

```python
def test_entry_point_shim_survives():
    # The Pi's installed console script resolves doggy.main:main under --no-sync;
    # this import path must never break.
    from doggy.main import main as entry
    from doggy.app import main as real
    assert entry is real
```

- [ ] **Step 1:** Shim test first (fails: no doggy.app) → implement move/shim/tidy.
- [ ] **Step 2:** Verify: suite → **140 passed**; `wc -l src/doggy/pipeline.py` ≤ ~90 (60-line target ± docstrings); `uv run python -c "from doggy.main import main; print('entry ok')"`; ruff clean.
- [ ] **Step 3:** Commit: `refactor: app.py composition root + main.py entry shim; pipeline is a thin facade`

---

### Task 10: `web/` split (app, envfile, routers, static)

**Files:**
- `git mv src/doggy/web.py src/doggy/web/app.py` (+ `web/__init__.py` re-exporting `create_app`, `serve` for wiring imports); `git mv src/doggy/static src/doggy/web/static`
- Extract `_write_env` → `src/doggy/web/envfile.py` (name it `write_env`, keep a `_write_env` alias if tests reference it)
- Create `src/doggy/web/routers/{__init__,status,settings,events,sounds,snooze}.py`; each exposes `build_router(...explicit deps...) -> APIRouter` with the routes exactly as assigned in the spec §Web split; `web/app.py`'s `create_app` keeps its signature, builds the routers, includes them, serves `GET /` from `Path(__file__).parent / "static" / "index.html"`, and keeps `/stream.mjpg` streaming behavior identical (same `_MJPEG_FRAME_INTERVAL_SECONDS`, same boundary).
- Tests: `git mv tests/test_web.py tests/web/test_api.py` (+ `__init__.py`) — assertions untouched; they are the byte-identical-contract proof.

- [ ] **Step 1:** Moves + split; every route path/method/response identical; update importers (`app.py` wiring, any test imports of `_write_env`).
- [ ] **Step 2:** Verify: suite → **140 passed** (web tests all green unmodified); `grep -rn "doggy\.web import\|doggy/static" src/ tests/ scripts/` — deploy scripts rsync the whole `src/doggy` tree so no script changes needed; confirm nothing references the old `src/doggy/static` path in Python; ruff clean.
- [ ] **Step 3:** Commit: `refactor: split web into app + envfile + per-domain routers`

---

### Task 11: ARCHITECTURE.md + final sweep

**Files:**
- Create `ARCHITECTURE.md` (repo root): one page. Sections: the domain map (tree from the spec with one-line purposes); **Applied patterns** — Observer (`reaction/hub.py`), Decorator (`SafeReaction`), Chain of Responsibility + Adapter (`vision/filters/`), Strategy registries (`vision/camera.py`, `reaction/sound.py`), State (`decision/trigger.py`), Template Method (`reaction/sound.py`), Facade (`pipeline.py`), composition root (`app.py`) — each with 2-3 lines of *why here*; **Patterns already present** — caching Proxy (`hardware/power.py`), Iterator (`vision/camera.py` frames()), Null Object (`LogAlerter`), single-slot drop-oldest buffer (`core/status.py` FrameBuffer); **Rejected on purpose** — Mediator, Builder/Abstract Factory, Command, hexagonal ports, Singleton — one line each on why (Speculative Generality at 1,700 lines); **Invariants** — `main.py` shim (entry metadata + `--no-sync`), single event writer, `run_once` reads the clock once, frozen external contract. Plain language, no emoji, link paths in backticks.
- Final sweep: `grep -rn "from doggy import\|from doggy\.\(config\|state\|pacer\|events import\|thermal\|power\|detection\|camera\|detector\|people\|zone\|trigger\|safety\|alerter\|clips import\|web import\)" src/ tests/` → nothing stale; `ls src/doggy/*.py` → exactly `main.py app.py pipeline.py __init__.py`; `uv run ruff check .` → no NEW findings vs the pre-branch baseline (the handful of pre-existing test-file findings are out of scope); full suite green.
- Update README's repo-layout mention if any (README has no tree today — add one line under "How it works" pointing at ARCHITECTURE.md).

- [ ] **Step 1:** Write ARCHITECTURE.md; run the sweep greps; fix stragglers.
- [ ] **Step 2:** Verify: suite → **140 passed**; commit: `docs: ARCHITECTURE.md pattern map + restructure sweep`

---

## Deploy & acceptance (after final review, controller-run)

Code-only change (no dep edits) → offline-safe deploy: `rsync -az --delete --exclude '__pycache__' src/doggy/ doggy@doggypi.local:doggy/src/doggy/` then `sudo systemctl restart doggy`. The `--delete` is REQUIRED (removes the old flat modules on the Pi; without it stale `state.py`/`safety.py` etc. linger). Verify: service `active`; `GET /api/status` healthy; dashboard renders; `POST /api/test-sound` audible; one detection cycle shows in history. `uv run doggy` entry works because `main.py` shim + unchanged pyproject mean the installed entry-point metadata stays valid with `--no-sync`.

## Expected test-count ledger

132 (baseline) → T4: 135 → T5: 136 → T7: 139 → T9: 140 → end: **140 passed, 1 deselected**.
