# Codebase Restructure Design — Domain Packages + Deliberate Patterns

**Repo:** https://github.com/devjerry0/watchdoggy

**Goal:** Reorganize the flat 16-module `src/doggy/` package into domain packages (see → decide → react) and introduce exactly the design patterns that earn their keep — Observer for catch reactions, Chain of Responsibility (with per-link Adapters) for detection filters, Strategy registries for hardware backends, a Facade pipeline, and a composition root — with **zero behavior change** and the live appliance's external contract frozen.

**Architecture:** The 202-line `pipeline.py` (which imports 15 of 16 modules) becomes a ~60-line Facade over five collaborators: frame source, detection analysis (detector + filter chain), trigger, fire gate, and a reaction hub. Everything else moves into small domain packages. All moves via `git mv` to preserve history.

**Tech Stack:** unchanged — Python 3.11, FastAPI, pydantic-settings, OpenCV, uv/pytest. No new dependencies.

## Global Constraints

- **Zero behavior change.** Every observable behavior — detection semantics, trigger timing, gate/mute logic, event persistence, clip capture, HTTP responses, dashboard — identical before and after.
- **Frozen external contract:** every `DOGGY_*` env var name; every HTTP route path and JSON shape; `.env` round-trip format; `events.jsonl` schema and `events/` directory layout; the `uv run doggy` console script; deploy scripts (`rsync` of `src/doggy/`); systemd units and drop-ins (`UV_OFFLINE=1`, `--no-sync`).
- **Entry-point shim is mandatory:** `src/doggy/main.py` remains, containing only `from doggy.app import main` (+ `__main__` guard). The Pi's installed entry-point metadata resolves `doggy.main:main`, and the service starts with `uv run --no-sync` (no re-sync on deploy), so deleting/renaming `main.py` bricks the appliance. `pyproject.toml` `[project.scripts]` stays `doggy = "doggy.main:main"`.
- **All moves via `git mv`** (history preserved). Full fast suite (`uv run pytest -m "not slow"`, currently 132 tests) green after every task; `uv run ruff check` clean on touched files.
- **No new dependencies, no async rewrite, no renames of config fields or endpoints.**

## Target layout (source → destination)

```
src/doggy/
  main.py                   # SHIM: from doggy.app import main
  app.py                    # composition root (wiring from old main.py)
  pipeline.py               # thin Facade orchestrator (rewritten, ~60 lines)
  core/
    config.py               # ← config.py (unchanged content)
    runtime.py              # ← state.py: RuntimeSettings
    status.py               # ← state.py: Status, StatusStore, CONFIDENCE_DECIMALS
    pacer.py                # ← pacer.py (pure timing utility)
  vision/
    detection.py            # ← detection.py (Detection, TARGET_LABEL, PERSON_LABEL)
    camera.py               # ← camera.py (+ registry factory, see Strategy)
    detector.py             # ← detector.py (+ registry factory)
    analysis.py             # NEW: FrameAnalysis dataclass + DetectionAnalyzer
    annotate.py             # ← pipeline.py: annotate(), _draw_box(), color constants
    filters/
      base.py               # NEW: DetectionFilter protocol + FilterChain
      person.py             # ← people.py logic, adapted to DetectionFilter
      zone.py               # ← zone.py logic, adapted to DetectionFilter
  decision/
    trigger.py              # ← trigger.py (TriggerLogic, unchanged)
    gate.py                 # ← safety.py: FireGate (rename of SafetyGovernor, minus record_fire)
  reaction/
    hub.py                  # NEW: DogCaught event + Reaction protocol + ReactionHub
    recorder.py             # NEW: Recorder (the EventStore write, from SafetyGovernor.record_fire)
    sound.py                # ← alerter.py (backends + registry factory) + SoundReaction
    clips.py                # ← clips.py (ClipBuffer, encode_clip) + ClipService
  events/
    store.py                # ← events.py (EventStore, EventRecord — unchanged content)
  hardware/                 # named `hardware` (not `platform`) to avoid stdlib-name confusion
    thermal.py              # ← thermal.py
    power.py                # ← power.py
  web/
    app.py                  # ← web.py: create_app + serve + GET / (index)
    envfile.py              # ← web.py: _write_env
    routers/
      status.py             # GET /api/status, GET /stream.mjpg
      settings.py           # PATCH /api/settings, POST /api/settings/save
      events.py             # GET /api/events, DELETE /api/events/{id}, POST /api/events/clear,
                            #   GET /api/stats, GET /events/{name}, GET /clips/{name}
      sounds.py             # GET/POST /api/sounds, POST /api/test-sound
      snooze.py             # POST /api/snooze, POST /api/snooze/cancel
    static/index.html       # ← static/index.html (unchanged file)
```

Old flat modules are deleted after their content moves (no compatibility re-export layer inside the package; only `main.py` keeps its name for the entry point).

## Pattern 1 — Chain of Responsibility with Adapter links (`vision/`)

**FrameAnalysis** (`vision/analysis.py`) carries everything downstream consumers need:

```python
@dataclass
class FrameAnalysis:
    shape: tuple[int, ...]            # frame.shape
    people: list[Detection]           # person-labeled detections
    dogs: list[Detection]             # dog-labeled, surviving suppression (what gets drawn)
    candidates: list[Detection]       # dogs that may trigger (post-zone); starts == dogs
```

**DetectionFilter** (`vision/filters/base.py`):

```python
class DetectionFilter(Protocol):
    def apply(self, analysis: FrameAnalysis, cfg: TunableSettings) -> None: ...

class FilterChain:
    def __init__(self, filters: Sequence[DetectionFilter]) -> None: ...
    def run(self, analysis: FrameAnalysis, cfg: TunableSettings) -> None:
        # walks filters in order; each mutates the analysis
```

Each concrete filter is an **Adapter**: it wraps its existing logic behind the common interface and checks its own enable flag (chain stays dumb):

- `person.py` `PersonSuppressionFilter`: if `cfg.person_suppression_enabled` and `analysis.people`: narrow `analysis.dogs` (and `candidates`) with the existing IoU-coincidence suppression (`iou`, `suppress_dogs_overlapping_people` move here).
- `zone.py` `ZoneInclusionFilter`: if `cfg.zone_enabled` and ≥3 points: narrow `analysis.candidates` with the existing cached-mask test (mask cache stays instance state, keyed on points+shape exactly as today; <3 points = pass-through).

**DetectionAnalyzer** (`vision/analysis.py`) = detector + chain: `analyze(frame, cfg) -> FrameAnalysis` runs `detector.detect`, splits dog/person by label, seeds the analysis, runs the chain. Chain order fixed in the composition root: `[PersonSuppressionFilter(), ZoneInclusionFilter()]`.

**Preserved exactly:** suppressed dogs are not drawn; `annotate` receives post-suppression dogs, in-zone set (`analysis.candidates`), people only when `person_suppression_enabled` (else None), zone points only when `zone_enabled`; status reports `dogs=len(candidates)`, `people=len(people) if person_suppression_enabled else 0`; the trigger sees `candidates`.

## Pattern 2 — Observer for catch reactions (`reaction/`)

```python
@dataclass(frozen=True)
class DogCaught:
    record: EventRecord          # the persisted catch (id, confidence, latency, thumb)
    frame: np.ndarray            # raw frame at the fire edge
    mono_ts: float               # pipeline monotonic time of the fire

class Reaction(Protocol):
    def on_dog_caught(self, event: DogCaught) -> None: ...

class ReactionHub:
    def __init__(self, reactions: Sequence[Reaction]) -> None: ...
    def publish(self, event: DogCaught) -> None:
        # dumb fan-out: calls each reaction in registration order.
        # Crash isolation is NOT the hub's job -- see SafeReaction (Decorator).

class SafeReaction:
    """Decorator: wraps a Reaction so an exception is logged (log.exception)
    and swallowed -- one failing reaction must not stop the others or kill the
    detect loop. The composition root wraps every registered reaction."""
    def __init__(self, inner: Reaction) -> None: ...
    def on_dog_caught(self, event: DogCaught) -> None: ...
```

**Why recording is NOT an observer:** `ClipService` needs the `EventRecord.id` to name and attach the clip, so persistence is the primary effect that *produces* the event, not a peer subscriber. Sequence on a fire (pins today's semantics):

1. `gate.allow(now)` was already checked this frame (drives `muted` status every frame, fired or not).
2. `record = recorder.record(frame, trigger.fire_confidence, trigger.fire_latency, time.time(), now)` → `EventStore.add` (single writer unchanged).
3. `gate.note_fire(now)` → appends to the hourly-rate deque (exactly when `SafetyGovernor.record_fire` appended today).
4. `hub.publish(DogCaught(record, frame, now))` → `SoundReaction` (plays the deterrent via the alerter), `ClipService` (registers the pending clip).
5. Pipeline updates `last_fire_ts`/`last_fire_thumb` status from `record`.

**Registered reactions at boot:** `[SafeReaction(SoundReaction(alerter)), SafeReaction(clip_service)]`. A future turret/notification = one new `Reaction` wrapped and registered in `app.py`.

**ClipService** (`reaction/clips.py`) is both a per-frame stage and a reaction: the pipeline calls `clip_service.on_frame(annotated_frame, now, cfg)` every loop (JPEG-push into the rolling buffer when `clips_enabled`) and `clip_service.finalize_due(now, cfg)` (encode pendings whose post-roll elapsed → `event_store.attach_clip`); `on_dog_caught` registers the pending. Buffer window, encode fallback (mp4 → webp), retention semantics unchanged.

## Pattern 3 — Strategy registries (factories)

Replace `if/elif` factories with module-level registries; behavior identical, unknown key error messages preserved or improved:

- `vision/camera.py`: `_BACKENDS = {"opencv": ..., "file": ...}`; `build_camera(settings)`.
- `vision/detector.py`: `build_detector(settings, runtime)` (yolo; `StubDetector` stays for tests).
- `reaction/sound.py`: `_BACKENDS = {"sounddevice": ..., "command": ..., "log": ...}`; `build_alerter(settings, runtime)`. `selected_sound`/`max_volume` handling unchanged.

## Pattern 3b — State for the trigger FSM (`decision/trigger.py`)

`TriggerLogic` is a hand-rolled finite state machine (`IDLE → CONFIRMING → COOLDOWN`, `if/elif` on a state enum). Apply the State pattern: each state becomes a small class owning its own frame-handling rules and returning the next state.

```python
class TriggerState(Protocol):
    def on_frame(self, ctx: "TriggerLogic", has_dog: bool, frame_max: float,
                 m_of_n: bool, window_full: bool, now: float) -> "TriggerState":
        """Handle one frame; return the state to be in next (may be self).
        Sets ctx.fired for the frame via ctx (fire edge only in Confirming)."""

class Idle: ...        # dog seen -> Confirming (records confirm_start / confirm_max)
class Confirming: ...  # window fails -> Idle; confirmed -> sets fire_confidence/
                       #   fire_latency, computes cooldown_until -> Cooldown
class Cooldown: ...    # expired -> Idle (then normal Idle handling for this frame)
```

`TriggerLogic.update(detections, now) -> bool` keeps its exact public signature, the `state` property keeps exposing the current state name string (the dashboard's `IDLE / CONFIRMING / COOLDOWN`), and every existing trigger test passes unmodified — window bookkeeping, M-of-N, confirm timing, jittered cooldown, `fire_confidence`, `fire_latency` all byte-identical in behavior. Payoff: an escalation stage later ("chirp, then louder if she stays") or the squirrel-turret FSM (`TRACKING → AIMING → FIRING`) is a new state class, not a rewrite of a conditional block.

## Pattern 3c — Template Method for sound backends (`reaction/sound.py`)

Every alerter shares the same skeleton — resolve the clip (`selected_sound` or random from `clips_dir`), clamp volume, spawn the play thread — and differs only in how it plays. Today that skeleton is duplicated across backends. Fix with Template Method:

```python
class BaseAlerter:
    def alert(self) -> None:
        # template: cfg = runtime.get(); clip = self._resolve_clip(cfg);
        # if clip: spawn daemon thread -> self._play(clip, clamped_volume)
    def _resolve_clip(self, cfg) -> Path | None: ...   # shared, one implementation
    def _play(self, clip: Path, volume: float) -> None: raise NotImplementedError

class SounddeviceAlerter(BaseAlerter): ...  # scales samples by volume
class CommandAlerter(BaseAlerter): ...      # pw-play/afplay with volume args
class LogAlerter(BaseAlerter): ...          # Null Object: logs, plays nothing
```

Existing alerter tests keep passing; clip-selection behavior (missing file → random fallback) unchanged.

## Pattern 4 — FireGate (`decision/gate.py`)

`SafetyGovernor` renames to `FireGate` and loses persistence (moved to `Recorder`):

```python
class FireGate:
    def __init__(self, runtime: RuntimeSettings) -> None: ...   # no EventStore anymore
    def allow(self, now: float) -> bool        # safety_enabled + snooze + hourly cap (same order/semantics)
    def note_fire(self, now: float) -> None    # append to rate deque
    def fires_last_hour(self, now: float) -> int
    def snooze(self, seconds: float, now: float) / cancel_snooze() / snooze_remaining(now)
```

The web snooze router and status wiring use `FireGate` (parameter names may change internally; routes unchanged).

## Pattern 5 — Facade pipeline + composition root

**`pipeline.py`** (rewritten, ~60 lines) holds: `raw/annotated buffers, analyzer, trigger, gate, recorder, hub, clip_service, status, pacer, thermal, power, clock`. `run_once(frame)`:

```
now = clock()                     # exactly once (invariant)
cfg = runtime.get()
analysis = analyzer.analyze(frame, cfg)
annotated = annotate(frame, analysis, cfg)   # + buffer set
clip_service.on_frame(annotated, now, cfg)   # service does the JPEG encode internally
clip_service.finalize_due(now, cfg)
fired = trigger.update(analysis.candidates, now)
muted = not gate.allow(now)
if fired and not muted: (sequence from Pattern 2)
status.update(...)                # same fields/values as today
return fired and not muted
```

`run()` loop (capture thread, thermal/power reads, pacing, fps) unchanged in behavior.

**`app.py`** (composition root, from old `main.py`): builds settings → logging → runtime/status/buffers → `EventStore` → `FireGate` → `Recorder` → alerter/`SoundReaction` → `ClipService` → `ReactionHub` → camera/detector/`FilterChain`/`DetectionAnalyzer` → `Pipeline` → web thread (`serve(...)`) → signals → `pipeline.run(stop)`. `main.py` = shim.

## Web split (`web/`)

`create_app` keeps its public signature style (explicit dependencies in, `FastAPI` out) but internally builds routers: each `web/routers/*.py` exposes `build_router(<deps>) -> APIRouter`, included by `web/app.py`. Route paths, methods, status codes, and JSON shapes are byte-identical (existing `tests/test_web.py` assertions are the proof). `GET /` serves `web/static/index.html` (path constant updated). `envfile.py` holds `_write_env` (same format, same tests).

## Documentation deliverable — `ARCHITECTURE.md`

A repo-root `ARCHITECTURE.md` mapping every package to its pattern and the reason it earns its keep: the five applied patterns above, plus naming the patterns that were **already present** (PowerMonitor = caching Proxy over `vcgencmd`; `camera.frames()` = Iterator; `LogAlerter` = Null Object; `FrameBuffer` = single-slot drop-oldest buffer) and a short "rejected patterns and why" section (Mediator, Builder, Command, hexagonal ports — Speculative Generality at this scale). One page, links into the packages.

## Tests

- Mirror the packages: `tests/core/`, `tests/vision/` (incl. `tests/vision/filters/`), `tests/decision/`, `tests/reaction/`, `tests/events/`, `tests/hardware/`, `tests/web/`; `tests/test_pipeline.py` and `tests/test_smoke.py` stay at root. Moves via `git mv`, imports updated, zero assertions weakened.
- New seam tests: FilterChain runs links in order and respects enable flags; ReactionHub fans out in registration order; SafeReaction swallows+logs a raising inner reaction (others still run, nothing escapes); FireGate `allow`/`note_fire` reproduce today's `allow_fire`/`record_fire` deque semantics (incl. snooze); DetectionAnalyzer end-to-end (stub detector → analysis fields); trigger State classes covered by the existing trigger suite unchanged (behavioral lock) plus one test asserting the state-object transition wiring (Idle→Confirming→Cooldown→Idle round trip).
- The full existing suite (132 tests) passes unmodified in behavior at every task boundary.

## Migration strategy

Feature branch `refactor/domain-packages`. One package per task, each ending green + ruff-clean, moves via `git mv` first, then edits. Task order: core → events → hardware → vision (incl. filters) → decision → reaction → pipeline+app (the Facade rewrite — the only task with real logic movement) → web split → test-tree mirror + final sweep. Subagent-driven development with per-task review and a final whole-branch Opus review. Deploy = rsync + restart (code-only, no new deps, offline-safe per the firewalled-appliance runbook); verify dashboard + `/api/status` + a test sound after restart.

## Non-goals

No behavior changes; no new features; no endpoint/env/config renames; no new dependencies; no async; no ABCs where a Protocol + one registry suffices; no repository/port interfaces with a single implementation; no DI container. The refactoring.guru catalog's "when not to use" guidance governs: if a pattern's only justification is symmetry, it stays out.
