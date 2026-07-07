# Watcher Smarts Implementation Plan (Deterrence Lab batch, Plan 1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Class-configurable watching (dog/cat/bird), counter inventory with theft forensics, post-fire outcome measurement, escalation ladder, per-sound Deterrence Lab stats, and a weekly report card.

**Architecture:** The detector stops discarding non-dog classes; a `FrameAnalysis` carries targets/people/inventory; a new `OutcomeWatcher` (per-frame observer + hub Reaction, same shape as `ClipService`) measures how long the target stayed after a fire, diffs the counter inventory, and drives escalation strikes through the existing `FireGate` and alerter. `EventStore` grows attach methods mirroring `attach_clip`.

**Tech Stack:** Existing only: Python 3.11, pydantic, FastAPI, OpenCV, vanilla JS dashboard. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-07-deterrence-lab-batch-design.md` (sections 1, 2, 2b, 3, 4, and report card from 9).

## Global Constraints

- Fully local appliance; egress firewall stays armed; NO new packages (uv.lock unchanged).
- `src/doggy/main.py` entry shim must not move or change.
- Existing `events.jsonl` files must load unchanged: every new `EventRecord` field defaults when absent.
- `EventStore` stays the single writer; every new public method takes `self._lock` like `attach_clip`.
- Dashboard copy: plain language a non-engineer understands; no emoji; sentence case.
- Stats/report-card bucketing uses Pi-local wall time (`datetime.fromtimestamp`), like `EventStore.stats()`.
- Per task: `uv run pytest -m "not slow"` green and `uv run ruff check src tests` clean before commit.
- Commits as `devjerry0 <99199491+devjerry0@users.noreply.github.com>`; no personal info in the repo.
- `TunableSettings` is frozen pydantic; new knobs get `DOGGY_`-prefixed env names automatically and must be PATCHable via `/api/settings` (dict-merge + revalidate already handles any field).

---

### Task 1: Watch-for classes (dog / cat / bird)

**Files:**
- Modify: `src/doggy/core/config.py`, `src/doggy/vision/detection.py`, `src/doggy/vision/detector.py`, `src/doggy/vision/analysis.py`, `src/doggy/vision/filters/person.py`, `src/doggy/pipeline.py`, `src/doggy/core/status.py`, `src/doggy/web/static/index.html`
- Test: `tests/core/test_config.py`, `tests/vision/test_analysis.py`, `tests/vision/filters/test_person.py`, plus every test currently using `FrameAnalysis.dogs` / `Status.dogs` / `TARGET_LABEL` (find with `grep -rn "TARGET_LABEL\|\.dogs\b\|dogs=" src tests`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TunableSettings.target_labels: tuple[str, ...]` (detected classes, default `("dog",)`, non-empty) and `TunableSettings.alert_labels: tuple[str, ...]` (classes that may fire, default `("dog",)`, must be a subset of `target_labels`, MAY be empty = monitor mode); `detection.ANIMAL_TARGETS = ("dog", "cat", "bird")`; `FrameAnalysis.targets` (renamed from `dogs`; all detected animals, drawn) with `candidates` seeded only from alert-class detections; `Status.targets` (renamed from `dogs`; counts detected animals); status JSON key `targets`.

- [ ] **Step 1: Write failing config tests** in `tests/core/test_config.py`:

```python
def test_target_labels_default_and_parsing():
    assert TunableSettings().target_labels == ("dog",)
    assert TunableSettings().alert_labels == ("dog",)
    got = TunableSettings(target_labels="dog,cat", alert_labels="dog,cat")
    assert got.target_labels == ("dog", "cat") and got.alert_labels == ("dog", "cat")
    assert TunableSettings(target_labels='["cat"]', alert_labels='["cat"]').target_labels == ("cat",)
    assert TunableSettings(target_labels=["bird", "dog"], alert_labels=["dog"]).alert_labels == ("dog",)


def test_target_labels_rejects_unknown_and_empty():
    with pytest.raises(ValidationError):
        TunableSettings(target_labels="dragon")
    with pytest.raises(ValidationError):
        TunableSettings(target_labels=[])


def test_alert_labels_subset_rule():
    # detect-only birds: valid; alerting on an undetected class: not.
    ok = TunableSettings(target_labels="dog,bird", alert_labels="dog")
    assert ok.alert_labels == ("dog",)
    assert TunableSettings(target_labels="dog", alert_labels=[]).alert_labels == ()
    with pytest.raises(ValidationError):
        TunableSettings(target_labels="dog", alert_labels="cat")
```

- [ ] **Step 2: Run them** — `uv run pytest tests/core/test_config.py -q`. Expected: FAIL (no field).

- [ ] **Step 3: Implement.** In `src/doggy/vision/detection.py` replace the `TARGET_LABEL` constant block with:

```python
# The classes the watcher may act on, offered as the dashboard menu. person is
# also always detected (never alerted on) to suppress misclassified humans.
ANIMAL_TARGETS = ("dog", "cat", "bird")
PERSON_LABEL = "person"
```

In `src/doggy/core/config.py` add to `TunableSettings` (after `confidence`):

```python
    # Which animals are detected (drawn + counted), comma-separated in .env
    # ("dog,cat"), and which of those may fire the deterrent. alert_labels
    # must be a subset of target_labels; empty alert_labels = monitor mode.
    target_labels: tuple[str, ...] = ("dog",)
    alert_labels: tuple[str, ...] = ("dog",)
```

and validators (import `field_validator` from pydantic; `ANIMAL_TARGETS` from `doggy.vision.detection` would be a core->vision import cycle, so define the allowed set locally):

```python
    _ALLOWED_TARGETS = ("dog", "cat", "bird")

    @field_validator("target_labels", "alert_labels", mode="before")
    @classmethod
    def _parse_labels(cls, v):
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                v = json.loads(s)
            else:
                v = [part.strip() for part in s.split(",") if part.strip()]
        labels = tuple(dict.fromkeys(v))  # de-dupe, keep order
        unknown = [x for x in labels if x not in cls._ALLOWED_TARGETS]
        if unknown:
            raise ValueError(f"unknown watch classes: {unknown}")
        return labels
```

plus, inside the existing `_check_ranges` model validator, the cross-field rules:

```python
        if not self.target_labels:
            raise ValueError("select at least one animal to watch for")
        extra = [x for x in self.alert_labels if x not in self.target_labels]
        if extra:
            raise ValueError(f"alert classes must also be detected: {extra}")
```

(`import json` at top. Make `detection.ANIMAL_TARGETS` reference the same values; a comment in each file pointing at the other is enough — a shared import would create a cycle.)

- [ ] **Step 4: Rename `dogs` -> `targets` through the vision/status spine.** In `src/doggy/vision/analysis.py`:

```python
@dataclass
class FrameAnalysis:
    """The detection state for one frame, narrowed in place by the filter chain.

    - `targets`: watched-class detections that survived suppression (all drawn).
    - `candidates`: the subset of `targets` still eligible to trigger (post-zone).
    - `people`: person-labeled detections (used for suppression / overlay).
    """

    shape: tuple[int, ...]
    people: list[Detection]
    targets: list[Detection]
    candidates: list[Detection]
```

and in `DetectionAnalyzer.analyze`:

```python
        detections = self._detector.detect(frame)
        detected = set(cfg.target_labels)
        alertable = set(cfg.alert_labels)
        targets = [d for d in detections if d.label in detected]
        people = [d for d in detections if d.label == PERSON_LABEL]
        analysis = FrameAnalysis(
            shape=frame.shape, people=people, targets=targets,
            # Only alert-class animals may trigger; detect-only ones are
            # drawn (in the ignored grey) but never enter the candidate set.
            candidates=[d for d in targets if d.label in alertable])
```

(import `PERSON_LABEL` only; drop `TARGET_LABEL`.)

In `src/doggy/vision/detector.py` replace the module-level `_RETURNED_LABELS` and the filter line in `YoloDetector.detect` with per-call logic:

```python
    def detect(self, frame: np.ndarray) -> list[Detection]:
        cfg = self._runtime.get()
        wanted = set(cfg.target_labels) | {PERSON_LABEL}
        results = self._model.predict(
            frame, conf=cfg.confidence, device=self._device, verbose=False
        )
        out: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if label not in wanted:
                    continue
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                out.append(Detection(label, float(box.conf[0]), (x1, y1, x2, y2)))
        return out
```

In `src/doggy/vision/filters/person.py`: rename `suppress_dogs_overlapping_people` -> `suppress_targets_overlapping_people` (parameter `dogs` -> `targets`; docstring: "A person misclassified as a target..."), and in `PersonSuppressionFilter.apply` narrow `analysis.targets` then reseed candidates WITHOUT resurrecting detect-only animals:

```python
        analysis.targets = suppress_targets_overlapping_people(
            analysis.targets, analysis.people, cfg.person_iou_threshold)
        alertable = set(cfg.alert_labels)
        analysis.candidates = [d for d in analysis.targets if d.label in alertable]
```

In `src/doggy/core/status.py`: `dogs: int = 0` -> `targets: int = 0` (comment: "watched animals in frame").

In `src/doggy/pipeline.py` `run_once`: `analysis.dogs` -> `analysis.targets` in the annotate call, and the status update becomes `targets=len(analysis.targets)` — it counts DETECTED animals (matching the "in view" label and monitor mode, where candidates is always empty while animals are still drawn). The certainty readout stays fed from `candidates` (it describes what may fire). Comment on the annotate line: `targets` are drawn.

- [ ] **Step 5: Dashboard.** In `src/doggy/web/static/index.html`:
  - Settings card, directly under the "Ignore people" toggle, add a two-column detect/alert grid:

```html
        <div class="field">
          <div class="row"><label>Watch for</label></div>
          <div class="desc">Detect means it shows on the live view. Alert means the deterrent fires. People never set it off.</div>
          <div class="watchgrid" id="watch_grid">
            <span></span><span class="wg-h">Detect</span><span class="wg-h">Alert</span>
            <span class="wg-name">Dogs</span>
            <input type="checkbox" data-kind="detect" value="dog" />
            <input type="checkbox" data-kind="alert" value="dog" />
            <span class="wg-name">Cats</span>
            <input type="checkbox" data-kind="detect" value="cat" />
            <input type="checkbox" data-kind="alert" value="cat" />
            <span class="wg-name">Birds</span>
            <input type="checkbox" data-kind="detect" value="bird" />
            <input type="checkbox" data-kind="alert" value="bird" />
          </div>
        </div>
```

  with CSS next to `.toggle` styles:

```css
  .watchgrid{display:grid;grid-template-columns:1fr auto auto;gap:.55rem 1.4rem;
             align-items:center;margin-top:.6rem}
  .watchgrid .wg-h{font-family:var(--mono);font-size:.62rem;letter-spacing:.14em;
             text-transform:uppercase;color:var(--stone);justify-self:center}
  .watchgrid .wg-name{font-size:.92rem;font-weight:550}
  .watchgrid input{accent-color:var(--lamp);width:17px;height:17px;margin:0;
             justify-self:center;cursor:pointer}
  .watchgrid input:disabled{opacity:.3;cursor:default}
```

  - JS rules for the grid, applied on any checkbox change, then one `patch({target_labels: detected, alert_labels: alerted})`:
    - Alert checkbox is `disabled` while its row's Detect is unchecked; unchecking Detect also unchecks Alert.
    - At least one Detect must stay checked: if a change would empty the detect set, revert it (`e.target.checked = true`) and skip the patch.
    - Empty Alert set is allowed (monitor mode: it watches, never reacts).
    In `poll()`, when no grid checkbox has focus, sync all six from `s.settings.target_labels` / `s.settings.alert_labels` (and the disabled states). Replace `s.dogs` with `s.targets` for the `dogs` readout element, and make the labels dynamic:

```js
function targetNoun(labels, plural){
  if (labels.length === 1){
    const names = {dog:["dog","Dogs"], cat:["cat","Cats"], bird:["bird","Birds"]};
    return names[labels[0]] ? names[labels[0]][plural ? 1 : 0] : labels[0];
  }
  return plural ? "Animals" : "intruder";
}
```

  Each `poll()`: the "in view" readout label uses the DETECTED classes (`targetNoun(s.settings.target_labels, true) + " in view"`); the certainty labels ("Certainty it's a dog" stat label, "How sure it must be it's a dog" slider label) and the CONFIRMING state word use the ALERT classes — they describe what fires. Single alert class -> capitalized noun + " spotted"; several -> "Intruder spotted"; empty alert set -> fall back to the detected classes for the certainty labels (the wording still reads naturally and nothing can fire). Give the three labels ids (`lbl_inview`, `lbl_certainty`, `lbl_confidence`) so JS can set textContent; keep the current English as the static HTML default.

- [ ] **Step 6: Update every remaining reference.** `grep -rn "TARGET_LABEL\|suppress_dogs\|\.dogs\b\|dogs=" src tests` and fix all hits (tests construct `FrameAnalysis(..., dogs=...)` and assert `status.dogs`; stub detections keep the literal `"dog"` label). Add one new analyzer test:

```python
def test_analyzer_honors_target_labels():
    cat = Detection("cat", 0.9, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[cat, dog]]), FilterChain([]))
    cfg = TunableSettings(target_labels=["cat"], alert_labels=["cat"])
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert analysis.targets == [cat]


def test_detect_only_class_never_becomes_candidate():
    bird = Detection("bird", 0.9, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[bird, dog]]), FilterChain([]))
    cfg = TunableSettings(target_labels=["dog", "bird"], alert_labels=["dog"])
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), cfg)
    assert analysis.targets == [bird, dog]   # both drawn
    assert analysis.candidates == [dog]      # only the dog can fire
```

- [ ] **Step 7: Full gates.** `uv run pytest -m "not slow" -q` green; `uv run ruff check src tests` clean.

- [ ] **Step 8: Commit** — `feat: watch-for classes (dog/cat/bird) end to end`.

---

### Task 2: Counter inventory (detection + status + dashboard)

**Files:**
- Modify: `src/doggy/vision/detection.py`, `src/doggy/vision/detector.py`, `src/doggy/vision/analysis.py`, `src/doggy/vision/filters/zone.py`, `src/doggy/vision/annotate.py`, `src/doggy/core/config.py`, `src/doggy/core/status.py`, `src/doggy/pipeline.py`, `src/doggy/web/static/index.html`
- Test: `tests/vision/test_analysis.py`, `tests/vision/filters/test_zone.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: Task 1's `target_labels` plumbing.
- Produces: `detection.INVENTORY_LABELS: frozenset[str]`; `FrameAnalysis.inventory: list[Detection]`; `analysis.InventoryTracker` with `update(labels: list[str]) -> list[dict]` (debounced `{"label": str, "count": int}` list) and `.labels() -> set[str]`; `Status.on_counter: list[dict]`; config `inventory_enabled: bool = True`, `inventory_confidence: float = 0.4`, `show_inventory_boxes: bool = False`.

- [ ] **Step 1: Failing tests** in `tests/vision/test_analysis.py`:

```python
def test_analyzer_collects_inventory_separately():
    snack = Detection("sandwich", 0.5, (0, 0, 10, 10))
    dog = Detection("dog", 0.9, (20, 20, 30, 30))
    analyzer = DetectionAnalyzer(StubDetector([[snack, dog]]), FilterChain([]))
    analysis = analyzer.analyze(np.zeros((100, 100, 3), np.uint8), TunableSettings())
    assert analysis.inventory == [snack]
    assert analysis.targets == [dog]


def test_inventory_tracker_debounces_two_of_five():
    t = InventoryTracker()
    assert t.update(["cup"]) == []                # seen once: not yet present
    assert t.update(["cup", "cup"]) == [{"label": "cup", "count": 2}]
    for _ in range(4):
        t.update([])                              # gone 4 frames
    assert t.update([]) == []
```

- [ ] **Step 2: Run** — expected FAIL (no `inventory`, no `InventoryTracker`).

- [ ] **Step 3: Implement.** `src/doggy/vision/detection.py` add:

```python
# Things a counter raider steals or knocks over. Observed, never targeted:
# these classes can appear in the inventory readout and theft forensics but
# cannot fire the deterrent. Fixtures (oven, sink...) are excluded: they
# never move, so they would only be noise.
INVENTORY_LABELS = frozenset({
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
})
```

Config additions (`TunableSettings`):

```python
    # Counter inventory: food/tableware detection for the readout + theft diffs.
    inventory_enabled: bool = True
    # Overhead food shots score low; independent, laxer threshold.
    inventory_confidence: float = Field(0.4, ge=0.0, le=1.0)
    show_inventory_boxes: bool = False
```

`YoloDetector.detect`: predict with `conf=min(cfg.confidence, cfg.inventory_confidence) if cfg.inventory_enabled else cfg.confidence`, then keep a box when either
`label in wanted and float(box.conf[0]) >= cfg.confidence` or
`cfg.inventory_enabled and label in INVENTORY_LABELS and float(box.conf[0]) >= cfg.inventory_confidence`.

`FrameAnalysis`: add `inventory: list[Detection] = dataclasses.field(default_factory=list)`; analyzer fills it (`[d for d in detections if d.label in INVENTORY_LABELS]` when `cfg.inventory_enabled`, else `[]`).

`ZoneInclusionFilter.apply`: after narrowing candidates, also narrow inventory — the zone is what defines "the counter":

```python
        analysis.inventory = self.filter(
            analysis.inventory, cfg.zone_points, analysis.shape)
```

`InventoryTracker` in `src/doggy/vision/analysis.py`:

```python
class InventoryTracker:
    """Debounced presence for inventory labels: an item counts as on the
    counter when seen in at least 2 of the last 5 analyzed frames. Counts are
    the max simultaneous instances seen in those frames (flicker-proof)."""

    WINDOW = 5
    NEEDED = 2

    def __init__(self) -> None:
        self._recent: deque[Counter] = deque(maxlen=self.WINDOW)

    def update(self, labels: list[str]) -> list[dict]:
        self._recent.append(Counter(labels))
        return self._present()

    def labels(self) -> set[str]:
        return {item["label"] for item in self._present()}

    def _present(self) -> list[dict]:
        seen: Counter = Counter()
        appearances: Counter = Counter()
        for frame_counts in self._recent:
            for label, count in frame_counts.items():
                appearances[label] += 1
                seen[label] = max(seen[label], count)
        return [{"label": label, "count": seen[label]}
                for label in sorted(seen)
                if appearances[label] >= self.NEEDED]
```

(`from collections import Counter, deque`.)

`Status`: add `on_counter: list = dataclasses.field(default_factory=list)`.

`Pipeline`: own `self.inventory_tracker = InventoryTracker()`; in `run_once` after `analyze`: `on_counter = self.inventory_tracker.update([d.label for d in analysis.inventory])` and include `on_counter=on_counter` in the big `status.update(...)` call. Pass `inventory=analysis.inventory if cfg.show_inventory_boxes else None` to `annotate`.

`annotate`: new keyword `inventory=None`; draw each with `_INVENTORY_COLOR = (140, 160, 180)  # muted sand BGR` and thickness 1 (add an optional `thickness` parameter to `_draw_box`, default `_BOX_THICKNESS`).

- [ ] **Step 4: Dashboard.** Readout area: add a full-width line under the four stats:

```html
          <div class="oncounter"><span class="k">On the counter</span> <span class="v" id="on_counter">–</span></div>
```

```css
  .oncounter{grid-column:1/-1;font-size:.8rem;color:var(--stone);margin-top:.2rem}
  .oncounter .v{font-family:var(--mono);color:var(--linen);font-size:.8rem}
```

  JS in `poll()`: render `s.on_counter` as `"sandwich, 2 cups"` (`count > 1 ? count + " " + label + "s" : label`, joined; hide/"nothing it recognizes" when empty or `!s.settings.inventory_enabled`). Settings card: a "Show counter items" toggle bound to `show_inventory_boxes` exactly like the `clips_enabled` toggle (label: "Show counter items", desc: "Outline food and dishes it sees on the live view").

- [ ] **Step 5: Pipeline test** in `tests/test_pipeline.py`: stub detector returns `[Detection("cup", 0.5, (10, 10, 20, 20))]` twice; after two `run_once` calls, `status.snapshot().on_counter == [{"label": "cup", "count": 1}]`.

- [ ] **Step 6: Gates** — full suite + ruff.

- [ ] **Step 7: Commit** — `feat: counter inventory detection with debounced readout`.

---

### Task 3: EventStore outcome fields + attach methods

**Files:**
- Modify: `src/doggy/events/store.py`, `src/doggy/web/routers/events.py` (extend `_event_dict`)
- Test: `tests/events/test_store.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `EventRecord` fields `sound: str | None = None`, `clear_seconds: float | None = None`, `strikes: int = 1`, `taken: list[str] = field(default_factory=list)`, `outcome_at: float | None = None` (wall time the outcome landed; None = still unknown). Methods `attach_sound(id, name)`, `attach_outcome(id, clear_seconds, taken, wall_time)`, `bump_strikes(id)`.

- [ ] **Step 1: Failing tests** (`tests/events/test_store.py`):

```python
def test_attach_sound_outcome_and_strikes_persist(tmp_path):
    s = EventStore(tmp_path, 100, 0)
    r = s.add(_img(), 0.9, 1.0, 1000.0, 1.0)
    s.attach_sound(r.id, "chirp.wav")
    s.bump_strikes(r.id)
    s.attach_outcome(r.id, clear_seconds=3.5, taken=["sandwich"], wall_time=1010.0)
    reloaded = EventStore(tmp_path, 100, 0).list()[0]
    assert reloaded.sound == "chirp.wav"
    assert reloaded.strikes == 2
    assert reloaded.clear_seconds == 3.5
    assert reloaded.taken == ["sandwich"]
    assert reloaded.outcome_at == 1010.0


def test_old_jsonl_lines_load_with_outcome_defaults(tmp_path):
    line = {"id": "fire_1", "ts": 1.0, "wall_time": 1000.0, "confidence": 0.9,
            "latency_s": 1.0, "thumb": "fire_1.jpg", "clip": None}
    (tmp_path / "events.jsonl").write_text(json.dumps(line) + "\n")
    r = EventStore(tmp_path, 100, 0).list()[0]
    assert r.sound is None and r.clear_seconds is None
    assert r.strikes == 1 and r.taken == [] and r.outcome_at is None
```

- [ ] **Step 2: Run** — FAIL (unknown fields/methods).

- [ ] **Step 3: Implement.** `EventRecord` gains the five fields (order after `clip`; use `dataclasses.field(default_factory=list)` for `taken`). `_load` fills them with `obj.get("sound")`, `obj.get("clear_seconds")`, `int(obj.get("strikes") or 1)`, `list(obj.get("taken") or [])`, `obj.get("outcome_at")`. New methods, each shaped exactly like `attach_clip` (lock, find by id, mutate, `_rewrite`):

```python
    def attach_sound(self, id: str, sound: str) -> None: ...
    def bump_strikes(self, id: str) -> None:   # record.strikes += 1
    def attach_outcome(self, id: str, clear_seconds: float | None,
                       taken: list[str], wall_time: float) -> None:
        # sets clear_seconds, taken, outcome_at=wall_time
```

In `src/doggy/web/routers/events.py`, extend `_event_dict` to include `sound`, `clear_seconds`, `strikes`, `taken` (read the file first; mirror how `clip` is included).

- [ ] **Step 4: Gates.** Full suite + ruff.

- [ ] **Step 5: Commit** — `feat: event outcome fields (sound, clear time, strikes, taken)`.

---

### Task 4: Sound attribution + alerter volume override

**Files:**
- Modify: `src/doggy/reaction/sound.py`, `src/doggy/app.py`, `src/doggy/web/routers/sounds.py` (only if it calls `alert()` and asserts on the return)
- Test: `tests/reaction/test_sound.py`

**Interfaces:**
- Consumes: Task 3's `attach_sound`.
- Produces: `BaseAlerter.alert(volume: float | None = None) -> str | None` (returns the played clip's filename, None when nothing played; `volume` overrides `cfg.max_volume`, still clamped to [0, 1]). `SoundReaction(alerter, store)` attaches the name to the event. `FakeAlerter.alert(volume=None)` returns `"fake.wav"` and records `self.volumes: list[float | None]`.

- [ ] **Step 1: Failing tests** (`tests/reaction/test_sound.py`):

```python
def test_alert_returns_clip_name_and_honors_override(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    runtime = RuntimeSettings(TunableSettings(clips_dir=tmp_path, max_volume=0.5))
    played = []

    class Probe(BaseAlerter):
        def _play(self, clip, volume):
            played.append((clip.name, volume))

    probe = Probe(runtime)
    assert probe.alert() == "a.wav"
    assert probe.alert(volume=2.0) == "a.wav"   # clamped
    _wait_for(lambda: len(played) == 2)          # _play runs on a daemon thread
    assert played[0] == ("a.wav", 0.5)
    assert played[1] == ("a.wav", 1.0)


def test_sound_reaction_attaches_played_sound(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 1.0)
    fake = FakeAlerter()
    SoundReaction(fake, store).on_dog_caught(_event(r))
    assert store.list()[0].sound == "fake.wav"
```

(Reuse the existing `_wait_for` helper if the file has one; otherwise a 2s poll loop.)

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement.** `BaseAlerter.alert`:

```python
    def alert(self, volume: float | None = None) -> str | None:
        cfg = self._runtime.get()
        clip = self._resolve_clip(cfg)
        if clip is None:
            return None
        level = max(0.0, min(1.0, cfg.max_volume if volume is None else volume))
        threading.Thread(target=self._play, args=(clip, level), daemon=True).start()
        return clip.name
```

`SoundReaction`:

```python
class SoundReaction:
    """Reaction (Observer): plays the deterrent and records which clip it was."""

    def __init__(self, alerter: Alerter, store: EventStore) -> None:
        self._alerter = alerter
        self._store = store

    def on_dog_caught(self, event) -> None:
        name = self._alerter.alert()
        if name:
            self._store.attach_sound(event.record.id, name)
```

Update the `Alerter` Protocol signature to match; `FakeAlerter.alert(self, volume=None)` appends `volume` to `self.volumes`, increments `calls`, returns `"fake.wav"`. Wire `SoundReaction(alerter, event_store)` in `src/doggy/app.py`. Check `src/doggy/web/routers/sounds.py` `test-sound` route still typechecks (it may ignore the return; leave behavior identical).

- [ ] **Step 4: Gates, then commit** — `feat: alerter reports played clip; events record their sound`.

---

### Task 5: OutcomeWatcher (clear-time measurement + theft diff)

**Files:**
- Create: `src/doggy/reaction/outcome.py`
- Modify: `src/doggy/pipeline.py`, `src/doggy/app.py`
- Test: `tests/reaction/test_outcome.py` (new)

**Interfaces:**
- Consumes: `EventStore.attach_outcome` (T3), `FrameAnalysis.candidates`/`.inventory` (T1/T2), `InventoryTracker` (T2), `DogCaught` hub event.
- Produces: `OutcomeWatcher(store, gate, alerter, runtime, clock=time.time)` with `on_dog_caught(event)` (hub Reaction) and `on_frame(analysis, now, cfg)` (pipeline per-frame stage, monotonic `now`). Constants `CLEAR_DEBOUNCE_SECONDS = 2.0`, `MAX_WATCH_SECONDS = 60.0`. Escalation hooks land in Task 6; this task measures only.

- [ ] **Step 1: Failing tests** (`tests/reaction/test_outcome.py`) — drive with a fake monotonic clock and hand-built `FrameAnalysis` values:

```python
def _analysis(candidates=(), inventory=()):
    return FrameAnalysis(shape=(100, 100, 3), people=[],
                         targets=list(candidates), candidates=list(candidates),
                         inventory=list(inventory))

def test_clear_time_measured_after_debounce(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    w = OutcomeWatcher(store, _gate(), FakeAlerter(), _runtime(), clock=lambda: 2000.0)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 11.0, _cfg())      # still there
    w.on_frame(_analysis([]), 14.0, _cfg())          # gone at 14.0
    w.on_frame(_analysis([]), 15.5, _cfg())          # 1.5s clear: not yet
    assert store.list()[0].outcome_at is None
    w.on_frame(_analysis([]), 16.1, _cfg())          # 2.1s clear: finalize
    rec = store.list()[0]
    assert rec.clear_seconds == pytest.approx(4.0)   # 14.0 - 10.0
    assert rec.outcome_at == 2000.0

def test_timeout_records_not_deterred(tmp_path):
    ...  # occupied frames past 60s -> clear_seconds None, outcome_at set

def test_taken_is_inventory_diff(tmp_path):
    ...  # sandwich present (2 frames) before fire, absent after clear -> taken == ["sandwich"]
```

Write all three fully (the second: frames at 11.0 and 71.0 with the dog present; the third: seed two pre-fire frames with `inventory=[sandwich]` via `on_frame` before `on_dog_caught`, then clear frames with empty inventory).

- [ ] **Step 2: Run** — FAIL (module missing).

- [ ] **Step 3: Implement** `src/doggy/reaction/outcome.py`:

```python
from __future__ import annotations

import logging
import time
from typing import Callable

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.decision.gate import FireGate
from doggy.events.store import EventStore
from doggy.reaction.hub import DogCaught
from doggy.reaction.sound import Alerter
from doggy.vision.analysis import FrameAnalysis, InventoryTracker

log = logging.getLogger("doggy")

# An incident is "cleared" after the zone stays target-free this long; the
# debounce absorbs detection flicker and is subtracted from the measurement.
CLEAR_DEBOUNCE_SECONDS = 2.0
# Give up measuring after this long: the target never left (not deterred).
MAX_WATCH_SECONDS = 60.0


class OutcomeWatcher:
    """Measures what happened after each fire: how long until the target left
    the zone, and which counter items disappeared during the incident.

    Per-frame stage + hub Reaction, like ClipService. Single-threaded: both
    entry points run on the pipeline thread. `clock` is wall time and is only
    stamped onto the stored outcome; measurements use pipeline monotonic time.
    """

    def __init__(self, store: EventStore, gate: FireGate, alerter: Alerter,
                 runtime: RuntimeSettings,
                 clock: Callable[[], float] = time.time) -> None:
        self._store = store
        self._gate = gate
        self._alerter = alerter
        self._runtime = runtime
        self._clock = clock
        self._tracker = InventoryTracker()
        self._incident: dict | None = None

    def on_dog_caught(self, event: DogCaught) -> None:
        if self._incident is not None:
            # A new confirmed fire while still watching: the old incident
            # never cleared. Close it honestly before tracking the new one.
            self._finalize(clear_seconds=None)
        self._incident = {
            "id": event.record.id,
            "fire_ts": event.mono_ts,
            "last_strike_ts": event.mono_ts,
            "strikes": 1,
            "clear_since": None,
            "before": self._tracker.labels(),
        }

    def on_frame(self, analysis: FrameAnalysis, now: float,
                 cfg: TunableSettings) -> None:
        self._tracker.update([d.label for d in analysis.inventory])
        if self._incident is None:
            return
        inc = self._incident
        if analysis.candidates:
            inc["clear_since"] = None
            if now - inc["fire_ts"] >= MAX_WATCH_SECONDS:
                self._finalize(clear_seconds=None)
            return
        if inc["clear_since"] is None:
            inc["clear_since"] = now
        if now - inc["clear_since"] >= CLEAR_DEBOUNCE_SECONDS:
            self._finalize(clear_seconds=inc["clear_since"] - inc["fire_ts"])

    def _finalize(self, clear_seconds: float | None) -> None:
        inc, self._incident = self._incident, None
        taken = sorted(inc["before"] - self._tracker.labels())
        self._store.attach_outcome(
            inc["id"], clear_seconds=clear_seconds, taken=taken,
            wall_time=self._clock())
```

- [ ] **Step 4: Wire it.** `app.py`: build `outcome = OutcomeWatcher(event_store, gate, alerter, runtime)` and register `SafeReaction(outcome)` in the hub (after the clip service); pass `outcome=outcome` to `Pipeline`. `pipeline.py`: accept `outcome` in `__init__`; in `run_once` call `self.outcome.on_frame(analysis, now, cfg)` immediately after the fire-handling block (so the fire frame that starts an incident is also its first observation). Update pipeline tests' constructor calls (give them a real `OutcomeWatcher` over the test store).

- [ ] **Step 5: Gates, then commit** — `feat: outcome watcher measures clear time and thefts per catch`.

---

### Task 6: Escalation ladder

**Files:**
- Modify: `src/doggy/core/config.py`, `src/doggy/decision/gate.py`, `src/doggy/reaction/outcome.py`
- Test: `tests/decision/test_gate.py`, `tests/reaction/test_outcome.py`

**Interfaces:**
- Consumes: T5's incident state, T4's `alert(volume=...)`, T3's `bump_strikes`.
- Produces: config `escalation_enabled: bool = False`, `escalation_seconds: float = 8.0 (ge=1)`, `escalation_max_strikes: int = 3 (ge=1)`, `escalation_volume_step: float = 0.2 (ge=0, le=1)`; `FireGate.allow_escalation(now)` = `allow` minus the cooldown concern (the gate has no cooldown; identical checks today, distinct method so the schedule task in Plan 2 can diverge them), strikes call `note_fire`.

- [ ] **Step 1: Failing tests.** Gate: `allow_escalation` respects `safety_enabled`, snooze, and the hourly cap (three asserts mirroring the existing `allow` tests). Watcher (extend `test_outcome.py`):

```python
def test_escalates_while_occupied_then_stops_at_max(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 10.0)
    fake = FakeAlerter()
    cfg = TunableSettings(escalation_enabled=True, escalation_seconds=8,
                          escalation_max_strikes=3, escalation_volume_step=0.2,
                          max_volume=0.5)
    runtime = RuntimeSettings(cfg)
    w = OutcomeWatcher(store, FireGate(runtime), fake, runtime)
    dog = Detection("dog", 0.9, (0, 0, 10, 10))
    w.on_dog_caught(DogCaught(r, _img(), 10.0))
    w.on_frame(_analysis([dog]), 17.9, cfg)
    assert fake.calls == 0                      # not yet 8s since strike 1
    w.on_frame(_analysis([dog]), 18.1, cfg)     # strike 2
    w.on_frame(_analysis([dog]), 26.2, cfg)     # strike 3
    w.on_frame(_analysis([dog]), 40.0, cfg)     # max reached: no strike 4
    assert fake.calls == 2
    assert fake.volumes == [pytest.approx(0.7), pytest.approx(0.9)]
    assert store.list()[0].strikes == 3
```

- [ ] **Step 2: Run** — FAIL.

- [ ] **Step 3: Implement.** Config fields with the `Field` bounds above (comment: "Escalation: fire again, louder, while the animal stands its ground."). Gate:

```python
    def allow_escalation(self, now: float) -> bool:
        """Follow-up strikes in one incident: no cooldown between them, but the
        master switch, snooze, and the hourly cap still apply."""
        cfg = self._runtime.get()
        if not cfg.safety_enabled:
            return False
        if now < self._snooze_until:
            return False
        return self.fires_last_hour(now) < cfg.max_fires_per_hour
```

(Refactor `allow` to call `allow_escalation` plus nothing else today — keep them textually separate methods so Plan 2's schedule lands in both.) Watcher: inside `on_frame`'s occupied branch, before the timeout check:

```python
            if (cfg.escalation_enabled
                    and inc["strikes"] < cfg.escalation_max_strikes
                    and now - inc["last_strike_ts"] >= cfg.escalation_seconds
                    and self._gate.allow_escalation(now)):
                level = min(1.0, cfg.max_volume
                            + inc["strikes"] * cfg.escalation_volume_step)
                if self._alerter.alert(volume=level):
                    self._gate.note_fire(now)
                    self._store.bump_strikes(inc["id"])
                    inc["strikes"] += 1
                    inc["last_strike_ts"] = now
```

- [ ] **Step 4: Dashboard knob.** Settings "Advanced" section: an "Escalate while it stays" toggle (`escalation_enabled`) with desc "Play again, louder, if the animal doesn't leave after the first sound." (Sliders for the three numeric knobs are deliberately NOT added — defaults are fine; the .env covers tuning. YAGNI.)

- [ ] **Step 5: Gates, then commit** — `feat: escalation ladder (repeat louder while the target stays)`.

---

### Task 7: Deterrence Lab endpoint + dashboard card + catch-log detail

**Files:**
- Modify: `src/doggy/events/store.py` (add `lab_stats()`), `src/doggy/web/routers/events.py` (add `GET /api/lab`), `src/doggy/web/static/index.html`
- Test: `tests/events/test_store.py`, `tests/web/test_api.py`

**Interfaces:**
- Consumes: T3 fields.
- Produces: `EventStore.lab_stats() -> dict`:

```python
{
  "sounds": [{"sound": "chirp.wav", "plays": 9, "completed": 8,
              "deterred_rate": 0.75, "avg_clear_s": 4.2,
              "wearing_off": false}, ...],   # sorted by plays desc
  "thefts_this_week": 2,
}
```

Definitions: `plays` = events with this `sound`; `completed` = of those, `outcome_at` set; `deterred` = completed AND `clear_seconds is not None and clear_seconds <= 15 and not taken`; `deterred_rate` over completed (None when completed == 0); `avg_clear_s` over non-None clears (None if none); `wearing_off` = completed >= 6 and second-half effective-clear average >= 1.5x first-half (effective clear = `clear_seconds` or 60.0 when None; halves split the completed events in time order). `thefts_this_week` counts `len(taken)` summed over events with `wall_time` in the last 7 days.

- [ ] **Step 1: Failing store test** with hand-crafted records: one sound with clears `[2.0, 3.0, 20.0, None, 30.0, None]` -> `completed 6`, `deterred_rate == pytest.approx(2/6)`, `wearing_off is True`; one taken-event this week -> `thefts_this_week == 1`; an event with sound but no outcome counts in `plays` not `completed`.

- [ ] **Step 2: Run** — FAIL. **Step 3:** implement `lab_stats()` (lock, snapshot records, pure computation; time from `self._clock()`); route in `events.py` router: `@router.get("/api/lab")` returning `store.lab_stats()`. Web test: seed the store, `GET /api/lab`, assert shape.

- [ ] **Step 4: Dashboard.** New card between Activity and System (`c-lab`, mobile `order:4`, bump settings/system to 5/6):

```html
      <div class="card c-lab">
        <h2>Deterrence</h2>
        <div id="lab_body"><div class="empty">No data yet. Once it reacts a few times, you'll see which sounds actually work.</div></div>
      </div>
```

  JS `refreshLab()` on the 4s poll: fetch `/api/lab`; render a table (reuse mono/`--stone` styling):
  sound name (strip extension), plays, avg escape (`secs()` or "stayed"), deterred % — plus a `wearing off` tag (`.tag{color:var(--ember);border:1px solid var(--ember);border-radius:4px;padding:0 .35rem;font-size:.65rem}`) and a thefts line ("2 items lost this week") when > 0.
  Catch-log rows (`evRow`): append to `ev-sub`: `" · " + e.strikes + " strikes"` when `e.strikes > 1`, and `" · took the " + e.taken.join(", ")` when `e.taken.length`.

- [ ] **Step 5: Gates, then commit** — `feat: deterrence lab (per-sound effectiveness) endpoint and card`.

---

### Task 8: Weekly report card

**Files:**
- Modify: `src/doggy/events/store.py` (extend `stats()`), `src/doggy/web/static/index.html`
- Test: `tests/events/test_store.py`

**Interfaces:**
- Consumes: T3 fields.
- Produces: `stats()["report_card"] = {"grade": "B+", "attempts": 11, "attempts_prev": 15, "deterred_rate": 1.0, "summary": "11 attempts, all deterred, down from 15 last week."}` (`deterred_rate` None when no completed outcomes; grade per the spec rubric).

- [ ] **Step 1: Failing tests.** Three cases: no events at all -> grade "A", summary "A quiet week."; attempts rose with weak deterrence (e.g. 8 now vs 3 prev, half deterred) -> grade in D/F range; attempts fell, all deterred -> A/B range. Compute expected letters by hand from the rubric and assert exactly.

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** in `stats()`: this-week = `wall_time` within 7 days of now, prev-week = 7-14 days; score = `100 - min(40, 5 * attempts)`; `-30` if `attempts > attempts_prev`, `+10` if `attempts < attempts_prev`; if any completed outcomes this week, `score *= deterred_rate` (deterred = same definition as Task 7); clamp to [0, 100]. Bands: >=90 A, >=80 B, >=65 C, >=50 D, else F; within a band, top third gets "+", bottom third "-" (A+ allowed, F bare). Summary assembled server-side in plain words: "<n> attempts" / "all deterred" / "<k> of <n> deterred" / "up from <p> last week" / "down from <p> last week" / "A quiet week." for zero events.

- [ ] **Step 4: Dashboard.** Activity card, under the bars: `<div class="act-note" id="report_card"></div>`; `refreshStats()` sets `"This week: " + rc.grade + ". " + rc.summary` (hide while `rc` absent).

- [ ] **Step 5: Gates, then commit** — `feat: weekly report card in activity stats`.

---

## Deploy note (after all tasks)

Code-only change: `rsync` the `src/` tree + `index.html` to the Pi and `sudo systemctl restart doggy` (no `uv sync`; see `updating-a-firewalled-uv-python-appliance`). Verify: dashboard loads, `/api/lab` returns JSON, live view still streams, and a `Test sound` fires.

## Self-review notes

- Type consistency checked: `attach_outcome(id, clear_seconds, taken, wall_time)` is used with those exact kwargs in T5/T7 tests; `alert(volume=...) -> str | None` matches T4/T6; `InventoryTracker.update -> list[dict]` matches T2's pipeline/status usage and T5 uses `labels() -> set[str]`.
- The `dogs -> targets` rename is confined to Task 1 and gated by the full suite; later tasks only use the new names.
- No task depends on a Plan 2 feature; `allow_escalation` is written so Plan 2's schedule check can slot into both gate methods.
