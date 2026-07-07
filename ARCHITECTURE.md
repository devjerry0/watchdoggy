# Architecture

watchdoggy is a small (~1,700 line) always-on appliance that watches one frame at a
time: **see** it (vision), **decide** whether it is a fire (decision), and **react**
(reaction). The code is organized into domain packages along those lines. This page maps
each package to its job, names the design patterns that earn their keep and why they are
here, and records the invariants that must not drift.

## Domain map

```
src/doggy/
  main.py              entry-point shim: `from doggy.app import main` (never move -- see Invariants)
  app.py               composition root: builds and wires every object, then runs the loop
  pipeline.py          Facade: orchestrates one detect cycle over the wired collaborators
  core/                cross-cutting primitives
    config.py          Settings (pydantic): `DOGGY_*` env vars read once at startup
    runtime.py         RuntimeSettings: thread-safe live-tunable settings, swapped atomically
    status.py          Status/StatusStore snapshot + FrameBuffer (latest-frame slot)
    pacer.py           Pacer: sleeps only the time still needed to hit a target interval
  vision/              see: a frame -> detections -> a narrowed candidate set
    detection.py       Detection dataclass + ANIMAL_TARGETS ("dog","cat","bird") / PERSON_LABEL
    camera.py          Camera protocol + OpenCV/file backends + registry factory
    detector.py        Detector protocol + YOLO/stub backends + plain factory
    analysis.py        FrameAnalysis + DetectionAnalyzer (runs the detector, then the chain)
    annotate.py        draws boxes and the watch-area onto a frame for the dashboard
    filters/           Chain of Responsibility links over FrameAnalysis
      base.py          DetectionFilter protocol + FilterChain
      person.py        suppress targets that are really people (IoU coincidence)
      zone.py          keep only candidates inside the drawn watch area
  decision/            decide: is this a fire, and is a fire allowed right now
    trigger.py         TriggerLogic FSM: M-of-N window + confirm timer + jittered cooldown
    gate.py            FireGate: master switch, snooze, per-hour cap
  reaction/            react: what happens on a confirmed catch
    hub.py             DogCaught event + Reaction protocol + ReactionHub + SafeReaction
    recorder.py        Recorder: persists a catch via the single-writer EventStore
    sound.py           alerter backends (Template Method) + registry + SoundReaction
    clips.py           ClipBuffer + ClipService (per-frame capture and a catch reaction)
    outcome.py         OutcomeWatcher: clear-time + theft measurement per catch, drives escalation (per-frame stage + hub reaction, like ClipService)
  events/
    store.py           EventStore/EventRecord: the only writer of events.jsonl + JPEGs
  hardware/            the Pi's physical signals
    thermal.py         ThermalGovernor: CPU temperature -> detect interval
    power.py           PowerMonitor: cached under-voltage flags from vcgencmd
  web/                 the LAN dashboard (FastAPI)
    app.py             create_app + serve + GET /
    envfile.py         in-place .env writer for settings saved from the dashboard
    routers/           one router per endpoint group: status, settings, events, sounds, snooze
    static/index.html  the single-page dashboard
```

## Applied patterns

Introduced by the restructure, each because a concrete pressure in this codebase called
for it -- not for symmetry.

- **Observer -- `reaction/hub.py`.** A catch fans out to several independent effects (play
  a sound, save a clip, update status). The hub lets `pipeline.py` publish one `DogCaught`
  without knowing who reacts, and the set of reactions is chosen in one place (`app.py`).
  Registration order is the fan-out order.
- **Decorator -- `SafeReaction` (`reaction/hub.py`).** One misbehaving reaction must never
  take down the detect loop or the other reactions. `SafeReaction` wraps each reaction with
  a log-and-swallow guard, so crash isolation is a single wrapper the composition root
  applies uniformly, instead of try/except smeared through the hub and every reaction.
- **Chain of Responsibility + Adapter -- `vision/filters/`.** Detection narrowing is a
  sequence of independent steps (person suppression, then zone inclusion) that each mutate
  the shared `FrameAnalysis` in place and decide for themselves whether they apply. A new
  rule slots in as another link. The former `people`/`zone` free functions are wrapped as
  `DetectionFilter` links (Adapter) so the chain sees one uniform `apply` interface.
- **Strategy registries -- `vision/camera.py` and `reaction/sound.py`.** The camera backend
  (opencv/file) and audio backend (sounddevice/command/log) are chosen at startup from a
  config string via a small name-to-factory dict. A dict lookup beats an if/elif ladder,
  keeps backend choice declarative, and lets unknown values fall back to a documented default.
- **State -- `decision/trigger.py`.** The trigger is a three-node FSM (Idle -> Confirming ->
  Cooldown). Each state is a stateless singleton whose `handle` returns the next state, so
  the transition rules live next to the state they belong to rather than in one big branch
  on an enum. All mutable data stays on the owning `TriggerLogic`.
- **Template Method -- `reaction/sound.py`.** Every alerter shares one skeleton: read config,
  resolve the clip, clamp the volume, hand off to a background thread. `BaseAlerter.alert()`
  owns that skeleton and calls a `_play` hook; each backend implements only `_play`. This
  removes the duplication three near-identical `alert()` methods would carry.
- **Facade -- `pipeline.py`.** `Pipeline` is the one object that runs a detect cycle:
  analyze -> annotate -> trigger -> gate -> record -> publish -> update status. It sequences
  the collaborators but holds no detection, filtering, or reaction logic of its own, so
  `run_once` reads as the high-level story of a single frame.
- **Composition root -- `app.py`.** All construction and wiring happens once, in `main()`.
  Every other module takes its collaborators as constructor arguments and never builds them.
  That is what keeps the object graph in one readable place and the whole tree testable with
  fakes.

## Patterns already present

Named here because they predate the restructure and still carry their weight.

- **Caching Proxy -- `hardware/power.py`.** `PowerMonitor.read()` looks like a plain read
  but only actually shells out to `vcgencmd` once per interval, serving a cached
  `PowerStatus` in between, so calling it every loop is cheap.
- **Iterator -- `vision/camera.py`.** `Camera.frames()` yields frames one at a time behind a
  uniform interface, so a real webcam and a canned test video are consumed by identical loop
  code.
- **Null Object -- `LogAlerter` (`reaction/sound.py`).** A do-nothing audio backend that
  still resolves a clip but plays nothing, so headless and dev runs need no
  `if alerter is None` guards anywhere.
- **Single-slot drop-oldest buffer -- `core/status.py` `FrameBuffer`.** Holds only the most
  recent frame; a new `set` overwrites the old one. The capture thread and the detect loop
  stay decoupled and the loop always works on the freshest frame instead of a backlog.

## Rejected on purpose

At this size these would be Speculative Generality -- structure justified only by symmetry,
not by a real second case.

- **Mediator** -- the hub already decouples the publisher from its reactions; a mediator on
  top would only move the same wiring behind another indirection.
- **Builder / Abstract Factory** -- `app.py` builds each object once with plain
  constructors; there are no product families or multi-step assembly to justify one.
- **Command** -- a reaction is a single method (`on_dog_caught`); there is no queue, undo, or
  replay that would make reified command objects earn their keep.
- **Hexagonal ports/adapters** -- every "port" here has exactly one real implementation plus
  a test fake, which a Protocol already covers without a ports-and-adapters layer.
- **Singleton** -- shared objects are created once in the composition root and passed in; a
  Singleton would only add global state and hurt testability.

## Invariants

- **`main.py` must never move or be renamed.** The Pi's installed console script resolves
  `doggy.main:main`, and the service starts with `uv run --no-sync` (no re-install on
  deploy), so the module name is entry-point metadata. It stays a two-line shim over
  `doggy.app:main`.
- **Single event writer.** `EventStore` (`events/store.py`) is the only writer of
  `events.jsonl` and the JPEG/clip files. Every catch goes through `Recorder` -> `EventStore`,
  and the web layer only reads that directory. Nothing else writes there.
- **`run_once` reads the clock exactly once.** One `now = self.clock()` per frame feeds the
  trigger, the gate, and the recorded event, so the whole cycle shares a single consistent
  timestamp (and tests can inject a fixed clock).
- **Frozen external contract.** The `DOGGY_*` env var names, the HTTP route paths and JSON
  shapes, the `.env` round-trip format, and the `events.jsonl` / `events/` layout are the
  appliance's public interface. The restructure changed none of them; a rename here would
  break a deployed Pi.
