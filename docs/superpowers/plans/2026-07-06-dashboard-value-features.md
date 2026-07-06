# Dashboard Value Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent event history (real timestamps, time-to-react, per-event delete, retention), snooze, sound picker + volume, activity stats, and opt-in video clips to the watchdoggy dashboard.

**Architecture:** A new disk-backed `EventStore` (`src/doggy/events.py`) becomes the source of truth for reactions (JPEG + `events.jsonl` + optional clip), replacing the in-memory-only deque. `SafetyGovernor` delegates persistence to it. A `ClipBuffer` (`src/doggy/clips.py`) holds recent JPEG frames in memory so clips are derived on a catch with pre-roll and no continuous SD writes. Stats and clips build on `EventStore`.

**Tech Stack:** Python 3.11, FastAPI, pydantic-settings, OpenCV, Pillow (clip fallback), systemd-timesyncd (LAN NTP), uv/pytest.

## Global Constraints

- Fully local, no internet: no outbound calls; clock syncs from the LAN gateway; UI falls back to relative time if unsynced. Do not modify `harden-pi.sh`'s egress rules.
- SD-wear conscious: no new continuous writes; every store prunes; clips opt-in + separately capped.
- Backward compatible: old `events.jsonl` lines missing new fields must load with defaults (`wall_time=None`, `latency_s=None`, `id`=thumb stem, `clip=None`).
- Conventions: `DOGGY_` env prefix; frozen `TunableSettings`; tests as `tests/test_*.py`; run fast suite with `uv run pytest -m "not slow"`; no emoji in UI; plain-language labels.
- Every task ends green on `uv run pytest -m "not slow"` and `uv run ruff check` on its new/changed files.

## File Structure

- `src/doggy/events.py` (NEW) — `EventRecord` dataclass + `EventStore` (load, add, list, delete, clear, prune, stats, attach_clip). Owns the events dir files.
- `src/doggy/clips.py` (NEW) — `ClipBuffer` (rolling JPEG window) + `encode_clip()` (mp4 with animated-fallback).
- `src/doggy/config.py` — new tunables (retention, snooze default, selected_sound, clip settings).
- `src/doggy/trigger.py` — add `fire_latency`.
- `src/doggy/safety.py` — delegate persistence to `EventStore`; add snooze gating.
- `src/doggy/pipeline.py` — pass `wall_time`/`latency`; feed `ClipBuffer`; finalize pending clips.
- `src/doggy/state.py` — `Status` gains `snoozed_until_seconds`; keep event access via `EventStore`.
- `src/doggy/web.py` — events/stats/snooze/sounds/clips endpoints.
- `src/doggy/alerter.py` — honor `selected_sound` + volume.
- `src/doggy/static/index.html` — history list, snooze, stats card, sound picker, clip toggle/playback.
- `scripts/deploy-to-pi.sh` — write `timesyncd.conf` pointing at the gateway.
- Tests: `tests/test_events.py`, `tests/test_clips.py`, plus additions to `test_trigger.py`, `test_safety.py`, `test_pipeline.py`, `test_web.py`, `test_alerter.py`.

---

### Task 1: EventStore core — schema, load (back-compat), add, prune

**Files:**
- Create: `src/doggy/events.py`
- Modify: `src/doggy/config.py` (add two fields)
- Test: `tests/test_events.py`

**Interfaces:**
- Produces:
  - `@dataclass EventRecord: id:str; ts:float; wall_time:float|None; confidence:float; latency_s:float|None; thumb:str; clip:str|None=None`
  - `EventStore(event_dir:Path, max_events:int=500, max_age_days:int=30, clock:Callable[[],float]=time.time)`
  - `EventStore.add(frame:np.ndarray, confidence:float, latency_s:float|None, wall_time:float|None, mono_ts:float) -> EventRecord`
  - `EventStore.list(limit:int|None=None) -> list[EventRecord]` (most-recent-first)
  - `EventStore.prune() -> None`
- Consumes: nothing (foundation).

**Config additions** (`TunableSettings`, after `thermal_cooldown_interval_seconds`):
```python
    event_retention_max: int = Field(500, ge=0)   # 0 = unlimited
    event_retention_days: int = Field(30, ge=0)    # 0 = no age limit
```

- [ ] **Step 1: Write failing tests** (`tests/test_events.py`)
```python
import json, numpy as np
from pathlib import Path
from doggy.events import EventStore, EventRecord

def _img(): return np.zeros((8, 8, 3), np.uint8)

def test_add_writes_jpeg_and_jsonl(tmp_path):
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    r = s.add(_img(), confidence=0.8, latency_s=1.5, wall_time=1000.0, mono_ts=5.0)
    assert (tmp_path / r.thumb).is_file()
    lines = (tmp_path / "events.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["confidence"] == 0.8
    assert r.latency_s == 1.5 and r.wall_time == 1000.0 and r.id

def test_list_is_recent_first(tmp_path):
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    a = s.add(_img(), 0.5, None, 1.0, 1.0); b = s.add(_img(), 0.6, None, 2.0, 2.0)
    assert [e.id for e in s.list()] == [b.id, a.id]

def test_loads_existing_and_old_schema(tmp_path):
    # old line missing id/wall_time/latency_s/clip must still load
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"ts": 3.0, "confidence": 0.7, "thumb": "old.jpg"}) + "\n")
    s = EventStore(tmp_path, max_events=10, max_age_days=0)
    e = s.list()[0]
    assert e.thumb == "old.jpg" and e.wall_time is None and e.latency_s is None
    assert e.id == "old"  # stem of thumb

def test_prune_by_count_deletes_files(tmp_path):
    s = EventStore(tmp_path, max_events=2, max_age_days=0)
    r0 = s.add(_img(), 0.5, None, 1.0, 1.0)
    s.add(_img(), 0.5, None, 2.0, 2.0); s.add(_img(), 0.5, None, 3.0, 3.0)
    assert len(s.list()) == 2
    assert not (tmp_path / r0.thumb).exists()   # oldest file removed

def test_prune_by_age(tmp_path):
    s = EventStore(tmp_path, max_events=100, max_age_days=1, clock=lambda: 1_000_000.0)
    old = s.add(_img(), 0.5, None, 1_000_000.0 - 2*86400, 1.0)  # 2 days old by wall_time
    fresh = s.add(_img(), 0.5, None, 1_000_000.0, 2.0)
    s.prune()
    ids = [e.id for e in s.list()]
    assert fresh.id in ids and old.id not in ids
```

- [ ] **Step 2: Run to verify FAIL** — `uv run pytest tests/test_events.py -q` (import error / not defined).

- [ ] **Step 3: Implement `EventStore`.** Key rules:
  - `id = f"fire_{int(round((wall_time or mono_ts) * 1000))}"`; jpeg name `f"{id}.jpg"`, written with `cv2.imwrite`.
  - Load: parse each jsonl line; `id = obj.get("id") or Path(obj["thumb"]).stem`; `wall_time/latency_s/clip = obj.get(...)`. Keep in a list ordered oldest→newest; `list()` returns reversed, sliced to `limit`.
  - `add`: build record, write jpeg, append one json line, append to memory, `prune()`, return.
  - `prune`: drop by age first (only for records with a real `wall_time`, when `max_age_days>0`: `now - wall_time > max_age_days*86400` using `self._clock()`), then by count (keep newest `max_events` when `max_events>0`); for each dropped record delete its `thumb` and `clip` files (guard `is_file`) and rewrite `events.jsonl` from the surviving records.
  - JSON serialization: write all fields including `id`, `wall_time`, `latency_s`, `clip`.

- [ ] **Step 4: Run to verify PASS** — `uv run pytest tests/test_events.py -q`.

- [ ] **Step 5: Commit** — `git add src/doggy/events.py src/doggy/config.py tests/test_events.py && git commit -m "feat: EventStore core (persistent history + retention)"`

---

### Task 2: EventStore delete, clear, stats, attach_clip

**Files:**
- Modify: `src/doggy/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Produces:
  - `EventStore.delete(id:str) -> bool`
  - `EventStore.clear() -> None`
  - `EventStore.attach_clip(id:str, clip_name:str) -> None`
  - `EventStore.stats() -> dict` with keys `today:int, this_week:int, per_day:list[dict{day:str,count:int}]` (last 7 days, oldest→newest), `busiest_hour:int|None`, `avg_latency_s:float|None`
- Consumes: Task 1 internals.

- [ ] **Step 1: Write failing tests**
```python
def test_delete_removes_record_and_file(tmp_path):
    s = EventStore(tmp_path, 10, 0)
    r = s.add(_img(), 0.5, None, 1.0, 1.0)
    assert s.delete(r.id) is True
    assert s.list() == [] and not (tmp_path / r.thumb).exists()
    assert s.delete("nope") is False

def test_clear_removes_all(tmp_path):
    s = EventStore(tmp_path, 10, 0)
    s.add(_img(), 0.5, None, 1.0, 1.0); s.add(_img(), 0.5, None, 2.0, 2.0)
    s.clear()
    assert s.list() == []
    assert (tmp_path / "events.jsonl").read_text() == ""

def test_stats_counts_and_latency(tmp_path):
    # fixed "now" = 2026-07-06 18:00 UTC; two events today, one 3 days ago
    now = 1783360800.0
    s = EventStore(tmp_path, 100, 0, clock=lambda: now)
    s.add(_img(), 0.5, 1.0, now - 3600, 1.0)          # today, 1h ago
    s.add(_img(), 0.5, 2.0, now - 7200, 2.0)          # today, 2h ago
    s.add(_img(), 0.5, 3.0, now - 3*86400, 3.0)       # 3 days ago
    st = s.stats()
    assert st["today"] == 2 and st["this_week"] == 3
    assert abs(st["avg_latency_s"] - 2.0) < 1e-9
    assert isinstance(st["busiest_hour"], int)
    assert len(st["per_day"]) == 7

def test_stats_busiest_hour_none_without_wall_time(tmp_path):
    s = EventStore(tmp_path, 100, 0)
    s.add(_img(), 0.5, 1.0, None, 1.0)
    assert s.stats()["busiest_hour"] is None
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement.**
  - `delete`: find by id; if absent return False; delete `thumb`+`clip` files; remove from memory; rewrite jsonl; return True.
  - `clear`: delete every record's files; empty memory; write empty `events.jsonl`.
  - `attach_clip`: set `clip` on the record, rewrite jsonl.
  - `stats`: use `datetime.fromtimestamp(wall_time)` (records with wall_time only) for `today`/`this_week` (calendar day boundaries via `self._clock()`), `per_day` for the last 7 calendar days (label `YYYY-MM-DD`), `busiest_hour` = mode of `.hour` (None if no wall_time records), `avg_latency_s` = mean of non-None `latency_s` (None if none).

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit** — `git commit -am "feat: EventStore delete/clear/stats/attach_clip"`

---

### Task 3: Trigger latency + integrate EventStore into the pipeline

**Files:**
- Modify: `src/doggy/trigger.py`, `src/doggy/safety.py`, `src/doggy/pipeline.py`, `src/doggy/state.py`
- Test: `tests/test_trigger.py`, `tests/test_safety.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `EventStore` (Task 1/2).
- Produces:
  - `TriggerLogic.fire_latency: float` (set on the fire edge = `now - self._confirm_start`).
  - `SafetyGovernor(runtime, event_store: EventStore)` — `record_fire(frame, confidence, latency_s, wall_time, now) -> EventRecord` delegates to `event_store.add(...)`; keeps the rate deque.
  - Pipeline builds `EventStore` and passes it to `SafetyGovernor`; on fire calls `record_fire(frame, self.trigger.fire_confidence, self.trigger.fire_latency, time.time(), now)`.

- [ ] **Step 1: Write failing tests**
```python
# test_trigger.py
def test_fire_latency_is_time_since_first_sighting():
    t = make()  # window_m=2,n=3,confirm=1.0
    d = [Detection("dog", 0.9, (0,0,10,10))]
    t.update(d, now=0.0); t.update(d, now=0.5)
    assert t.update(d, now=1.0) is True
    assert t.fire_latency == 1.0

# test_safety.py
def test_record_fire_delegates_to_store(tmp_path):
    from doggy.events import EventStore
    from doggy.config import Settings
    from doggy.state import RuntimeSettings
    import numpy as np
    store = EventStore(tmp_path, 10, 0)
    gov = SafetyGovernor(RuntimeSettings(Settings().tunable()), store)
    ev = gov.record_fire(np.zeros((8,8,3), np.uint8), confidence=0.7, latency_s=1.2,
                         wall_time=1000.0, now=5.0)
    assert ev.latency_s == 1.2 and len(store.list()) == 1
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement.**
  - `trigger.py`: add `self.fire_latency: float = 0.0` in `__init__`; on the fire edge (where `fire_confidence` is set) add `self.fire_latency = now - self._confirm_start`.
  - `safety.py`: constructor takes `event_store`; remove direct file writing; `record_fire(frame, confidence, latency_s, wall_time, now)` appends `now` to the rate deque and returns `event_store.add(frame, confidence, latency_s, wall_time, mono_ts=now)`.
  - `pipeline.py`: in `__init__` accept/build the `EventStore` (from `settings.event_log_dir`, `event_retention_max`, `event_retention_days`) — reuse the one the app wires in; in `run_once` change the fire branch to `event = self.safety.record_fire(frame, self.trigger.fire_confidence, self.trigger.fire_latency, time.time(), now)`. Keep `status.add_event`/`update` working (Task 4 switches status/history to the store; for now keep the in-memory event list populated too, or point `status.events()` at the store — coordinate with Task 4).
  - `state.py`: no change needed here beyond leaving `add_event` intact; Task 4 removes the duplication.
  - Wire the app entrypoint (`main.py`) to construct one `EventStore` and pass it to both `SafetyGovernor` and `web.create_app`.

- [ ] **Step 4: Run to verify PASS** — `uv run pytest tests/test_trigger.py tests/test_safety.py tests/test_pipeline.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "feat: trigger latency + EventStore-backed record_fire"`

---

### Task 4: Web endpoints — events, stats, clips route

**Files:**
- Modify: `src/doggy/web.py`
- Test: `tests/test_web.py`

**Interfaces:**
- Consumes: `EventStore` (list/delete/clear/stats), served images.
- Produces endpoints:
  - `GET /api/events?limit=N` -> `{"events":[{id,ts,wall_time,confidence,latency_s,thumb,clip,age_seconds}, ...]}` (recent-first; `age_seconds = now_mono - ts` using `time.monotonic()`).
  - `DELETE /api/events/{id}` -> 200 `{ok:true}` or 404.
  - `POST /api/events/clear` -> `{ok:true}`.
  - `GET /api/stats` -> `EventStore.stats()`.
  - `GET /clips/{name}` -> `FileResponse` from `event_log_dir` (`Path(name).name` guard), 404 if missing.
  - `/api/status` keeps working; its `events` now come from `EventStore.list(limit=10)` mapped to dicts.

- [ ] **Step 1: Write failing tests** (use FastAPI `TestClient`, an `EventStore` in a tmp dir seeded with 2 events, injected into `create_app`)
```python
def test_events_list_and_delete(client, store, seeded_ids):
    r = client.get("/api/events").json()
    assert len(r["events"]) == 2 and "age_seconds" in r["events"][0]
    assert client.delete(f"/api/events/{seeded_ids[0]}").status_code == 200
    assert len(client.get("/api/events").json()["events"]) == 1
    assert client.delete("/api/events/nope").status_code == 404

def test_stats_endpoint(client):
    assert "today" in client.get("/api/stats").json()

def test_clear(client):
    client.post("/api/events/clear")
    assert client.get("/api/events").json()["events"] == []
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement** the endpoints in `create_app` (thread the `EventStore` through `create_app(... event_store=...)`). Map `EventRecord` -> dict with `age_seconds`.

- [ ] **Step 4: Run to verify PASS** — `uv run pytest tests/test_web.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "feat: events/stats/clips HTTP endpoints"`

---

### Task 5: Snooze

**Files:**
- Modify: `src/doggy/safety.py`, `src/doggy/state.py`, `src/doggy/web.py`
- Test: `tests/test_safety.py`, `tests/test_web.py`

**Interfaces:**
- Produces:
  - `SafetyGovernor.snooze(seconds:float, now:float)`, `SafetyGovernor.cancel_snooze()`, `SafetyGovernor.snooze_remaining(now:float) -> float` (0 if not snoozed).
  - `allow_fire(now)` also returns False while `now < self._snooze_until`.
  - `Status.snoozed_until_seconds: float = 0.0` (remaining, set each loop).
  - Endpoints: `POST /api/snooze {minutes:float}`; `POST /api/snooze/cancel`.

- [ ] **Step 1: Write failing tests**
```python
def test_snooze_blocks_then_expires(tmp_path):
    from doggy.events import EventStore
    gov = SafetyGovernor(RuntimeSettings(Settings().tunable()), EventStore(tmp_path,10,0))
    gov.snooze(60, now=100.0)
    assert gov.allow_fire(now=100.0) is False
    assert gov.snooze_remaining(now=130.0) == 30.0
    assert gov.allow_fire(now=161.0) is True   # expired
    gov.snooze(60, now=200.0); gov.cancel_snooze()
    assert gov.allow_fire(now=200.0) is True
```
(Assumes `safety_enabled=True` and under the hourly cap.)

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement.** Add `self._snooze_until = 0.0`; `snooze` sets it to `now+seconds`; `cancel_snooze` sets 0; `snooze_remaining` = `max(0, until-now)`; `allow_fire` early-returns False when snoozed. Pipeline sets `status.update(snoozed_until_seconds=self.safety.snooze_remaining(now))`. Endpoints call through.

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit** — `git commit -am "feat: snooze the deterrent"`

---

### Task 6: Sound picker + volume (+ upload)

**Files:**
- Modify: `src/doggy/config.py`, `src/doggy/alerter.py`, `src/doggy/web.py`
- Test: `tests/test_alerter.py`, `tests/test_web.py`

**Interfaces:**
- Config: `selected_sound: str = "random"`. (Reuse existing `max_volume`.)
- Alerter: when `selected_sound != "random"` and the file exists, play it; else random. Apply volume (command backend adds `--volume <max_volume>` to `pw-play`; sounddevice scales samples by `max_volume`).
- Endpoints: `GET /api/sounds -> {"sounds":[names], "selected":"random"}`; `POST /api/sounds` (multipart, `.wav/.mp3/.ogg` only, sanitized `Path(name).name`, saved to `clips_dir`) -> `{ok:true, name}`.

- [ ] **Step 1: Write failing tests**
```python
def test_alerter_plays_selected(monkeypatch, tmp_path):
    # runtime.selected_sound set -> chosen file passed to the play call
    ...
def test_upload_rejects_non_audio(client):
    assert client.post("/api/sounds", files={"file": ("x.txt", b"x", "text/plain")}).status_code == 422
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement** selection + volume in `alerter.py`; list/upload endpoints in `web.py`.

- [ ] **Step 4: Run to verify PASS.**

- [ ] **Step 5: Commit** — `git commit -am "feat: choose deterrent sound + volume from the UI"`

---

### Task 7: Video clips — ClipBuffer, encode, deferred finalization

**Files:**
- Create: `src/doggy/clips.py`
- Modify: `src/doggy/config.py`, `src/doggy/pipeline.py`, `src/doggy/events.py` (uses `attach_clip`)
- Test: `tests/test_clips.py`

**Interfaces:**
- Config: `clips_enabled: bool = False`, `clip_window_seconds: float = 20`, `clip_preroll_seconds: float = 5`, `clip_postroll_seconds: float = 3`, `clip_fps: int = 6`, `clip_retention: int = 10`.
- Produces:
  - `ClipBuffer(window_seconds:float)`: `push(mono_ts:float, jpeg:bytes)`, `slice(start:float, end:float) -> list[bytes]`. Drops entries older than `window_seconds` on push.
  - `encode_clip(frames:list[bytes], fps:int, out_path:Path) -> Path` — decode JPEGs, write mp4 via `cv2.VideoWriter('mp4v')`; on failure write an animated `.webp` via Pillow and return that path.
- Pipeline: when `clips_enabled`, push each loop's annotated JPEG to the buffer; on fire register a pending `(event_id, fire_ts, end=fire_ts+postroll)`; each loop finalize pendings whose `end <= now` by encoding `buffer.slice(fire_ts-preroll, end)` and calling `event_store.attach_clip`. Clip retention handled by a `clip_retention` cap in `EventStore.prune`.

- [ ] **Step 1: Write failing tests** (`tests/test_clips.py`)
```python
def test_buffer_drops_old_frames():
    b = ClipBuffer(window_seconds=2.0)
    b.push(0.0, b"a"); b.push(1.0, b"b"); b.push(3.0, b"c")  # 0.0 now older than 2s
    assert b.slice(0.0, 3.0) == [b"b", b"c"]

def test_encode_falls_back(monkeypatch, tmp_path):
    # force cv2.VideoWriter to produce nothing -> webp fallback path returns .webp
    ...
```

- [ ] **Step 2: Run to verify FAIL.**

- [ ] **Step 3: Implement** `ClipBuffer` + `encode_clip` (with fallback); wire pipeline pending-clip finalization; add config; extend `EventStore.prune` to also cap clips to `clip_retention` (delete oldest clip files beyond the cap, clearing their `clip` field).

- [ ] **Step 4: Run to verify PASS** — `uv run pytest tests/test_clips.py -q`.

- [ ] **Step 5: Commit** — `git commit -am "feat: opt-in pre-roll video clip per catch"`

---

### Task 8: Dashboard UI — history, snooze, stats, sounds, clips

**Files:**
- Modify: `src/doggy/static/index.html`

**Interfaces:** consumes `/api/events`, `/api/stats`, `/api/snooze`, `/api/sounds`, `/events/{n}`, `/clips/{n}`, and the existing settings endpoints.

- [ ] **Step 1: History** — replace "Recent reactions" with a scrollable list from `GET /api/events`. Each row: thumbnail (click -> existing lightbox; if `clip` present show a play control that loads `/clips/{clip}` in a `<video controls>` inside the lightbox), a time label (`wall_time`, when it is a real date >= 2024, formatted as a local date/time; else relative from `age_seconds` -> "just now / 5m ago / 2h ago / 3d ago"), `reacted in Xs` (from `latency_s`, omitted if null), certainty `NN%`, and a delete "×" that calls `DELETE /api/events/{id}`. A "Clear all" button -> `POST /api/events/clear` behind a `confirm()`.
- [ ] **Step 2: Snooze** — in the status area, "Snooze" with 15m / 1h buttons (`POST /api/snooze {minutes}`). When `snoozed_until_seconds > 0`, show "Snoozed, 42m left" + "Resume now" (`POST /api/snooze/cancel`).
- [ ] **Step 3: Stats card** — from `GET /api/stats`: "N today, M this week", a 7-bar mini row from `per_day`, "busiest around Hh" (hidden if `busiest_hour` null), "avg reaction Xs".
- [ ] **Step 4: Sound + clips settings** — a sound `<select>` (Random + `/api/sounds`) patching `selected_sound`, a volume slider patching `max_volume`, and a "Save video clips" toggle patching `clips_enabled`.
- [ ] **Step 5: Manual verify + commit** — load the page against the running Pi (or `uv run doggy` with seeded events), confirm each panel renders and actions work; `git commit -am "feat: dashboard UI for history, snooze, stats, sounds, clips"`.

---

### Task 9: LAN clock sync at deploy

**Files:**
- Modify: `scripts/deploy-to-pi.sh`

- [ ] **Step 1:** In the remote provisioning block, derive the gateway (`GW=$(ip route | awk '/default/{print $3; exit}')`) and, if non-empty, write `/etc/systemd/timesyncd.conf` with `[Time]\nNTP=$GW\nFallbackNTP=` and `systemctl restart systemd-timesyncd`. Echo a note that timestamps depend on the router serving NTP; otherwise the UI shows relative time. No firewall change.
- [ ] **Step 2: Commit** — `git commit -am "feat: sync Pi clock from the LAN router at deploy"`

---

## Notes for the executor
- Start by creating a feature branch off `main` before Task 1.
- Hardware-only validation (real MP4 encode on the Pi, router NTP, `pw-play --volume`) happens at deploy after the whole-branch review, not in unit tests.
- Keep `run_once` calling `self.clock()` exactly once (existing invariant).
