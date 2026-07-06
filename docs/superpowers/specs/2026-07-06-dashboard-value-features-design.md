# Dashboard Value Features Design

**Repo:** https://github.com/devjerry0/watchdoggy

**Goal:** Add a batch of dashboard features that turn raw catches into a usable, self-managing log and give day-to-day controls: persistent event history with real timestamps, per-event delete, time-to-react, snooze, sound picker + volume, activity stats, and opt-in video clips.

**Architecture:** Introduce a disk-backed `EventStore` as the single source of truth for reactions (JPEG + `events.jsonl` + optional clip on the SD card), replacing the in-memory-only event deque. `SafetyGovernor` delegates event persistence to it. A `ClipBuffer` holds recent JPEG frames in memory so clips are derived on a catch (pre-roll included) with no continuous SD writes. Features 3 (stats) and 5 (clips) build on the `EventStore` foundation. Everything stays fully local: the clock is synced from the LAN router, never the internet.

**Tech Stack:** Python 3.11, FastAPI, pydantic-settings, OpenCV, systemd-timesyncd (LAN NTP), existing uv/pytest setup.

## Global Constraints

- **Fully local, no internet.** No outbound calls. Clock syncs from the LAN gateway via `systemd-timesyncd`; if that fails, the UI falls back to relative times. The egress firewall (`harden-pi.sh`) is unchanged.
- **SD-wear conscious.** No new continuous writes. Every store has a retention cap that prunes old files. Clips are opt-in and separately capped.
- **Backward compatible.** Existing `events.jsonl` lines (missing new fields) must still load, using defaults.
- **Existing conventions.** `DOGGY_` env prefix, frozen `TunableSettings`, current test style (`tests/test_*.py`, `-m "not slow"`), no emoji in UI, plain-language labels.
- **Config stays live-tunable** where it makes sense (retention, snooze default, clip settings, selected sound, volume) via the dashboard + `.env`.

## Event schema (shared foundation)

`events.jsonl`, one JSON object per line:

```json
{"id": "fire_1751808840123", "ts": 16291.834, "wall_time": 1751808840.123,
 "confidence": 0.82, "latency_s": 1.8, "thumb": "fire_1751808840123.jpg",
 "clip": "fire_1751808840123.mp4"}
```

- `id` — stable unique id (`fire_<wall_ms>`); names the jpeg/clip files; used for delete.
- `ts` — monotonic seconds (for age/relative time, always correct).
- `wall_time` — Unix epoch (real timestamp); may be wrong if the clock never synced.
- `latency_s` — seconds from first sighting to fire (time-to-react).
- `confidence` — triggering confidence (already correct post conf-0 fix).
- `thumb` — full-frame JPEG filename.
- `clip` — clip filename, or absent/null.

Missing fields on old lines default to: `wall_time=null`, `latency_s=null`, `id`=thumb stem, `clip=null`.

---

## Feature 1 (foundation): Persistent history + delete + timestamps + time-to-react

**New module `src/doggy/events.py` — `EventStore`:**
- `__init__(event_dir, max_events, max_age_days)` — loads existing `events.jsonl` into an in-memory list (most recent `max_events`).
- `add(frame, confidence, latency_s, wall_time, mono_ts) -> EventRecord` — writes the JPEG, appends the record to memory + `events.jsonl`, then `prune()`. Returns the record.
- `attach_clip(id, clip_name)` — records a clip filename on an existing event (used by Feature 5).
- `list(limit=None) -> list[EventRecord]` — most-recent-first.
- `delete(id) -> bool` — remove the record from memory, rewrite `events.jsonl` without it, delete its JPEG and clip files.
- `clear()` — delete all events, jpegs, clips; truncate `events.jsonl`.
- `prune()` — enforce `max_events` and `max_age_days`; delete the dropped events' files. Called after every `add`.
- `stats()` — see Feature 3.

**`SafetyGovernor` change:** `record_fire` delegates file/record writing to `EventStore` (inject the store). The rate-limit deque stays in `SafetyGovernor`. The trigger provides `latency_s` (see below); the pipeline passes `wall_time=time.time()`.

**Trigger change (`trigger.py`):** on the fire edge, set `self.fire_latency = now - self._confirm_start` alongside `fire_confidence`. The pipeline records it.

**Config (`TunableSettings`):**
- `event_retention_max: int = 500` (ge=0; 0 = unlimited)
- `event_retention_days: int = 30` (ge=0; 0 = no age limit)

**Endpoints (`web.py`):**
- `GET /api/events?limit=N` — full history: each item gets `age_seconds` (server-computed `now_mono - ts`) plus `wall_time`, `latency_s`, `confidence`, `thumb`, `clip`, `id`.
- `DELETE /api/events/{id}` — delete one (404 if missing; `Path(id).name` guard, no traversal).
- `POST /api/events/clear` — delete all.
- Existing `GET /events/{name}` (jpeg) unchanged; add `GET /clips/{name}` (Feature 5).

**Clock sync (deploy):** `deploy-to-pi.sh` (or a small `setup-clock.sh`) writes `/etc/systemd/timesyncd.conf` with `NTP=<gateway-ip>` (derived from `ip route`) and restarts `systemd-timesyncd`. Documented as best-effort; no firewall change.

**UI:** replace "Recent reactions" with a scrollable **History** list. Each row: thumbnail (click -> lightbox), timestamp (absolute "Today 2:14pm" when `wall_time` is a real date, else relative "2h ago" from `age_seconds`), "reacted in 1.8s", certainty, and a delete "x". A "Clear all" button with a confirm. Poll refreshes it.

**Tests:** `test_events.py` — load (incl. old-schema lines), add, prune-by-count, prune-by-age, delete (removes files), clear, stats; trigger latency captured on the fire edge; pipeline records `wall_time`/`latency_s`.

---

## Feature 2: Snooze

Temporarily pause the deterrent, auto-re-arm. Transient (not persisted; a reboot clears it).

- **State:** a `snooze_until` monotonic timestamp held in a small runtime holder (or on `SafetyGovernor`). `SafetyGovernor.allow_fire(now)` returns `False` while `now < snooze_until` (in addition to the existing `safety_enabled` + rate checks).
- **Endpoints:** `POST /api/snooze {minutes}` sets `snooze_until = now + minutes*60`; `POST /api/snooze/cancel` clears it. Status exposes `snoozed_until_seconds` (remaining) so the UI can show a countdown.
- **UI:** a "Snooze" control in the status area with quick options (15m / 1h) and, when active, a live countdown + "Resume now". Distinct from the master Deterrent toggle.
- **Tests:** `allow_fire` false while snoozed, true after expiry and after cancel.

---

## Feature 3: Activity stats

Built on `EventStore`.

- **`EventStore.stats()`** returns: `today`, `this_week` counts; `per_day` (last 7 days, list of {day, count}); `busiest_hour` (0-23, from `wall_time`; `null` if clock unsynced); `avg_latency_s`.
- **Endpoint:** `GET /api/stats`.
- **UI:** an "Activity" card: "3 today, 14 this week", a small 7-day bar row, "busiest around 6pm" (hidden if unknown), "avg reaction 1.6s".
- **Tests:** `stats()` aggregation over a fixed event list (fixed clock), busiest-hour null when `wall_time` missing.

---

## Feature 4: Sound picker + volume (+ optional upload)

- **Config:** `selected_sound: str = "random"` (filename or "random"); reuse existing `max_volume` (0..1).
- **Alerter:** play `selected_sound` if set (else random from `clips_dir`); apply volume (command backend: `pw-play --volume`; sounddevice: scale samples).
- **Endpoints:** `GET /api/sounds` (list clip filenames); `POST /api/sounds` (multipart upload -> save to `clips_dir`, sanitized filename, audio extensions only); `DELETE /api/sounds/{name}` optional.
- **UI (Settings):** a sound dropdown (Random + each file), a volume slider, keep Test sound. Small "upload" control.
- **Tests:** alerter picks `selected_sound` when set, random otherwise; upload rejects non-audio/sanitizes names.

---

## Feature 5: Video clip per catch (opt-in)

- **`ClipBuffer` (`clips.py`):** a time-windowed deque of `(mono_ts, jpeg_bytes)`. `push(mono_ts, jpeg_bytes)` appends and drops entries older than `clip_window_seconds`. Only fed when `clips_enabled`. Frames come from the pipeline (reuse the annotated-frame JPEG encode).
- **Deferred finalization for post-roll:** on a fire, register a pending clip with `end = fire + clip_postroll_seconds`. Each pipeline loop, finalize any pending clip whose `end` has passed: slice the buffer `[fire - preroll, end]`, encode, and `EventStore.attach_clip`.
- **Encode:** try MP4 via `cv2.VideoWriter` (`mp4v`); if it fails, fall back to an animated GIF/WebP via Pillow. Store in `event_dir`, served at `/clips/{name}`.
- **Config:** `clips_enabled: bool = False`, `clip_window_seconds=20`, `clip_preroll_seconds=5`, `clip_postroll_seconds=3`, `clip_fps=6`, `clip_retention=10` (separate tight cap; pruned by `EventStore`).
- **UI:** "Save video clips" toggle (Settings). In history, events with a clip get a play badge; the lightbox plays the clip (`<video>` for mp4, `<img>` for animated).
- **Tests:** `ClipBuffer` windowing (drops old frames); pending-clip finalizes after post-roll; encode fallback path selects the alternate format when the primary raises. Actual on-Pi encode validated at deploy.

---

## Rollout / testability

- Pure logic (EventStore, ClipBuffer, stats, snooze gating, latency, sound selection) is unit-tested with injected clocks/temp dirs, no hardware.
- Hardware-dependent bits validated on the Pi at deploy: router NTP sync, real MP4 encode, `pw-play --volume`.
- Build on a feature branch; deploy after the whole-branch review; keep 640/fan work out of scope.
