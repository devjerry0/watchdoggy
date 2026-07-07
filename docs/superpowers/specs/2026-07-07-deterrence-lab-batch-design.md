# Deterrence Lab Batch: Design

Nine features that turn watchdoggy from a noise-maker into a system that
measures whether the noise works. Fully local as always: no cloud, no new
internet dependencies, egress firewall untouched.

Decisions already made with the user:
- No smart-home hub: notifications are browser notifications from the open
  dashboard tab. No MQTT, no webhooks.
- HTTPS with a self-signed certificate is accepted (one-time browser warning
  per device) to unlock microphone access for push-to-talk and the
  Notification API.

## 1. Watch-for classes (dog / cat / bird), detect and alert separately

The deployed YOLO26n model detects 80 COCO classes. The watcher becomes
class-configurable over a curated animal menu: **dog, cat, bird** — with
two independent checkmarks per animal:

- **Detect**: the animal is detected, drawn on the live view, and counted
  in the "in view" readout. Never fires anything by itself.
- **Alert**: the animal can trigger the deterrent (and everything downstream:
  events, outcomes, escalation). Alert requires Detect: the UI disables the
  Alert checkmark until Detect is on, and unchecking Detect clears Alert.

So "watch birds but never chirp at them" is: birds Detect on, Alert off.
Zero Alert classes is valid (pure monitor mode: it watches and shows, never
reacts).

- Config: `DOGGY_TARGET_LABELS` (detected classes, comma-separated, default
  `dog`, at least one required) and `DOGGY_ALERT_LABELS` (default `dog`),
  validated as a subset of `target_labels`; may be empty. Both tunable from
  the dashboard.
- `vision`: the detector keeps detections whose label is `person`, any
  detected class, or an inventory class (section 2b). `FrameAnalysis.dogs`
  renames to `FrameAnalysis.targets` (all detected animals, drawn);
  `candidates` seeds from the subset whose label is in `alert_labels`, so
  detect-only animals are drawn in the existing "ignored" grey and can
  never trigger. Person suppression applies to any target box coinciding
  with a person box (same IoU rule, unchanged threshold).
- Status JSON: `dogs` key renames to `targets` (dashboard is the only
  consumer; updated in the same change).
- Dashboard copy goes dynamic: "Dogs in view" derives from the detected
  classes ("Dogs in view", "Cats in view", "Animals in view"); "Certainty
  it's a dog" and the "Dog spotted" state word derive from the alert
  classes (they describe what fires). Existing plain-language style kept.
- Trigger, zone, cooldowns, hourly cap: unchanged (they operate on
  confirmed targets regardless of class).
- Existing `.env` files without the new var behave exactly as today.

## 2. Outcome watcher (the Lab's core measurement)

After each fire, measure how long until the target actually left.

- New `reaction/outcome.py`: `OutcomeWatcher`, a per-frame observer in the
  pipeline (same pattern as `ClipService`): on fire it starts a pending
  measurement; each subsequent frame checks whether any confirmed target
  remains in the zone.
- `clear_seconds` = time from fire until the zone has been target-free for
  a debounce period (default 2.0s of consecutive clear frames), minus the
  debounce. Capped: if still occupied after 60s, record `null` (= not
  deterred) and stop watching.
- Persists via `EventStore.attach_outcome(id, clear_seconds)` (same shape
  as `attach_clip`). `EventRecord` gains `clear_seconds: float | None` and
  `sound: str | None`, both defaulting to None when absent so existing
  events.jsonl files load unchanged.
- Sound attribution: `BaseAlerter.alert()` returns the chosen clip's name
  (None when nothing played); `SoundReaction` calls
  `EventStore.attach_sound(event.record.id, name)`.

## 2b. Counter inventory (kitchen and food objects)

The same forward pass already detects COCO's food and tableware classes;
we stop discarding them and put them to work. Inventory classes are
observed, never targeted: they cannot fire the deterrent.

- Classes: `banana, apple, sandwich, orange, broccoli, carrot, hot dog,
  pizza, donut, cake` (food) and `bottle, wine glass, cup, fork, knife,
  spoon, bowl` (tableware). Fixtures (oven, sink, refrigerator...) are
  excluded: they never move, so they are noise.
- Config: `inventory_enabled` (default on) and `inventory_confidence`
  (default 0.4, independent of the target threshold: overhead food shots
  score lower). Only items inside the watch area count: the zone is what
  defines "the counter".
- `FrameAnalysis` gains `inventory` (label + box list, zone-filtered).
  Presence is debounced: an item label counts as present when seen in at
  least 2 of the last 5 analyzed frames (flicker-proofing, tracked by the
  consumer, not the detector).
- Dashboard: an "On the counter" line in the monitor card's readout area
  listing current item labels ("sandwich, bowl, 2 cups"; "nothing it
  recognizes" when empty). A "Show counter items" toggle (default off)
  additionally draws thin outline boxes on the live view; target and
  person boxes are unaffected.
- Theft forensics: when a fire happens, the outcome watcher snapshots the
  debounced inventory labels; when the incident clears (same debounce as
  `clear_seconds`), it snapshots again. Labels present before but missing
  after are recorded as `EventRecord.taken: list[str]` (default empty,
  old event logs load unchanged). The catch log line becomes
  "87% sure - reacted in 1.2s - took the sandwich", and the browser
  notification body includes it. Items that merely moved within the zone
  do not count as taken (label-level diff, not box tracking).
- The Deterrence card shows a theft tally for the week ("2 items lost").
  A catch with anything taken counts as not deterred in the deterred
  rate, whatever its clear time.

## 3. Escalation ladder

If the target is still in the zone after the first sound, get louder.

- Config: `escalation_enabled` (default off), `escalation_seconds`
  (default 8), `escalation_max_strikes` (default 3),
  `escalation_volume_step` (default 0.2).
- The `OutcomeWatcher` drives it: when a pending measurement is still
  occupied `escalation_seconds` after the last strike and strikes <
  max, it requests another alert at volume
  `min(1.0, max_volume + strike_index * escalation_volume_step)`.
- Escalation strikes bypass the cooldown (they are the same incident) but
  still honor `safety_enabled`, snooze, the armed schedule, and the hourly
  cap via a dedicated `FireGate.allow_escalation(now)`.
- `EventRecord` gains `strikes: int` (default 1). The catch log shows
  "3 strikes" when > 1. Strikes do not create new events.

## 4. Deterrence Lab stats

- `/api/lab`: per-sound aggregate over events that have both `sound` and an
  outcome: plays, deterred rate (cleared within 15s), average
  `clear_seconds`, and a habituation signal (average clear time of that
  sound's first half of plays vs. second half; "wearing off" when the
  second half is at least 50% slower with 6+ plays).
- Dashboard: new "Deterrence" card next to Activity: a small table of
  sounds with plays / avg escape time / deterred %, a "wearing off" tag
  when flagged, and plain-language empty state until there is data.
- Buckets and week boundaries use Pi-local time, like `/api/stats`.

## 5. Arming schedule

- Config: `schedule_enabled` (default off) and `armed_windows` (JSON list
  of `{"days": [0-6 Mondays-first], "start": "HH:MM", "end": "HH:MM"}`,
  stored in `.env` as a JSON string like `zone_points`). Windows crossing
  midnight (end <= start) wrap to the next day.
- `FireGate.allow` gains the schedule check (order: safety, schedule,
  snooze, hourly cap). Detection and the live view keep running while
  disarmed; only reactions stop.
- Status gains `armed: bool` and `next_change_seconds`; the pill shows a
  new "Off duty" state (dim lamp) with "back on at 21:00" wording.
- Dashboard: schedule editor in Settings: toggle plus rows of day-chips
  and start/end time inputs; times use the Pi's local day.

## 6. HTTPS (private CA, decided over self-signed and Let's Encrypt)

Let's Encrypt was considered and rejected: it cannot issue for `.local`
names, would require an owned domain plus DNS-01 automation, and its
~90-day renewals need recurring internet access the firewalled Pi
deliberately lacks. A household private CA gives the same green-lock UX
with zero renewals and zero internet.

- `scripts/setup-https.sh <user@host>`: on the Pi, generates a 10-year EC
  private CA ("watchdoggy home CA", `~/doggy/certs/ca.pem` + key, 0600),
  then issues a server certificate signed by it (SAN: `doggypi.local`
  plus the Pi's LAN IP, EKU serverAuth, 825 days — Apple's maximum
  trusted TLS lifetime; re-running the script re-issues without touching
  the CA, so devices never need re-onboarding). Sets
  `DOGGY_SSL_CERT`/`DOGGY_SSL_KEY` in the Pi's `.env`.
- `GET /ca.pem` serves the CA certificate (public material) so each
  device can download and trust it once: iOS (install profile + enable
  in Certificate Trust Settings), Android, macOS Keychain, Windows.
  After that one-time step the dashboard shows a normal padlock — no
  warnings. README documents the per-platform steps in plain words.
- `web.serve()` passes `ssl_certfile`/`ssl_keyfile` to uvicorn when both
  are set. With TLS configured the dashboard serves https on port 8443,
  and port 8000 stays plain HTTP as an "onboarding door": it probes
  whether the visiting device trusts the home CA (an http page can fetch
  an https /ping on the same host; the fetch fails iff untrusted) and
  either redirects straight to the https dashboard or walks the user
  through the one-time CA install (Apple devices get a .mobileconfig,
  others the raw .pem). Existing bookmarks keep working forever. No cert
  vars = plain HTTP exactly as today (Mac dev stays
  http://127.0.0.1:8000, already a secure context for mic/notifications).
- The CA private key lives on the Pi (0600, LAN-only box) — right-sized
  for a household threat model; noted in the README.

## 7. Push-to-talk

- Dashboard: a hold-to-talk button on the monitor card. While held:
  `getUserMedia` mono audio, an `AudioWorklet` downsamples to 16 kHz
  s16 PCM frames, sent as binary WebSocket messages to `/ws/talk`.
- Server: accepts one talk connection at a time (second connection gets
  a "busy" close). Frames pipe to a `pw-cat --playback --rate 16000
  --channels 1 --format s16 -` subprocess (PipeWire routes to the JBL,
  same as the deterrent). Subprocess ends on socket close. On hosts
  without pw-cat (Mac dev), fall back to the `afplay`-style null: log
  and discard.
- Half-duplex by design; deterrent sounds may overlap (PipeWire mixes).

## 8. Browser notifications

- Dashboard toggle "Notify this device" requests Notification permission
  (needs the HTTPS from #6; on plain http the toggle explains why it is
  unavailable).
- The existing 500ms poll fires a notification when a new event id appears:
  title "Dog on the counter" (class-aware wording), body with confidence,
  the snapshot as the notification image. Tab must be open (documented
  honestly in the UI copy).

## 9. Kiosk mode, report card, export

- Kiosk: a "Fullscreen" button on the monitor card calls
  `requestFullscreen()` on the monitor; `:fullscreen` CSS shows only the
  video, OSD, and status. Esc exits (native).
- Report card: `/api/stats` gains `report_card`: this week vs last week
  attempt counts, deterred rate, average escape time, and a letter grade.
  Rubric: start at 100 points; -5 per attempt this week (capped at -40);
  -30 if attempts rose vs last week, +10 if they fell; scale by deterred
  rate when outcomes exist (multiply by rate, so 50% deterred halves it).
  90+ A, 80+ B, 65+ C, 50+ D, else F; +/- from the top/bottom third of
  each band. No events at all = A ("quiet week"). Dashboard renders it as
  one line in the Activity card ("This week: B+. 11 attempts, all
  deterred, escapes trending faster.").
- Export: `GET /api/export` streams a zip of `events.jsonl` plus all
  snapshots/clips (built with `zipfile` into a spooled temp file, not
  memory). "Export all" button in the Catch log header.

## Build order

Two plans, each independently shippable:

- **Plan 1 — watcher smarts:** watch-for classes; counter inventory +
  theft forensics; outcome watcher + sound attribution; escalation; lab
  stats + deterrence card; report card.
- **Plan 2 — comms and polish:** HTTPS script + serving; push-to-talk;
  browser notifications; arming schedule; kiosk; export.

## Constraints (unchanged project invariants)

- Fully local; egress firewall stays armed; no new runtime dependencies
  that require opening egress (everything above uses stdlib, existing
  deps, or on-Pi tools; `pw-cat` ships with PipeWire).
- `main.py` entry shim untouched; deploys remain rsync + restart.
- EventStore stays the single writer through its lock; new attach methods
  follow `attach_clip`'s locking.
- Existing events.jsonl files must load unchanged (new fields default).
- All dashboard copy stays plain-language; no emoji.
- Tests green (`uv run pytest -m "not slow"`) and ruff clean per task;
  no personal info in the repo.
