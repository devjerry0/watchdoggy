# Soothing Sounds Mode: Design

A calm-audio mode: upload soothing tracks (up to 1 GB total), loop them
through the speaker, and get out of the way the moment the deterrent has
business. Fully local as always.

## Library

- New `soothing/` directory beside `events/` and `sounds/` (gitignored,
  created on demand). Formats: `.mp3, .wav, .flac, .ogg` — mp3 confirmed
  decodable by the Pi's pw-play (system libsndfile links libmpg123;
  verified on the device 2026-07-08).
- `GET /api/soothing` — tracks (name, size) + `total_bytes` + `limit_bytes`.
- `POST /api/soothing` — multipart upload STREAMED to disk in chunks (a
  200 MB file must not be buffered in RAM). Rejected with 413 and a plain
  message when the upload would push the library past 1 GB
  (`soothing_limit_bytes`, structural, default 1 GiB) or past the same
  limit per file. Extension checked; `Path(name).name` traversal guard
  like the sounds router.
- `DELETE /api/soothing/{name}` — removes a track (player skips it on the
  next loop iteration; deleting the currently-playing track just ends it
  early via the missing-file play error, which the loop treats as skip).

## Player

- New `reaction/soothing.py`: `SoothingPlayer`, a daemon-thread loop owned
  by the composition root. While `soothing_enabled` (tunable, default
  False): play tracks in name order, one `pw-play --volume <v> <file>`
  subprocess at a time (registry fallback exactly like CommandAlerter:
  pw-play/pw-cat -> afplay on Mac -> log-and-idle). Track ends -> next;
  list exhausted -> loop. Empty library -> idle poll.
- Volume: `soothing_volume` tunable (default 0.4), read fresh each track.
- **Alarms interrupt (the point):** the player is a hub `Reaction` (wrapped
  in SafeReaction like the others). `on_dog_caught`: terminate the current
  track subprocess immediately and hold playback until
  `soothing_resume_seconds` (tunable, default 45) after the LAST catch —
  long enough to cover escalation strikes within an incident, since those
  do not publish separate hub events. Resume starts from the next track
  (no mid-track seek; simplicity over fidelity).
- Test sound and push-to-talk deliberately OVERLAP the music (PipeWire
  mixes; both are short and user-initiated). Only real catches interrupt.
- Toggling off (PATCH) terminates the current track within a second (the
  loop re-checks cfg between chunks of a bounded subprocess wait).
- Service restart with the mode enabled resumes looping on boot.
- `Status.soothing_track: str | None` — the currently playing track name,
  shown on the dashboard; None when idle/held/disabled.

## Dashboard

- New "Soothing sounds" card (side column, after Deterrence): a "Play
  soothing sounds" toggle (`soothing_enabled`), a "Now playing: <name>"
  line (or "Paused for the alarm." while held / hidden when off), a
  volume slider ("Soothing loudness"), the track list with per-track size
  and delete, a usage line ("312 MB of 1 GB used"), and an upload input
  (desc: "Add calm music or white noise. Up to 1 GB total.").
- Upload UX: the existing sound-upload idiom, plus a simple progress note
  ("Uploading, this can take a minute for big files.").

## Constraints (project invariants unchanged)

- No new dependencies; stdlib + existing tools only. Egress firewall
  untouched. `main.py` shim untouched. Plain-language copy, no emoji.
- The deterrent path must be unaffected when the mode is off (player
  thread parked, zero subprocesses).
- Tests green + ruff clean per task; deploy remains rsync + restart, with
  one live probe at deploy time: `pw-play` an uploaded mp3 on the Pi.

## Build order (one plan, 3 tasks)

1. Library: storage + routers (list/upload/delete, caps, streaming).
2. Player: loop thread, interruption via hub, status field, wiring.
3. Dashboard card + docs (README section).
