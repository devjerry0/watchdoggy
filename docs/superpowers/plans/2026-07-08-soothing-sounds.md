# Soothing Sounds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload up to 1 GB of calm audio, loop it through the speaker, and interrupt it instantly when the deterrent fires (resume 45s after the last catch).

**Architecture:** A `soothing/` library with a streaming upload router; a `SoothingPlayer` daemon-thread loop (subprocess per track, CommandAlerter-style backend fallback) that doubles as a hub `Reaction` — `on_dog_caught` terminates the current track and holds playback; a dashboard card.

**Tech Stack:** Existing only (FastAPI, stdlib subprocess/threading, pw-play on the Pi — mp3 decode verified on-device).

**Spec:** `docs/superpowers/specs/2026-07-08-soothing-sounds-design.md`

## Global Constraints

- No new dependencies; egress firewall untouched; `src/doggy/main.py` untouched.
- Uploads must STREAM to disk (chunked); never buffer a whole file in RAM (Pi has 4 GB total).
- The deterrent path is unaffected when the mode is off: player thread parked, zero subprocesses.
- Per task: `uv run pytest -m "not slow" -q` green and `uv run ruff check src tests` clean; `node --check` the inline script when JS changes.
- Commits authored `devjerry0 <99199491+devjerry0@users.noreply.github.com>`; plain-language dashboard copy, sentence case, no emoji, no em dashes.
- Path-traversal guard on every filename that reaches the filesystem (`Path(name).name`, the sounds-router idiom).

---

### Task 1: Library storage + endpoints

**Files:**
- Create: `src/doggy/web/routers/soothing.py`
- Modify: `src/doggy/core/config.py` (structural `Settings`), `src/doggy/web/app.py` (include router; pass settings), `.gitignore` (`/soothing/`)
- Test: `tests/web/test_soothing.py` (new)

**Interfaces:**
- Produces: `Settings.soothing_dir: Path = Path("soothing")`, `Settings.soothing_limit_bytes: int = 1_073_741_824` (1 GiB; also the per-file cap). Router endpoints:
  - `GET /api/soothing` -> `{"tracks": [{"name": str, "size": int}...] (name-sorted), "total_bytes": int, "limit_bytes": int}`
  - `POST /api/soothing` (multipart `file`) -> 200 `{"name": ...}`; 413 with `{"detail": "That would go over the 1 GB limit. Delete a track first."}` when total would exceed the limit; 415 for extensions outside `{".mp3", ".wav", ".flac", ".ogg"}`
  - `DELETE /api/soothing/{name}` -> 200 / 404
- Upload mechanics: stream `UploadFile` in 1 MiB chunks to `<dir>/.upload.part`, counting bytes; abort (delete partial, 413) the moment `existing_total + written > limit`; on success `os.replace` the part file to the final name (`Path(filename).name`). The part file must never be listed as a track (filter dotfiles/extension in GET).

- [ ] **Step 1: Failing tests** — list empty library; upload a small wav -> appears in list with correct size and total; upload pushing past a monkeypatched tiny limit (e.g. settings with `soothing_limit_bytes=1000` and a 2 KB payload) -> 413, partial file cleaned up (dir contains no `.part`), library unchanged; bad extension -> 415; delete -> gone, 404 on repeat; traversal name (`../evil.mp3`) lands as `evil.mp3` inside the dir. Use the existing web-test client/builder idioms with `tmp_path` as `soothing_dir`.
- [ ] **Step 2: Implement** (router shaped like `sounds.py`; `async def` upload with `await file.read(CHUNK)` loop). Include router in `create_app`.
- [ ] **Step 3: Gates, commit** — `feat: soothing sounds library (1 GB, streamed uploads)`.

---

### Task 2: SoothingPlayer loop + alarm interruption

**Files:**
- Create: `src/doggy/reaction/soothing.py`
- Modify: `src/doggy/core/config.py` (tunables), `src/doggy/core/status.py` (`soothing_track`), `src/doggy/app.py` (build/wire/start)
- Test: `tests/reaction/test_soothing.py` (new)

**Interfaces:**
- Produces: tunables `soothing_enabled: bool = False`, `soothing_volume: float = Field(0.4, ge=0.0, le=1.0)`, `soothing_resume_seconds: float = Field(45.0, ge=0.0)`. `Status.soothing_track: str | None = None`. `SoothingPlayer(runtime, library_dir, status, clock=time.monotonic, spawn=None)` with `start()` (daemon thread), `on_dog_caught(event)` (hub Reaction), and an injectable `spawn(path, volume) -> subprocess.Popen | None` for tests (default `_spawn_player`: pw-play/pw-cat with `--volume`, afplay `-v` on darwin, else None -> log once and idle).

- Player loop semantics (pin each in a test):
  - Disabled or empty library: no subprocess, `soothing_track` None, cheap idle poll (0.5s).
  - Enabled: play name-sorted tracks round-robin; `soothing_track` set while a track plays, None otherwise. Volume read fresh per track.
  - Between 0.5s waits on the running subprocess, re-check cfg: toggling off terminates the current track within ~1s.
  - `on_dog_caught`: under a small lock, `hold_until = clock() + cfg.soothing_resume_seconds` and terminate the current subprocess immediately (this runs on the detect thread via SafeReaction — keep it non-blocking: `terminate()`, no `wait()`).
  - While `clock() < hold_until`: no playback, `soothing_track` None. Resume with the NEXT track after the hold. A second catch during the hold extends it.
  - A track whose file vanished or whose player exits nonzero: log at info, skip to next (no crash loop; if EVERY track fails in a full pass, idle one poll interval before retrying the list — pin this with a spawn that always returns a failing proc and assert bounded spawn count per time window).
- Wiring in `app.py`: build after clip_service, register `SafeReaction(player)` LAST in the hub list, `player.start()` just before `pipeline.run(stop)`. Library dir comes from `settings.soothing_dir`.

- [ ] **Step 1: Failing tests** with a fake spawn recording `(path, volume)` and returning controllable fake procs (poll()/terminate()), and a fake clock — cover every semantic above (sequential order, loop-around, interruption + hold + extension, off-toggle, volume, all-fail backoff, disabled = zero spawns).
- [ ] **Step 2: Implement.** **Step 3: Gates, commit** — `feat: soothing player loop, interrupted by the deterrent`.

---

### Task 3: Dashboard card + README

**Files:**
- Modify: `src/doggy/web/static/index.html`, `README.md`
- Test: `tests/web/test_api.py` (copy presence), `tests/web/test_soothing.py` (unchanged)

- [ ] **Step 1:** "Soothing sounds" card in the side column after the Deterrence card (mobile order: c-soothe between c-lab and c-set; bump c-set/c-sys). Contents: toggle "Play soothing sounds" (tunable idiom, like clips_enabled); status line `id="soothing_now"` — "Now playing: <name>" when `s.soothing_track`, "Paused for the alarm." when enabled-but-held (enabled true and track null and state is not IDLE... keep it simple: enabled && !track -> "Quiet right now."), hidden when disabled; "Soothing loudness" slider (`soothing_volume` in KNOBS/FMT pct — it inherits the touch-lock automatically); track list (name, human size, delete button) + "<X> MB of 1 GB used" line from `/api/soothing`; upload input (accept=".mp3,.wav,.flac,.ogg,audio/*", desc "Add calm music or white noise. Up to 1 GB total.", progress note "Uploading, this can take a minute for big files." while in flight, uses the existing uploadSound idiom against `/api/soothing`).
- [ ] **Step 2:** `refreshSoothing()` on the 4s interval (fetch `/api/soothing`, render list; skip re-render when unchanged, the `_evKey` idiom). `soothing_track` rides the existing 500ms status poll.
- [ ] **Step 3:** README: a short "Soothing sounds" paragraph under Using the dashboard (what it does, 1 GB cap, alarms interrupt).
- [ ] **Step 4:** Gates + `node --check`; commit — `feat: soothing sounds dashboard card`.

---

## Deploy note

rsync + restart (no new deps). Live probe: upload an mp3 through the dashboard, toggle on, confirm audio from the JBL; fire the test sound (should MIX, not interrupt); simulate a catch if convenient (or wait for Holly) to verify the interrupt + resume.

## Self-review notes

- The player's hold logic uses the injected monotonic clock everywhere; no wall time.
- Escalation strikes don't publish hub events — covered by the 45s default hold per spec (documented in code).
- The upload part-file never collides with tracks (dot-prefixed, filtered from GET; os.replace atomic on same fs).
