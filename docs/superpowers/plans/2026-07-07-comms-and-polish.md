# Comms and Polish Implementation Plan (Deterrence Lab batch, Plan 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HTTPS on the Pi, push-to-talk from the dashboard to the JBL, browser notifications on catches, a weekly arming schedule, kiosk fullscreen, and an export-everything zip.

**Architecture:** `serve()` grows optional TLS; a `/ws/talk` websocket pipes raw PCM from the browser mic into `pw-cat` (PipeWire mixes it with deterrent audio); an `is_armed` pure function in `decision/schedule.py` gates fires by local wall time; the rest is dashboard JS plus two small endpoints.

**Tech Stack:** Existing deps only, with ONE conditional exception called out in Task 3 (websocket protocol library, pure-Python, vendorable offline).

**Spec:** `docs/superpowers/specs/2026-07-07-deterrence-lab-batch-design.md` (sections 5-9).

**Base:** Execute only after Plan 1 (`2026-07-07-watcher-smarts.md`) has merged; dashboard edits anchor to Plan 1's final `index.html`.

## Global Constraints

Same as Plan 1 (fully local, firewall stays armed, `main.py` shim untouched, `EventStore` lock discipline, plain-language copy, per-task pytest + ruff gates, devjerry0 author), plus:

- The Pi deploy path stays `rsync + systemctl restart` — any new Python dependency must be a pure-Python wheel vendorable offline, and the deploy script must install it explicitly.
- Plain HTTP must keep working exactly as today when the two SSL env vars are absent (Mac dev stays `http://127.0.0.1:8000`).
- Schedule decisions use Pi-local wall time.

---

### Task 1: HTTPS serving + private-CA certificate script

Decision record: private CA over plain self-signed (no per-browser
warnings after a one-time CA install per device) and over Let's Encrypt
(cannot issue for `.local`; renewals would need recurring internet the
firewalled Pi deliberately lacks).

**Files:**
- Create: `scripts/setup-https.sh`
- Modify: `src/doggy/core/config.py` (structural `Settings`), `src/doggy/web/app.py` (`serve` + `/ca.pem` route), `README.md`
- Test: `tests/web/test_api.py`

**Interfaces:**
- Produces: `Settings.ssl_cert: Path | None = None`, `Settings.ssl_key: Path | None = None`, `Settings.ca_cert: Path | None = None`; `serve()` passes `ssl_certfile`/`ssl_keyfile` to `uvicorn.run` when BOTH are set (else exactly today's call); `GET /ca.pem` returns the CA certificate as `application/x-pem-file` when `ca_cert` is set and the file exists, 404 otherwise.

- [ ] **Step 1: Failing test** (`tests/web/test_api.py`):

```python
def test_serve_passes_ssl_when_configured(monkeypatch, tmp_path):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    cert, key = tmp_path / "c.pem", tmp_path / "k.pem"
    cert.write_text("x"), key.write_text("x")
    s = _settings(ssl_cert=cert, ssl_key=key)   # reuse the module's settings helper
    serve(s, *_serve_deps(s))
    assert calls["ssl_certfile"] == str(cert) and calls["ssl_keyfile"] == str(key)

def test_serve_plain_http_without_ssl(monkeypatch):
    calls = {}
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.update(kw))
    s = _settings()
    serve(s, *_serve_deps(s))
    assert "ssl_certfile" not in calls
```

(Write `_serve_deps` once: builds runtime/buffers/status/alerter/store/gate the same way existing web tests do.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** `Settings` gains the two fields (comment: "Optional TLS: set both to serve https; needed for mic + notifications"). `serve()`:

```python
    kwargs: dict = {}
    if settings.ssl_cert and settings.ssl_key:
        kwargs = {"ssl_certfile": str(settings.ssl_cert),
                  "ssl_keyfile": str(settings.ssl_key)}
    uvicorn.run(app, host=settings.web_host, port=settings.web_port,
                log_level="warning", **kwargs)
```

- [ ] **Step 4: `/ca.pem` route** in `create_app` (next to the index route), with a failing test first (`GET /ca.pem` -> 200 + PEM body when configured; 404 when not):

```python
    @app.get("/ca.pem")
    def ca_cert() -> FileResponse:
        # Public material: lets each device trust the home CA once, after
        # which the dashboard shows a normal padlock (no warnings).
        if settings.ca_cert and Path(settings.ca_cert).is_file():
            return FileResponse(settings.ca_cert, media_type="application/x-pem-file",
                                filename="watchdoggy-ca.pem")
        raise HTTPException(status_code=404, detail="not set up")
```

- [ ] **Step 5: `scripts/setup-https.sh`** — same shape as `sync-pi-clock.sh` (ssh heredoc, `set -euo pipefail`, usage line). Idempotent: the CA is created ONCE and reused; server certs are re-issued on every run (825 days, Apple's max trusted TLS lifetime) without touching the CA, so devices never need re-onboarding:

```bash
#!/usr/bin/env bash
# Give the Pi's dashboard HTTPS with a household private CA.
# Browsers only allow the microphone (push-to-talk) and notifications on
# secure pages. This creates a "watchdoggy home CA" on the Pi, issues the
# dashboard a certificate signed by it, and serves the CA at /ca.pem so
# each of your devices can trust it once. After that: a normal padlock,
# no warnings, no renewals, no internet needed.
# Usage: ./scripts/setup-https.sh <user@host> [appdir]
set -euo pipefail
TARGET="${1:?usage: setup-https.sh <user@host> [appdir]}"
APPDIR="${2:-doggy}"
ssh "$TARGET" "APPDIR='$APPDIR' bash -s" <<'REMOTE'
set -euo pipefail
cd "$HOME/$APPDIR"
mkdir -p certs && chmod 700 certs
HOST="$(hostname).local"
IP="$(hostname -I | awk '{print $1}')"

if [ ! -f certs/ca.pem ]; then
  echo "==> creating the home CA (once)"
  openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout certs/ca-key.pem -out certs/ca.pem -days 3650 -nodes \
    -subj "/CN=watchdoggy home CA/O=watchdoggy" \
    -addext "basicConstraints=critical,CA:TRUE,pathlen:0" \
    -addext "keyUsage=critical,keyCertSign,cRLSign"
  chmod 600 certs/ca-key.pem
fi

echo "==> issuing the dashboard certificate (825 days, re-run to renew)"
openssl req -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 \
  -keyout certs/key.pem -out certs/req.csr -nodes -subj "/CN=$HOST"
openssl x509 -req -in certs/req.csr -CA certs/ca.pem -CAkey certs/ca-key.pem \
  -CAcreateserial -out certs/cert.pem -days 825 \
  -extfile <(printf "subjectAltName=DNS:%s,IP:%s\nextendedKeyUsage=serverAuth\nbasicConstraints=CA:FALSE\n" "$HOST" "$IP")
rm -f certs/req.csr
chmod 600 certs/key.pem

grep -q '^DOGGY_SSL_CERT=' .env || cat >> .env <<EOF
DOGGY_SSL_CERT=certs/cert.pem
DOGGY_SSL_KEY=certs/key.pem
DOGGY_CA_CERT=certs/ca.pem
EOF
sudo systemctl restart doggy
echo "==> dashboard now at https://$HOST:8000"
echo "    On each device, download https://$HOST:8000/ca.pem (one warning"
echo "    this first time) and trust it:"
echo "      iPhone/iPad: open the file, install the profile, then Settings >"
echo "        General > About > Certificate Trust Settings > enable it"
echo "      Mac: double-click it in Keychain Access, set Trust to Always"
echo "      Android: Settings > Security > Install a certificate > CA"
echo "    After that the padlock is normal everywhere. No renewals needed"
echo "    until $(date -d '+825 days' '+%Y-%m' 2>/dev/null || echo '~2028'); re-run this script then."
REMOTE
```

- [ ] **Step 6: README** — under "Using the dashboard", a short "HTTPS (for push-to-talk and notifications)" section: run the script, install the home CA on each device once (the per-platform steps above, plain words, no em dashes), padlock is normal afterwards; note the CA never leaves your Pi and nothing talks to the internet.
- [ ] **Step 7: Gates, commit** — `feat: https via a household private CA + setup script`.

---

### Task 1b: Onboarding door (auto-detect missing CA)

Decision record: a page cannot install a CA (OS security model), but a
plain-HTTP page CAN probe whether this device trusts the home CA — an
http page may fetch an https URL on the same host, and the fetch fails
with a TLS error iff the CA is untrusted. Port 8000 therefore stays
plain HTTP forever as a "door" (bookmarks keep working); the real
dashboard serves HTTPS on 8443.

**Files:**
- Create: `src/doggy/web/door.py` (tiny FastAPI app: door page, `/ca.pem`, `/ca.mobileconfig`, `/ping`)
- Modify: `src/doggy/core/config.py` (`Settings.ssl_port: int = 8443`), `src/doggy/web/app.py` (`serve` orchestrates both), `scripts/setup-https.sh` (URL wording), `README.md`
- Test: `tests/web/test_door.py` (new)

**Interfaces:**
- Produces: with TLS configured, `serve()` runs the dashboard app via
  uvicorn on `ssl_port` (https) and the door app on `web_port` (http) in
  a daemon thread; without TLS, exactly today (dashboard http on
  `web_port`, no door). Door endpoints: `GET /` (door page), `GET
  /ping` (204, CORS `Access-Control-Allow-Origin: *`), `GET /ca.pem`
  (moves here from the dashboard app), `GET /ca.mobileconfig` (Apple
  profile wrapping the same CA DER, generated with `uuid.uuid5` on the
  cert bytes so re-serving is stable).
- The dashboard app also keeps `/ping` (that is the probe target on the
  https side; 204, no auth) and keeps `/ca.pem` for direct access.

- [ ] **Step 1: Failing tests** (`tests/web/test_door.py`): door page 200
  and contains "Secure this device" + a link to `/ca.pem`; `/ping` 204
  with the CORS header; `/ca.mobileconfig` 200 with
  `application/x-apple-aspen-config` content type and the base64 CA
  payload inside a plist containing `com.apple.security.root`; 404s for
  everything when `ca_cert` unset. Dashboard app: `/ping` 204.
- [ ] **Step 2: Implement door.py.** The door page is a small inline-HTML
  page in the visual language of the dashboard (dark, amber, serif
  wordmark; plain-language copy, no emoji). Its JS probes
  `fetch("https://" + location.hostname + ":" + SSL_PORT + "/ping", {mode: "cors"})`:
  on success, `location.replace` to the https dashboard (same path);
  on failure, reveal the setup section: platform-detected download
  button (`navigator.userAgent` Apple -> `/ca.mobileconfig`, else
  `/ca.pem`), the per-platform trust steps, and a "Check again" button
  that re-probes (plus an automatic re-probe every 5s so finishing the
  OS steps auto-advances). SSL_PORT is templated server-side into the
  page.
- [ ] **Step 3: serve() orchestration.** With TLS configured: build the
  door app and run it with its own `uvicorn.run` in a
  `threading.Thread(daemon=True)` on `web_port`, then run the dashboard
  https on `ssl_port` in the main web thread. Log both URLs at startup.
  A one-line comment: the door is intentionally unauthenticated and
  serves only public material.
- [ ] **Step 4: script + README wording** — the script's final echo now
  says: open `http://<host>:8000` on each device and follow the page
  (it detects setup automatically); direct https URL is
  `https://<host>:8443`.
- [ ] **Step 5: Gates, commit** — `feat: onboarding door auto-detects missing CA and guides setup`.

### Task 2: Arming schedule

**Files:**
- Create: `src/doggy/decision/schedule.py`
- Modify: `src/doggy/core/config.py`, `src/doggy/decision/gate.py`, `src/doggy/pipeline.py`, `src/doggy/core/status.py`, `src/doggy/web/static/index.html`
- Test: `tests/decision/test_schedule.py` (new), `tests/decision/test_gate.py`

**Interfaces:**
- Produces: config `schedule_enabled: bool = False` and `armed_windows: tuple[ArmedWindow, ...] = ()` where `ArmedWindow(BaseModel, frozen=True)` has `days: tuple[int, ...]` (0=Monday), `start: str`, `end: str` ("HH:MM"; `end <= start` wraps past midnight; validated by regex + range). `schedule.armed_state(cfg, wall_now: float) -> tuple[bool, float | None]` returning (armed, seconds_until_next_change; None when schedule disabled or no windows). `FireGate(runtime, wall_clock=time.time)`; both `allow` and `allow_escalation` return False while disarmed. `Status.armed: bool = True`, `Status.next_change_seconds: float | None = None`.

- [ ] **Step 1: Failing tests** (`tests/decision/test_schedule.py`) — pin these with exact datetimes (build epochs via `datetime(2026, 7, 6, 23, 0).timestamp()` etc.; 2026-07-06 is a Monday):

```python
def _cfg(**kw):
    return TunableSettings(schedule_enabled=True, **kw)

WINDOW_NIGHT = {"days": [0, 1, 2, 3, 4], "start": "21:00", "end": "07:00"}

def test_inside_window_is_armed():
    armed, _ = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                           datetime(2026, 7, 6, 23, 0).timestamp())
    assert armed

def test_overnight_wrap_covers_early_morning():
    # Tuesday 03:00 belongs to Monday's 21:00-07:00 window.
    armed, _ = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                           datetime(2026, 7, 7, 3, 0).timestamp())
    assert armed

def test_outside_window_is_off_duty_with_countdown():
    armed, nxt = armed_state(_cfg(armed_windows=[WINDOW_NIGHT]),
                             datetime(2026, 7, 6, 12, 0).timestamp())
    assert not armed
    assert nxt == pytest.approx(9 * 3600)   # 12:00 -> 21:00

def test_schedule_disabled_means_always_armed():
    armed, nxt = armed_state(TunableSettings(), 0.0)
    assert armed and nxt is None

def test_bad_window_times_rejected():
    with pytest.raises(ValidationError):
        TunableSettings(armed_windows=[{"days": [0], "start": "25:00", "end": "07:00"}])
```

Gate tests: with a fake `wall_clock` pinned inside/outside a window, `allow` and `allow_escalation` flip accordingly.

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.**
  - `ArmedWindow` lives in `core/config.py` (pydantic model, `field_validator` for "HH:MM" via `re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v)`, days each in 0..6, non-empty). `armed_windows` parses env JSON strings via the same `mode="before"` JSON trick as the label fields.
  - `decision/schedule.py`:

```python
def _window_active(w, dt):
    minutes = dt.hour * 60 + dt.minute
    start = _to_minutes(w.start)
    end = _to_minutes(w.end)
    if end > start:
        return dt.weekday() in w.days and start <= minutes < end
    # Overnight wrap: the window belongs to its START day.
    if dt.weekday() in w.days and minutes >= start:
        return True
    return (dt.weekday() - 1) % 7 in w.days and minutes < end


def armed_state(cfg, wall_now):
    if not cfg.schedule_enabled or not cfg.armed_windows:
        return True, None
    dt = datetime.fromtimestamp(wall_now)
    armed = any(_window_active(w, dt) for w in cfg.armed_windows)
    return armed, _seconds_to_flip(cfg.armed_windows, dt, armed)
```

  `_seconds_to_flip`: walk forward minute-by-minute is O(20160) worst case — instead evaluate `_window_active` at each window boundary within the next 8 days and take the earliest boundary whose active-state differs; a simple loop over (day, start/end) pairs, fully unit-tested by the countdown test above.
  - Gate: `__init__(self, runtime, wall_clock=time.time)`; both allow methods start with:

```python
        armed, _ = armed_state(cfg, self._wall_clock())
        if not armed:
            return False
```

  (import from `doggy.decision.schedule`; order: safety, schedule, snooze, cap.)
  - Pipeline `run`-loop: each iteration compute `armed, next_change = armed_state(cfg, time.time())` and include `armed=armed, next_change_seconds=next_change` in the per-loop `status.update` call (the one that already writes temp/power).
- [ ] **Step 4: Dashboard.** Settings: an "On a schedule" toggle (`schedule_enabled`, desc: "Only react during the times you pick. It keeps watching around the clock either way.") and a windows editor under it: one row per window — seven day chips (M T W T F S S, multi-toggle), two `<input type="time">` fields, a remove button; an "Add times" button appends a row. Any change patches the full `armed_windows` array. Pill: when `s.armed === false`, force class `state-cooling` styling with text "Off duty", and show "Back on duty in <h/m>" (from `next_change_seconds`) in the snooze label area. Keep it dumb: no timezone math in JS (server already decided).
- [ ] **Step 5: Gates, commit** — `feat: weekly arming schedule (off-duty hours)`.

---

### Task 2b: Zone overlap threshold (stop edge-scrape alarms)

User-reported failure: an animal in an allowed spot whose box merely
scrapes the watch-area boundary fires the alarm, because
`ZoneInclusionFilter.in_zone` passes on ANY mask pixel under the box
(`mask[y1:y2, x1:x2].any()`). Fix: require a minimum FRACTION of the
box's area inside the zone.

**Files:**
- Modify: `src/doggy/vision/filters/zone.py`, `src/doggy/core/config.py`, `src/doggy/web/static/index.html`
- Test: `tests/vision/filters/test_zone.py`

**Interfaces:**
- Produces: `TunableSettings.zone_overlap: float = Field(0.4, ge=0.0, le=1.0)`; `ZoneInclusionFilter.overlap_fraction(box, points, shape) -> float` (mask pixels under the clipped box / clipped-box area; 0.0 for degenerate boxes); `in_zone(box, points, shape, min_overlap)` passes iff `overlap_fraction >= min_overlap` (with `min_overlap == 0.0` preserving today's any-pixel behavior via `> 0` semantics: use `fraction > 0 if min_overlap == 0 else fraction >= min_overlap`); `filter(...)` and `apply(...)` take/read the threshold from cfg.
- Inventory zone-scoping deliberately KEEPS any-overlap (items half-off
  the counter should still count; inventory never alerts) — `apply`
  narrows candidates with the threshold and inventory without it. Comment
  this in the code.

- [ ] **Step 1: Failing tests** (`tests/vision/filters/test_zone.py`): a
  box 30% inside the zone with `zone_overlap=0.4` is excluded from
  candidates; the same box with `zone_overlap=0.2` passes; fraction math
  pinned with a half-in box (`overlap_fraction == pytest.approx(0.5,
  abs=0.02)` on a rectangle straddling a rectangular zone); inventory
  list still uses any-overlap under a high threshold; degenerate box ->
  0.0; `zone_overlap=0` preserves any-pixel semantics.
- [ ] **Step 2: Implement** — `overlap_fraction` uses the existing cached
  mask: `self._mask[y1:y2, x1:x2].sum() / ((x2-x1)*(y2-y1))` after the
  existing clipping; guard zero-area. `apply` reads `cfg.zone_overlap`.
- [ ] **Step 3: Dashboard** — slider in Settings under the watch-area
  hint: label "How much of the animal must be inside the area", desc
  "Stops a box that only scrapes the edge from setting it off.", range
  0-1 step 0.05, percent formatting (add `zone_overlap` to KNOBS and
  FMT: pct).
- [ ] **Step 4: Gates, commit** — `feat: zone overlap threshold (no more edge-scrape alarms)`.

### Task 3: Push-to-talk

**Files:**
- Create: `src/doggy/web/routers/talk.py`
- Modify: `src/doggy/web/app.py` (include router), `src/doggy/web/static/index.html`, `scripts/deploy-to-pi.sh` (dependency note below)
- Test: `tests/web/test_talk.py` (new)

**Interfaces:**
- Produces: `WS /ws/talk` accepting binary frames of 16 kHz mono s16 PCM; one client at a time (second gets close code 1013); frames pipe to `pw-cat --playback --rate 16000 --channels 1 --format s16 -` (found via `shutil.which`; absent -> `log.info` and discard, connection still accepted so the UI works in dev).

- [ ] **Step 0: Dependency check.** FastAPI websockets need `websockets` or `wsproto` in the venv. Run `uv run python -c "import websockets"`. If it fails: `uv add websockets` (pure-Python wheel exists; on the Pi it is vendorable offline per `updating-a-firewalled-uv-python-appliance` — add it to `pyproject.toml` so `uv sync` keeps it, and note it in the deploy script comment listing out-of-band installs). Record which path applied in your report.

- [ ] **Step 1: Failing test** (`tests/web/test_talk.py`) using Starlette's `TestClient.websocket_connect`:

```python
def test_talk_pipes_frames_to_player(monkeypatch):
    written = []

    class FakeProc:
        def __init__(self):
            self.stdin = self
        def write(self, b): written.append(bytes(b))
        def flush(self): pass
        def terminate(self): pass
        def wait(self, timeout=None): pass

    monkeypatch.setattr(talk, "_spawn_player", lambda: FakeProc())
    client = _client()   # existing helper building the app
    with client.websocket_connect("/ws/talk") as ws:
        ws.send_bytes(b"\x01\x02")
    assert written == [b"\x01\x02"]


def test_second_talker_is_rejected(monkeypatch):
    monkeypatch.setattr(talk, "_spawn_player", lambda: None)
    client = _client()
    with client.websocket_connect("/ws/talk"):
        with pytest.raises(WebSocketDisconnect) as exc:
            with _client_ws_second_connection():   # second connect attempt
                pass
    # accept-then-close(1013) is fine; assert the close code
```

(Adapt the second test to Starlette's actual close semantics — accept, then `close(code=1013)`; assert via the raised `WebSocketDisconnect.code`.)

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `talk.py`:

```python
_busy = threading.Lock()

def _spawn_player():
    exe = shutil.which("pw-cat") or shutil.which("pw-play")
    if not exe:
        log.info("push-to-talk: no pw-cat on this host; discarding audio")
        return None
    return subprocess.Popen(
        [exe, "--playback", "--rate", "16000", "--channels", "1",
         "--format", "s16", "-"], stdin=subprocess.PIPE)

def build_router() -> APIRouter:
    router = APIRouter()

    @router.websocket("/ws/talk")
    async def talk(ws: WebSocket) -> None:
        if not _busy.acquire(blocking=False):
            await ws.accept()
            await ws.close(code=1013)   # try again later: someone is talking
            return
        proc = None
        try:
            proc = _spawn_player()      # inside try: a spawn failure must not leak the lock
            await ws.accept()
            while True:
                data = await ws.receive_bytes()
                if proc and proc.stdin:
                    try:
                        proc.stdin.write(data)
                        proc.stdin.flush()
                    except OSError:
                        # Player died mid-stream (e.g. the Bluetooth speaker
                        # dropped): end the session gracefully.
                        break
        except WebSocketDisconnect:
            pass
        finally:
            # Every step guarded: stdin.close() re-flushes and ALSO raises
            # BrokenPipeError on a dead pipe, and nothing here may prevent
            # the lock release (a held lock bricks push-to-talk until restart).
            if proc is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                try:
                    proc.terminate()
                except OSError:
                    pass
            _busy.release()

    return router
```

(Post-review correction: the original snippet here leaked `_busy` when the
player died mid-stream or spawn failed — found Critical in review; this is
the fixed shape.)

- [ ] **Step 4: Dashboard.** Monitor card `btnrow` gains a hold-button: `<button id="ptt">Hold to talk</button>`. JS: on pointerdown — `getUserMedia({audio: {channelCount: 1, echoCancellation: true}})`, `new AudioContext()`, a `ScriptProcessorNode(4096, 1, 1)` (deprecated but dependency-free and fine for an appliance) whose `onaudioprocess` downsamples `inputBuffer.getChannelData(0)` from `ctx.sampleRate` to 16000 by index-stepping, converts to Int16Array, and `ws.send(int16.buffer)` over `new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/talk")`. On pointerup/pointercancel/pointerleave: stop tracks, close context and socket. While held: button text "Talking...", lamp-colored border. If `getUserMedia` throws (plain http on the Pi): `alert("The microphone needs the https address. Run scripts/setup-https.sh and use https://... instead.")`. Keep all of it inside one `setupTalk()` function.
- [ ] **Step 5: Gates, commit** — `feat: push-to-talk from the dashboard to the speaker`.

---

### Task 4: Browser notifications

**Files:**
- Modify: `src/doggy/web/static/index.html` only
- Test: `tests/web/test_api.py` (string-presence only: the toggle label ships)

- [ ] **Step 1:** Settings card, under "Save video clips": a "Notify this device" toggle (NOT bound to server settings — it stores per-device consent in `localStorage.doggy_notify`). Desc: "Pops a notification on this device when something is caught. Works while the dashboard is open."
- [ ] **Step 2: JS.** On toggle-on: `Notification.requestPermission()`; if not granted, revert the toggle; if `!window.isSecureContext`, revert and set the desc line to "Needs the https address (run scripts/setup-https.sh)." In `loadEvents()`, remember the newest event id; when a NEW id appears (and it isn't the first load), permission is granted, and the toggle is on: `new Notification(word + " on the counter", {body: pct(e.confidence) + " sure" + (e.taken && e.taken.length ? " · took the " + e.taken.join(", ") : ""), icon: "/events/" + e.thumb})` where `word` is the alert-class noun from `targetNoun` (Plan 1). Tag notifications (`tag: "watchdoggy-catch"`) so bursts collapse.
- [ ] **Step 3:** Gates (test asserts "Notify this device" in html), commit — `feat: browser notifications for new catches`.

---

### Task 5: Kiosk fullscreen + export zip

**Files:**
- Modify: `src/doggy/web/static/index.html`, `src/doggy/web/routers/events.py`
- Test: `tests/web/test_api.py`

- [ ] **Step 1: Failing test** for export:

```python
def test_export_returns_zip_with_events(tmp_path, ...):
    # seed store with one event (existing helper), GET /api/export
    r = c.get("/api/export")
    assert r.status_code == 200
    z = zipfile.ZipFile(io.BytesIO(r.content))
    names = z.namelist()
    assert "events.jsonl" in names and any(n.endswith(".jpg") for n in names)
```

- [ ] **Step 2: Implement** in `events.py`:

```python
    @router.get("/api/export")
    def api_export() -> Response:
        records = event_store.list()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
            jsonl = Path(settings.event_log_dir) / "events.jsonl"
            if jsonl.is_file():
                z.write(jsonl, "events.jsonl")
            for r in records:
                for name in (r.thumb, r.clip):
                    p = Path(settings.event_log_dir) / name if name else None
                    if p and p.is_file():
                        z.write(p, name)
        return Response(buf.getvalue(), media_type="application/zip", headers={
            "Content-Disposition": "attachment; filename=watchdoggy-export.zip"})
```

(ZIP_STORED: the JPEGs/WebPs are already compressed. Sizes here are tens of MB at worst — buffered bytes are fine on the Pi's RAM; do not add streaming machinery.)
- [ ] **Step 3: Kiosk.** An OSD-styled "Fullscreen" button in the monitor card btnrow: `document.getElementById("vidwrap").requestFullscreen()`. CSS: `#vidwrap:fullscreen #live{width:100%;height:100%;object-fit:contain}` and `#vidwrap:fullscreen{border:none;border-radius:0}`. The OSD elements already live inside `#vidwrap`, so the camera label, clock, and telemetry ride along for free. Esc exits natively.
- [ ] **Step 4: Dashboard button** for export in the Catch log `card-head` (a `.linkbtn` "Export all" as `<a href="/api/export" download>`).
- [ ] **Step 5: Gates, commit** — `feat: kiosk fullscreen + export-all zip`.

---

## Deploy note

Task 1's script is run once by the user (`./scripts/setup-https.sh doggy@doggypi.local`). If Task 3 added the `websockets` dependency, the deploy is NOT code-only: follow `updating-a-firewalled-uv-python-appliance` (vendor the pure-Python wheel or open egress briefly). Otherwise rsync + restart as usual. After deploy verify: https loads (after accepting the warning), PTT plays through the JBL, a test catch pops a notification, schedule "Off duty" pill appears when a window excludes now.

## Self-review notes

- PTT keeps FastAPI's async loop unblocked except `proc.stdin.write` (small frames, local pipe: acceptable; documented in code comment).
- Export reads only under the store's public `list()`; file reads race deletes at worst into a skipped file (is_file check) — no lock held during zip build, deliberately.
- Schedule math is pure and fully test-pinned; the gate stays the single fire-decision point.
