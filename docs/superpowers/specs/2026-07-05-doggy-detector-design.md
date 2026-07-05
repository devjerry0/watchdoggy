# Doggy Detector — Design Spec

**Date:** 2026-07-05
**Status:** Approved (design), pending implementation plan
**Repo:** https://github.com/devjerry0/doggy

## 1. Goal

Detect a dog in a camera view and play a pre-recorded sound clip on a speaker as
a deterrent (the motivating use case is a dog counter-surfing in the kitchen).
The system runs **fully locally** — no cloud / OpenRouter inference.

- **Develop first** on an Apple Silicon Mac (macOS, arm64).
- **Deploy later** to a Raspberry Pi 5 with a **USB webcam** and a wired USB/aux speaker.

## 2. Scope of v1

- **Trigger:** fire on **any confirmed dog in view**. v1 is honestly a
  *dog-presence detector*, not yet a counter-zone detector.
- **Output:** play a pre-recorded sound clip (randomized from a folder).
- **Local web UI:** a stupid-simple single-page dashboard on `localhost` showing
  the live annotated video, current status/events, and **live-tunable knobs** for
  the tunable params — so you can watch detections and tune thresholds without
  restarting. Localhost-bound, no auth. See §7.
- **Language:** Python first. Rust is deferred — only revisit if the Pi hot loop
  proves too slow, and only for the capture/inference loop.

**Explicit non-goals for v1 (documented upgrade path, not built now):**
- Counter-zone / ROI targeting (fire only when the dog overlaps an off-limits
  region). **This is the very next milestone** — the detector already returns
  bounding boxes, so it is added as pure logic on the same output.
- Pose detection ("paws on counter").
- Pi CSI ribbon-camera support (v1 requires a USB webcam — see §7).
- Any vision-LLM / natural-language reasoning about the scene.

## 3. Model & license decision

**Chosen model: YOLO26n (Ultralytics, released Jan 2026).**
- Newest nano-class detector; NMS-free / end-to-end; ~40.9 COCO mAP; ~2.4M params.
- `dog` is COCO class 16 — detected out of the box; no fine-tuning needed for v1.
- Speed on Pi 5 CPU: ~15 FPS via NCNN export (vs ~2.8 FPS raw PyTorch). Ample for
  a lingering dog. On the Mac, runs via the same Ultralytics API (CPU or MPS).
- **License: AGPL-3.0.** The project repo will be **relicensed from Apache-2.0 to
  AGPL-3.0** to match (the user owns the repo and accepts AGPL). AGPL's network-
  service clause does not trigger for a local device; the only practical effect is
  that the whole project must remain AGPL if distributed.

**Swappability:** `detector.py` exposes a single backend interface, so the model
can be replaced later (e.g. a permissively-licensed RF-DETR-Nano / NanoDet /
YOLOX, or a model fine-tuned on the user's own kitchen footage) without touching
the rest of the pipeline.

**Rejected alternatives:**
- Vision LLM (moondream/llava/OpenRouter): seconds per frame, heavy on Pi,
  contradicts local+real-time. Overkill for a yes/no.
- TFLite whole-frame classifier: no localization, no upgrade path to zones/pose.
- Permissive-license models were offered (RF-DETR-Nano Apache-2.0, ~1 FPS on Pi;
  NanoDet/YOLOX Apache-2.0, ~10 FPS) but the user opted for the best/easiest model
  and relicensing the repo.

## 4. Architecture

Three cooperating threads sharing a single stop-event, wired by `main.py`:

```
 ┌──────────────┐  latest    ┌──────────────┐  Detection[]  ┌──────────────┐
 │ Capture      │  frame     │ Detect       │  (dog only)   │ TriggerLogic │
 │ thread       │──────────▶ │ thread       │─────────────▶ │ + Safety     │
 │ (keep newest │  (drop     │ YOLO26n →    │               │ state machine│
 │  frame only) │   stale)   │ filter 'dog' │               └──────┬───────┘
 └──────────────┘            └──────────────┘                      │ fire!
                                                                    ▼
                                                            ┌──────────────┐
                                                            │ Alert thread │
                                                            │ (non-blocking│
                                                            │  playback)   │
                                                            └──────────────┘
```

**Why three threads:** YOLO inference takes 100–500ms. In a single thread the
OpenCV capture buffer fills during inference and hands back increasingly stale
frames, so the dog is detected seconds after it left. The capture thread keeps
only the newest frame (`queue.Queue(maxsize=1)`, drop-oldest). Sound playback must
not block detection, so it runs on its own fire-and-forget worker. This is the
right amount of concurrency for the pipeline — no multiprocessing / async
framework in the hot loop. The optional web dashboard (§7) runs a uvicorn server
on a **4th thread** that only *reads* shared state, so it never blocks detection.

**Shared state between threads** (all thread-safe): a **latest-raw-frame** slot
(capture → detect), a **latest-annotated-frame** slot (detect draws boxes → web
MJPEG), a thread-safe **`RuntimeSettings`** holder wrapping the current validated
`TunableSettings` (seeded from `Settings` at boot; read every loop by
detect/trigger/alerter; the web `PATCH` validates input into a new
`TunableSettings` and swaps it in atomically), and a **status/events**
snapshot (state, FPS, fires-this-hour, last-fire thumbnail) the web UI polls.

### Modules

| Module | Responsibility | Interface (sketch) | Notes |
|--------|---------------|--------------------|-------|
| `camera.py` | Frame source (factory) | `Camera.frames() -> Iterator[np.ndarray]`, `open()`/`close()` | `OpenCVCamera` (USB webcam via `cv2.VideoCapture`, works on Mac + Pi) and `FakeCamera` (yields frames from a video file / image folder). Bounded reconnect-with-backoff; surfaces a clear signal after M failures. |
| `detector.py` | Frame → dog detections | `Detector.detect(frame) -> list[Detection]` | Wraps Ultralytics YOLO26n. Loads NCNN-exported model on Pi, torch/MPS on Mac — device auto-selected, never hardcode `mps`. Filters COCO output to `dog` here. |
| `trigger.py` | Decide when to fire | `TriggerLogic(clock, cfg).update(detections) -> bool` | Stateful class, **not** a pure function. Explicit state machine (below). Monotonic clock injected for deterministic tests. Time-based M-of-N confirmation. |
| `safety.py` | Guardrails around firing | `SafetyGovernor.allow_fire() -> bool`, `record_fire(frame)` | Rate limit (max N fires/hour → auto-mute + log), master off switch, event log with saved thumbnail + timestamp + confidence, volume cap. |
| `alerter.py` | Play a clip | `Alerter.alert()` | `SoundDeviceAlerter` (`sounddevice`+`soundfile`, CoreAudio on Mac / ALSA on Pi). Picks a **random** clip from a folder (anti-habituation). `CommandAlerter` fallback shells to `afplay`/`aplay`. `FakeAlerter` logs instead of playing. Async / non-blocking. |
| `config.py` | Load + validate settings | `Settings(BaseSettings)` from `pydantic-settings` | Reads `DOGGY_*` env vars + `.env` natively (env_prefix `DOGGY_`, case-insensitive). Pydantic validators enforce the §6 rules and fail fast at boot. A nested `TunableSettings` submodel is the live-tunable subset — reused directly as the FastAPI `PATCH /api/settings` body/response schema, so the same validation runs at boot and at runtime. |
| `web.py` | Local dashboard server | FastAPI app (`GET /`, `GET /stream.mjpg`, `GET /api/status`, `PATCH /api/settings`, `POST /api/test-sound`, `POST /api/settings/save`) | Runs on its own thread (uvicorn). Reads shared state (latest annotated frame, `RuntimeSettings`, status/events); patches `RuntimeSettings` on knob changes. MJPEG-encodes only when a client is connected, throttled/downscaled so it never starves detection. Serves one static `index.html` (vanilla JS, polls `/api/status`). Optional (`DOGGY_WEB_ENABLED`). |
| `main.py` | Orchestration | — | Loads `Settings` + model once at startup (fail fast), builds the shared `RuntimeSettings` holder from the boot `Settings`, starts the capture/detect/alert threads + (optional) web thread, installs SIGINT/SIGTERM handler for graceful shutdown (release camera, stop audio, stop server, join threads), owns top-level logging. |

### Data type

```python
@dataclass(frozen=True)
class Detection:
    label: str            # e.g. "dog"
    confidence: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2)
```

### Trigger state machine

```
IDLE        --dog seen-->                    CONFIRMING (start window timer)
CONFIRMING  --dog present >= X seconds
             AND M-of-N frames had a dog-->  FIRED  (emit alert, if Safety allows)
CONFIRMING  --no dog / conf below thr-->     IDLE   (reset window)
FIRED       --(same tick)-->                 COOLDOWN (deadline = now + W ± jitter)
COOLDOWN    --now < deadline-->              COOLDOWN (ignore dogs)
COOLDOWN    --now >= deadline-->             IDLE   (require fresh confirmation)
```

- **Time-based, not frame-count:** confirmation is "dog present for ≥ X seconds",
  so behavior is identical whether the Mac runs at 30 FPS or the Pi at 4 FPS.
- **M-of-N sliding window** (e.g. dog in ≥4 of last 6 evaluations) tolerates a
  single dropped/flickery frame instead of resetting on a strict break.
- **Jittered cooldown** (e.g. 12–20s) + randomized clip reduce habituation.
- After cooldown expires with the dog still present, a **fresh confirmation
  window** is required before re-firing (no machine-gunning a stationary dog).

## 5. Safety envelope (unattended aversive device)

An automated sound machine firing at an animal while the owner is away needs
guardrails. `safety.py` enforces:
- **Rate limit:** max N fires per rolling hour (default 6). On exceed → auto-mute
  and log; do not keep firing.
- **Event log:** every fire records timestamp, confidence, and a saved thumbnail
  so the owner can audit what actually triggered it (dog? cat? guest? a photo?).
- **Master off switch:** a simple config flag / file the owner can flip to disable
  firing while leaving detection + logging running.
- **Volume cap:** clip playback volume is bounded in config.

Known v1 limitations to document in the README:
- **Cat / photo / TV false positives** and **low-light/night** misses are possible.
  The event log + thumbnails make these visible; the counter-ROI milestone and
  optional cat-vs-dog confidence margin are the mitigations. Static-box
  suppression (a "dog" box unmoving for a long time → treat as furniture/picture)
  is a candidate follow-up.

## 6. Configuration (environment variables)

**All parameters are environment-variable configurable** — env vars are the single
source of truth (12-factor style). `config.py` uses **`pydantic-settings`
(`BaseSettings`)** with `env_prefix="DOGGY_"` to read them into a validated
`Settings` model at startup, applying the defaults below when a var is unset and
failing fast (Pydantic `ValidationError`) on invalid values. `BaseSettings`
auto-loads a `.env` file in the project root for local dev; on the Pi the systemd
unit supplies them with `EnvironmentFile=`. No YAML/JSON config file — env only,
so there is exactly one config mechanism. The **live-tunable** subset is a nested
`TunableSettings` model reused as the FastAPI knob schema (§7), so one definition
drives boot defaults, boot validation, and runtime knob validation.

All vars use the `DOGGY_` prefix. List-valued params are split into scalar vars
(env values are strings).

| Env var | Default | Meaning |
|---------|---------|---------|
| `DOGGY_CAMERA_BACKEND` | `opencv` | `opencv` (USB webcam) or `file` (FakeCamera for dev/CI) |
| `DOGGY_CAMERA_INDEX` | `0` | Webcam index (opencv backend) |
| `DOGGY_CAMERA_PATH` | *(unset)* | Video/image path (file backend) |
| `DOGGY_MODEL_PATH` | `models/yolo26n.pt` | `.pt` on Mac; NCNN export dir on Pi |
| `DOGGY_CONFIDENCE` | `0.55` | Min `dog` confidence; tune empirically (nano conf is noisy) |
| `DOGGY_CONFIRM_SECONDS` | `1.2` | Dog must be present this long before firing |
| `DOGGY_WINDOW_M` | `4` | M of… |
| `DOGGY_WINDOW_N` | `6` | …N recent evaluations must have a dog (flicker tolerance) |
| `DOGGY_COOLDOWN_MIN_SECONDS` | `12` | Cooldown lower bound (jittered) |
| `DOGGY_COOLDOWN_MAX_SECONDS` | `20` | Cooldown upper bound (jittered) |
| `DOGGY_ALERTER_BACKEND` | `sounddevice` | `sounddevice` \| `command` \| `log` (FakeAlerter) |
| `DOGGY_CLIPS_DIR` | `sounds/` | Folder of clips; one chosen at random per fire |
| `DOGGY_AUDIO_DEVICE` | *(unset)* | Optional output device name (Pi: pick USB sink) |
| `DOGGY_MAX_VOLUME` | `0.8` | Playback volume cap (0.0–1.0) |
| `DOGGY_SAFETY_ENABLED` | `true` | Master off switch |
| `DOGGY_MAX_FIRES_PER_HOUR` | `6` | Rate limit → auto-mute + log on exceed |
| `DOGGY_EVENT_LOG_DIR` | `events/` | Thumbnails + jsonl event log |
| `DOGGY_LOG_LEVEL` | `INFO` | Python logging level |
| `DOGGY_WEB_ENABLED` | `true` | Run the local dashboard (§7) |
| `DOGGY_WEB_HOST` | `127.0.0.1` | Bind address (set `0.0.0.0` to view from your phone on the LAN — no auth, so localhost by default) |
| `DOGGY_WEB_PORT` | `8000` | Dashboard port |

**Live-tunable vs restart-required.** Env vars set the *boot* values. The web UI
can change the **live-tunable** subset at runtime (in-memory, via `RuntimeSettings`)
— it takes effect on the next loop, no restart: `CONFIDENCE`, `CONFIRM_SECONDS`,
`WINDOW_M/N`, `COOLDOWN_MIN/MAX_SECONDS`, `MAX_VOLUME`, `SAFETY_ENABLED`,
`MAX_FIRES_PER_HOUR`, `CLIPS_DIR`, `LOG_LEVEL`. **Restart-required** (structural —
shown read-only in the UI): camera backend/index/path, `MODEL_PATH`, alerter
backend/audio device, web host/port. Live changes are session-only unless saved:
`POST /api/settings/save` writes the current tunable values back to `.env`.

Validation rules (fail fast at startup): `WINDOW_M <= WINDOW_N`;
`COOLDOWN_MIN <= COOLDOWN_MAX`; `0 <= CONFIDENCE <= 1`; `0 <= MAX_VOLUME <= 1`;
`CLIPS_DIR` non-empty and exists; `MODEL_PATH` exists; `CAMERA_PATH` set and exists
when backend is `file`.

## 7. Local web UI

A deliberately minimal localhost dashboard for watching detection and tuning knobs
live. **One static `index.html`** (vanilla JS, no build step, no framework) served
by a small **FastAPI + uvicorn** app running on its own thread. Interaction is
plain REST + client polling — **no WebSocket** — to keep it simple.

**Endpoints (`web.py`):**
- `GET /` → the static dashboard page.
- `GET /stream.mjpg` → MJPEG stream of the latest **annotated** frame (boxes +
  labels drawn by the detect thread). Encoded only while a client is connected,
  throttled (e.g. ≤10 FPS) and downscaled so it never starves detection.
- `GET /api/status` → JSON snapshot the page polls (~2 Hz): trigger state
  (IDLE/CONFIRMING/COOLDOWN), FPS, current dog confidence, fires-this-hour,
  last-fire timestamp + thumbnail URL, current settings, muted?.
- `PATCH /api/settings` → update live-tunable knobs (validated; applied on the next
  loop via `RuntimeSettings`). Restart-required fields are rejected with a clear
  message.
- `POST /api/test-sound` → play a random clip now (verify audio without a dog).
- `POST /api/settings/save` → persist current tunable values back to `.env`.

**The page shows:**
- The live annotated video (`<img src="/stream.mjpg">`).
- A status strip: state, FPS, confidence, fires-this-hour, muted indicator.
- **Knobs** for every live-tunable param (sliders/number inputs) that PATCH on
  change; restart-required params shown read-only/greyed with a note.
- A master **enable/off** toggle (`SAFETY_ENABLED`), a **Test sound** button, and a
  **Save to .env** button.
- A short recent-events list (from the safety event log) with thumbnails.

**Scope guardrails:** localhost-bound by default (`DOGGY_WEB_HOST=127.0.0.1`), no
auth, no accounts — it is a personal debug/tuning panel, not a product surface.
The whole pipeline runs fine headless with `DOGGY_WEB_ENABLED=false`.

## 8. Platform portability notes

- **Camera:** `cv2.VideoCapture` works with USB webcams on both Mac and Pi, but
  **cannot see the Pi 5 CSI ribbon camera** (OpenCV has no libcamera backend).
  **v1 requires a USB webcam.** The dev camera is a **Logitech C922 Pro Stream
  Webcam** (UVC, VendorID `0x046d` / ProductID `0x085c`) confirmed connected to the
  Mac — a standard UVC device that runs the identical code path on the Pi 5 (plug
  it straight in). The Mac also exposes the built-in FaceTime camera, so the C922
  is a non-default index: `DOGGY_CAMERA_INDEX` selects it (built-in is usually `0`,
  the C922 typically `1` — verify on first run). A `Picamera2Camera` backend can be
  added behind the camera factory later (requires a `--system-site-packages` venv
  for the apt-only `picamera2`, so it is deliberately out of v1).
- **Audio:** `sounddevice`+`soundfile` map to CoreAudio (Mac) and ALSA (Pi) with
  the same code. On a fresh Pi 5 (no 3.5mm jack) audio defaults to HDMI — the
  README will note selecting a USB speaker as the default sink and optionally
  naming the device in config. Wired USB speaker recommended over Bluetooth
  (latency) for v1.
- **macOS gotcha:** the terminal/IDE needs camera (TCC) permission or
  `cv2.VideoCapture` returns empty frames silently — documented in setup.
- **Device selection:** auto-detect (MPS on Mac, CPU/NCNN on Pi); never hardcode
  `mps`.

## 9. Tooling & dependencies

- **`uv` + `pyproject.toml`** (no `requirements.txt`).
- **`pydantic-settings`** for config: reads `DOGGY_*` env vars + `.env`, validates,
  and the models double as the FastAPI knob schemas (env vars are the config source
  of truth — see §6). No separate `python-dotenv`.
- **`fastapi` + `uvicorn`** for the localhost dashboard (§7); the frontend is a
  single static `index.html` with vanilla JS (no Node/build step).
- Torch/Ultralytics wheels differ between macOS-arm64 and Pi-aarch64-Linux, so
  platform-specific handling is required (per-platform locks / install notes; use
  Ultralytics' documented Pi install path for a known-good torch/torchvision pair).
- **CPU-only PyTorch (no CUDA).** `torch`/`torchvision` are pinned to the CPU wheel
  index (`download.pytorch.org/whl/cpu`) on Linux via `[tool.uv.sources]` so CI /
  x86_64 boxes don't pull the ~2.5GB CUDA build; macOS arm64 uses the default PyPI
  wheel (already CPU+MPS, no CUDA variant). Inference device is auto-selected
  (MPS/CPU), never CUDA on these targets.
- Use pip/uv `opencv-python` (or `-headless` on Pi); do **not** mix with apt's
  `python3-opencv`.
- On the Pi, export the model to **NCNN** once for speed; run inference via the
  Ultralytics API. (A torch-free raw-NCNN path is a possible later optimization if
  install size/RAM becomes a problem.)

## 10. Testing strategy

- **Pure unit tests** (fast, no hardware, no model): `trigger.py` (feed synthetic
  detection sequences + a fake monotonic clock, assert fire/no-fire timing across
  the state machine), `config.py` (validation), `safety.py` (rate limit, off
  switch, log).
- **Detector integration test:** run `detector.detect()` against a saved dog image
  (expect a dog) and a saved empty-room image (expect none).
- **End-to-end on Mac (no live dog):** `FakeCamera` streams a recorded video
  (`dog_walk.mp4`) + `FakeAlerter` logs → run `main.py` with a dev config and
  assert when it *would* fire. Record fixture clips: dog present, empty room, cat,
  low light — used as regression fixtures.
- **Web API tests:** FastAPI `TestClient` — `PATCH /api/settings` updates
  `RuntimeSettings` and is reflected in `GET /api/status`; restart-required fields
  are rejected; invalid values are rejected by validation. No browser test needed
  for v1 (the page is trivial static JS).

## 11. Deployment

- Dev on Mac (built-in/USB webcam + speakers).
- Pi: `uv` install per platform notes, USB webcam + USB speaker, NCNN model export.
- A **systemd unit** runs it headless as a service (auto-start on boot,
  `Restart=on-failure`) — the natural counterpart to headless logging + graceful
  SIGTERM shutdown.

## 12. Open follow-ups (post-v1, in rough priority order)

1. Counter-zone ROI targeting (fire only when the dog overlaps a drawn region).
2. Cat-vs-dog confidence margin + static-box suppression for false positives.
3. CSI camera (`Picamera2Camera`) support.
4. Fine-tune on the user's own kitchen footage to cut site-specific false positives.
5. Rust rewrite of the capture/inference loop *only if* Pi performance demands it.
