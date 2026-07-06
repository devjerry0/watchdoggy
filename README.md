# watchdoggy

**Counter Watch** — a ~$62 Raspberry Pi appliance that watches your kitchen counter, detects when the dog jumps up, and plays a deterrent sound. It only reacts inside an area you draw, ignores people, and self-regulates its temperature.

> **100% local. No cloud, no accounts, no internet.** All detection runs on the device. After a one-time model download at setup, it runs fully disconnected — and it's firewalled to your LAN so it *cannot* phone home. Your camera feed never leaves your network.

![Counter Watch dashboard](docs/dashboard.png)

## What it does

- **On-device dog detection** — YOLO running locally (NCNN) on the Pi's CPU. No cloud vision API.
- **Watch area** — only acts on dogs inside a zone you draw on the live view by tapping corners.
- **Ignores people** — suppresses a person misclassified as a dog, so it won't false-alarm on you.
- **Deterrent** — plays a sound through a Bluetooth/USB speaker, with randomized cooldowns and an hourly cap.
- **Adaptive thermal governor** — scales its work rate with CPU temperature so a fanless Pi never throttles.
- **Plain-language dashboard** — live view plus simple settings, served on your LAN (shown above).

## Fully local & self-contained

This is deliberately an offline appliance — privacy and reliability by design:

- **All inference is on-device.** No cloud, no external API, no account, no subscription.
- **No internet needed to run.** The YOLO model is downloaded once during setup; after that the appliance runs completely disconnected.
- **Firewalled to the LAN.** [`scripts/harden-pi.sh`](scripts/harden-pi.sh) installs an nftables egress firewall that blocks all outbound traffic except your local network. It literally cannot reach the internet.
- **Your video never leaves the device.** The dashboard is served only on your LAN; nothing is uploaded, stored remotely, or shared.
- **No telemetry.** Ultralytics analytics are disabled during setup.

## Hardware (~$62)

| Part | Price |
|---|---|
| Raspberry Pi 4 Model B | $35 |
| Aluminum heatsink case | $12 |
| 1080p USB webcam | $15 |
| **Total** | **~$62** |

Plus any Bluetooth or USB speaker you already have for the deterrent (this build uses a JBL Go).

## How it works

```
USB webcam → capture thread → YOLO (NCNN, on-CPU)
           → watch-area filter → person suppression
           → M-of-N + confirm-timer trigger → safety limits (cooldown, hourly cap)
           → deterrent sound
```

A FastAPI app streams the annotated view (MJPEG) and exposes the live-tunable settings. An adaptive governor reads the CPU temperature each loop and paces detection to hold the board below its throttle point.

## Quick start (dev, on a Mac)

```sh
uv sync
cp .env.example .env          # set DOGGY_CAMERA_INDEX for your webcam
uv run yolo export model=yolo26n.pt format=ncnn   # downloads yolo26n.pt
# drop at least one sound clip into sounds/
uv run doggy                  # dashboard at http://127.0.0.1:8000
```

Grant your terminal camera permission (System Settings → Privacy → Camera), or OpenCV returns empty frames silently.

## Deploy to a Raspberry Pi

```sh
./scripts/deploy-to-pi.sh <user@host>
```

Syncs the code, installs dependencies with `uv`, downloads and NCNN-exports the model for ARM, writes a Pi `.env`, and installs a systemd service that runs on boot.

Optional extras:

- [`scripts/setup-bt-speaker.sh`](scripts/setup-bt-speaker.sh) — Bluetooth speaker with hands-free auto-reconnect (PipeWire).
- [`scripts/harden-pi.sh`](scripts/harden-pi.sh) — lock it down: LAN-only egress firewall + key-only SSH.

## Using the dashboard

Open `http://<pi-host>:8000` from any device on your network.

- Status pill shows **Watching / Dog spotted / Cooling down**.
- **Draw the watch area** by tapping corners around the counter on the live view, then **Save area**.
- Simple settings: how sure it must be it's a dog, how long the dog must linger, wait between reactions, hourly cap, and **Ignore people**.
- **Advanced** holds the detection-window and person-matching knobs; **System** shows temperature, power, and processing speed.
- **Test sound** plays the deterrent; **Save settings** persists to the Pi's `.env`.

## Configuration

All config is set via `DOGGY_*` environment variables (see `.env.example`). Live-tunable params are also editable from the dashboard; structural params (camera, model, audio backend) require a restart.

> The CLI and Python package are named `doggy` (env prefix `DOGGY_`); the repository was renamed from `doggy` → `watchdoggy`.

## Tests

```sh
uv run pytest -m "not slow"    # fast suite, no hardware or weights
uv run pytest -m slow          # detector test (needs the model + fixtures)
```

## License

AGPL-3.0-or-later (matches YOLO26n, which is AGPL).
