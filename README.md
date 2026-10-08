# watchdoggy

A Raspberry Pi that keeps a dog off the kitchen counter. A USB webcam watches the
counter, a YOLO model running on the Pi's CPU looks for the dog, and when the dog
steps into an area you've marked, a speaker plays a deterrent sound.

Live detection runs entirely on the Pi, and the video stream never leaves your
network. Cloud training is optional. If you turn it on, the camera frames the Pi
saves for training are uploaded to your own [Modal](https://modal.com) account,
where the model is retrained.

![Dashboard](docs/dashboard.png)

## Features

- On-device dog detection with YOLO26n exported to NCNN. No cloud vision API.
- Fires only for dogs inside a watch area you draw on the live view.
- Ignores people, including a person bending over, which stock models often
  label as a dog.
- Plays the alarm through a Bluetooth or USB speaker. Cooldowns and an hourly
  cap limit how often it fires.
- Keeps a catch log with a snapshot and a short clip of every alarm. You can
  mark a false alarm as "Not a dog" and it becomes training data.
- Can play your own music or white noise while it watches. The alarm
  interrupts it.
- Dashboard on your LAN with pages for live view, labeling, and training.

With the optional `trainer` user set up (`scripts/setup-pi-trainer.sh`):

- **Self-training.** The Pi saves hard frames and uploads them to your Modal
  account. There they're auto-labeled overnight and used to retrain the model
  on a schedule. A new model is installed only if it beats the current one on a
  held-out set of human-verified frames.
- **Self-updates.** The Pi checks for new GitHub releases once a day, installs
  them, and rolls back if the detector doesn't come back up healthy.

### Results

Stock COCO weights caught 15 of 61 dog moments in this kitchen, with 5 false
alarms. The current fine-tuned model catches 113 of 127 with no false alarms
across 215 dog-free frames. That's on a later, larger held-out set, scored at
this appliance's 0.6 threshold using the exported NCNN model the Pi actually
runs.

## How it works

```
USB webcam -> capture thread -> YOLO (NCNN, separate inference process)
           -> watch-area filter -> person suppression -> oversize filter
           -> M-of-N window + confirm timer -> cooldown / hourly cap
           -> sound + catch log + clip + training-frame capture
```

A FastAPI app serves the dashboard, streams the annotated video over MJPEG, and
exposes the settings you can change live. A thermal governor slows detection as
the CPU heats up, so a fanless Pi doesn't throttle. Training runs as a separate
`trainer` user, the only user allowed through the firewall. It picks up job files
written by the dashboard or by its schedule, runs them on Modal, and installs the
result only if it passes the gate.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the code layout.

## Hardware

| Part | Approx. price |
|---|---|
| Raspberry Pi 4 Model B | $35 |
| Aluminum heatsink case (passive) | $12 |
| 1080p USB webcam (e.g. Logitech C922) | $15 |

You also need a Bluetooth or USB speaker (this build uses a JBL Go) and a 5V/3A
power supply. Cloud training is optional and needs a Modal account. A training
run costs well under $2.

## Getting started

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

### Run locally (macOS or Linux)

```sh
git clone https://github.com/devjerry0/watchdoggy.git
cd watchdoggy
uv sync
cp .env.example .env    # set DOGGY_CAMERA_INDEX to your webcam
uv run doggy            # dashboard at http://127.0.0.1:8000
```

The YOLO26n weights download to `models/` on first run. One default alarm sound
ships in `sounds/`. You can add more there; see [sounds/README.md](sounds/README.md).

On macOS, give your terminal camera access (System Settings > Privacy &
Security > Camera). Without it, OpenCV returns empty frames and doesn't report
an error.

### Deploy to a Raspberry Pi

```sh
./scripts/deploy-to-pi.sh doggy@doggypi.local
```

This syncs the code, installs dependencies, exports the model to NCNN for ARM,
writes the Pi's `.env`, and installs a systemd service that starts on boot. After
that the dashboard is at `http://doggypi.local:8000`.

For the full setup, see [docs/pi/README.md](docs/pi/README.md). It covers
flashing the SD card, Bluetooth speaker auto-reconnect, the egress firewall,
HTTPS, and enabling cloud training.

## Configuration

`.env` holds structural settings only: camera, model path, audio backend, ports,
and TLS. See `.env.example`; changes need a restart.

Everything you can adjust from the dashboard is saved to `settings.json` and takes
effect immediately. Every change is logged to `settings-changes.jsonl`.

## Privacy

**Live detection stays local. Optional cloud training sends saved camera frames
to your own cloud account.**

- The detector service has no internet access, enforced by an nftables egress
  firewall (`scripts/harden-pi.sh`). The live video stream is only served on
  your LAN.
- Cloud training is opt-in. When enabled, every frame the Pi saves for training
  is uploaded to your Modal volume. These are still frames, not video, and can
  include people in the kitchen. Only the separate `trainer` user can reach the
  internet, and only Modal and the GitHub API (for release checks and the
  clock).
- If you skip the trainer setup, nothing leaves the Pi. You can still train
  from a workstation with `scripts/train_kitchen_model.py`.

See [step 6 of the Pi guide](docs/pi/README.md#6-cloud-training-and-self-updates-optional)
for details and how to turn it off.

## Documentation

- [MANUAL.md](MANUAL.md): using the dashboard, how self-training works, the
  training console, HTTPS, and self-updates.
- [ARCHITECTURE.md](ARCHITECTURE.md): code layout, design patterns, and invariants.
- [docs/pi/README.md](docs/pi/README.md): Raspberry Pi setup.

## Development

```sh
uv run pytest -m "not slow"   # fast suite, no camera or model weights needed
uv run pytest -m slow         # detector tests; need the model and local photo fixtures
```

## License

The code is MIT licensed. See [LICENSE](LICENSE).

The project depends on [Ultralytics](https://github.com/ultralytics/ultralytics),
which is AGPL-3.0. None of its code is in this repository, but installing the
dependencies and running the combined software puts the combination under AGPL
terms, unless you have an Ultralytics commercial license. The YOLO26 weights,
and any model fine-tuned from them, are also under Ultralytics' license. The MIT
license covers this repository's code only.
