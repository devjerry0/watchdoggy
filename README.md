# watchdoggy

Counter Watch is a ~$62 Raspberry Pi appliance that watches your kitchen counter, spots the dog when it jumps up, and plays a deterrent sound. It only reacts inside an area you draw, ignores people, manages its own temperature — and **retrains its own vision model on the frames it captures**, so it gets better at your kitchen, your light, your dog, week after week.

Detection runs entirely on the device. The camera feed never leaves your network. The only thing that ever touches the internet — and only if you enable it — is a sandboxed trainer that sends anonymized training jobs to your own private cloud GPU account.

![Counter Watch dashboard](docs/dashboard.png)

## What it does

- Detects dogs with a YOLO model running locally on the Pi's CPU (via NCNN). No cloud vision API.
- Only acts on dogs inside a watch area you draw on the live view.
- Ignores people — including a size/geometry filter for the classic failure where a bent-over person reads as a "dog".
- Plays a deterrent sound through a Bluetooth or USB speaker, with cooldowns and an hourly cap.
- Keeps a catch log with a snapshot and a short clip of every alarm, viewable on the dashboard.
- Loops your own calm audio (soothing sounds) while it watches; the deterrent always interrupts and the music resumes on its own.
- Scales its work rate with CPU temperature so a fanless Pi doesn't throttle.
- Captures training frames from its own mistakes and edge cases, and **improves itself** (below).
- Serves a plain-language dashboard on your network: Dashboard, Label, and Training pages.

## It trains itself

The stock COCO-trained model has never seen *your* kitchen from *your* camera angle. Ours kept mistaking a person loading the dishwasher for a dog. The fix is a closed loop that lives on the appliance:

1. **Capture** — the detector saves interesting frames as they happen: every alarm (plus the raw seconds before it), borderline detections, suppressed boxes, person activity, and periodic background shots.
2. **Label** — the `/label` page is a fast one-tap verdict tool (dog / person only / nothing), with a full box editor when a frame needs hand-drawn truth. A large offline model (yolo26x) pre-draws boxes for every frame — each night it scores the day's new frames on a cloud GPU, so the morning queue is already annotated and most labels are one tap.

   ![Label page: filter chips, the big model's boxes, filmstrip navigation, one-tap verdicts](docs/label.png)
3. **Train** — every couple of days (configurable), a dedicated trainer process packs the labeled dataset, sends one job to a Modal cloud GPU, and gets back a fine-tuned model: big-model label fusion, blur/dark augmentation, training, frame-level evaluation scored the way the alarm actually fires, NCNN export, and a robustness stress test (blur, darkness, overexposure, compression, camera shift).
4. **Gate** — the new model deploys **only if it beats or ties the currently-live model on a held-out exam with zero false fires**. No regression can ship. The previous model is kept on disk for instant rollback.
5. **Repeat** — every mistake it makes becomes training data against it.

Your only job is a few minutes of tapping on the Label page now and then. Everything else — nightly prelabels, scheduled training, evaluation, gated deployment — happens on its own.

The `/training` page is the console: edit the recipe (epochs, batch, freeze, augmentation) and the schedule, queue a run manually, watch the cloud run's log live, and read every run's full report (leaderboard, deploy-truth scores, robustness table).

![Training console: live model score, dataset composition, cloud budget, and the improvement chart](docs/training.png)

In this kitchen the loop took the model from 15/61 dog moments caught with 5 false alarms (stock) to 63/64 caught with zero false-alarm signal — measured on the exported bundle the Pi actually runs, on frames the model never trained on.

### Privacy model

- The **detector service has no internet access, enforced by the firewall** — not by promises. `scripts/harden-pi.sh` sets an nftables egress lockdown (LAN only).
- Cloud training is **opt-in** and runs as a separate `trainer` user. A per-UID firewall exception lets *only that user* reach the internet (DNS + HTTPS), verified blocked for everything else. Frames go to your own private Modal volume and nowhere else.
- Skip `setup-pi-trainer.sh` and the appliance is 100% offline; you can still run the same training pipeline manually from a workstation (`scripts/train_kitchen_model.py`, including a 10-config hyperparameter sweep with `--sweep`).

## Hardware (~$62)

| Part | Price |
|---|---|
| Raspberry Pi 4 Model B | $35 |
| Aluminum heatsink case | $12 |
| 1080p USB webcam | $15 |
| Total | ~$62 |

Plus any Bluetooth or USB speaker you already have for the sound (this build uses a JBL Go), and optionally a Modal account for cloud training (a full training run costs well under a dollar).

## How it works

```
USB webcam -> capture thread -> YOLO (NCNN, on-CPU)
           -> watch-area filter -> person suppression -> oversize filter
           -> M-of-N + confirm-timer trigger -> safety limits (cooldown, hourly cap)
           -> deterrent sound + catch log + clip + training-frame capture
```

A FastAPI app streams the annotated view over MJPEG and exposes the live-tunable settings. A governor reads the CPU temperature each loop and paces detection to keep the board below its throttle point. A reaction hub fans each confirmed catch out to the sound, the event log, the clip recorder, and the dataset capturer.

The training side: the web UI writes job files to a queue directory; the `trainer` user's systemd timer picks them up (or synthesizes them on schedule), runs `scripts/modal_pipeline.py` — the whole pipeline in one cloud container — and applies gated results through a fixed-path root helper. The pipeline itself is the `scripts/kitchen_training/` package, path-configurable so the identical code runs on a workstation or in the cloud.

The package layout and the design patterns behind it are documented in [ARCHITECTURE.md](ARCHITECTURE.md).

## Quick start (dev, on a Mac)

```sh
uv sync
cp .env.example .env          # set DOGGY_CAMERA_INDEX for your webcam
uv run yolo export model=yolo26n.pt format=ncnn   # downloads yolo26n.pt
# drop at least one sound clip into sounds/
uv run doggy                  # dashboard at http://127.0.0.1:8000
```

Grant your terminal camera permission (System Settings, Privacy, Camera), or OpenCV returns empty frames with no error.

## Deploy to a Raspberry Pi

```sh
./scripts/deploy-to-pi.sh <user@host>
```

This syncs the code, installs dependencies with `uv`, downloads and NCNN-exports the model for ARM, writes a Pi `.env`, and installs a systemd service that runs on boot.

Optional:

- `scripts/setup-bt-speaker.sh` sets up a Bluetooth speaker with hands-free auto-reconnect (PipeWire).
- `scripts/harden-pi.sh` locks it down with a LAN-only egress firewall and key-only SSH.
- `scripts/setup-https.sh` gives the dashboard a padlock via a home CA (needed for push-to-talk and notifications).
- `scripts/setup-pi-trainer.sh` enables autonomous cloud training: creates the sandboxed `trainer` user, the per-UID firewall exception, the Modal client, and the training schedule. Needs a Modal account (`modal token` on the machine you run it from).

## Using the dashboard

Open `http://<pi-host>:8000` from any device on your network.

- The status pill shows Watching, Dog spotted, or Cooling down.
- Draw the watch area by tapping corners around the counter on the live view, then Save area.
- Simple settings cover how sure it must be, how long the dog must linger, the wait between reactions, the hourly cap, and Ignore people.
- Advanced holds the detection-window and person-matching knobs. System shows temperature, power, and speed.
- The catch log shows each alarm with its clip; a one-tap "Not a dog" button turns any false alarm straight into training data.
- **Label** is the annotation tool; **Training** is the training console.

Soothing sounds loops your own calm audio, music or white noise, through the speaker while it watches. Upload tracks from the Soothing sounds card, up to 1 GB in total, then turn it on. The deterrent always takes priority, so an alarm interrupts the music the moment a dog is confirmed and the loop resumes on its own a little later.

![Weekly activity report, per-sound deterrence rates, and the soothing schedule](docs/watch-panels.png)

### HTTPS (for push-to-talk and notifications)

Browsers only allow the microphone and notifications on secure pages, so those features need the dashboard served over https. One script sets it up:

```sh
./scripts/setup-https.sh <user@host>
```

It creates a "watchdoggy home CA" on the Pi, issues the dashboard a certificate signed by it, and restarts the service. The CA never leaves your Pi and nothing talks to the internet.

Then just open the same address you always use, `http://<pi-host>:8000`, on each device. The page checks whether that device already trusts the home certificate. If it does, it sends you straight to the secure dashboard. If it does not, it hands you the certificate and the one-time steps to trust it (it detects your platform and offers the right file):

- iPhone/iPad: open the profile, install it in Settings, then Settings > General > About > Certificate Trust Settings > enable it
- Mac: open the file in Keychain Access, set Trust to Always
- Android: Settings > Security > Install a certificate > CA certificate

The page rechecks on its own, so as soon as the device trusts the certificate it moves you along. After that the padlock is normal on every visit and your old bookmark keeps working. The certificate lasts about two years; re-run the script to renew it, and your devices keep working without any new steps. The secure dashboard also has a direct address, `https://<pi-host>:8443`, once a device is trusted.

## Configuration

Config is set with `DOGGY_*` environment variables (see `.env.example`). Live-tunable params are also editable from the dashboard. Structural params (camera, model, audio backend) need a restart. Training recipe and schedule live on the Training page.

## Tests

```sh
uv run pytest -m "not slow"    # fast suite (~350 tests), no hardware or weights
uv run pytest -m slow          # detector test (needs the model and fixtures)
```

## License

AGPL-3.0-or-later, matching YOLO26n.
