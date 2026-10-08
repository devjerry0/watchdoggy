# Raspberry Pi setup

This guide sets up watchdoggy on a Raspberry Pi 4B as a headless appliance. When
you're done, the Pi boots on WiFi, starts the detector automatically, allows only
key-based SSH, and has no internet access. All of this survives a power loss.

Tested with Raspberry Pi OS Lite 64-bit (Debian 13 "Trixie") and a Logitech C922
USB webcam. All commands run from the repo root on your workstation unless noted.

1. [Flash the SD card](#1-flash-the-sd-card)
2. [Headless config with cloud-init](#2-headless-config-with-cloud-init)
3. [Deploy](#3-deploy)
4. [Bluetooth speaker (optional)](#4-bluetooth-speaker-optional)
5. [Harden](#5-harden)
6. [Cloud training and self-updates (optional)](#6-cloud-training-and-self-updates-optional)
7. [HTTPS (optional)](#7-https-optional)
8. [SD card longevity](#8-sd-card-longevity)

## 1. Flash the SD card

Find the SD card's device with `diskutil list`. It's the removable disk, never
`disk0`. Then:

```sh
diskutil unmountDisk /dev/diskN
xz -dc raspios-lite-arm64.img.xz | sudo dd of=/dev/rdiskN bs=1m
sync
```

Raspberry Pi Imager also works, and can do the headless config in step 2 for you.

## 2. Headless config with cloud-init

Before first boot, copy the three templates from [`cloud-init/`](cloud-init/)
onto the SD card's FAT `bootfs` partition and fill in the placeholders:

- `user-data` creates the `doggy` user, installs your SSH key, and enables SSH.
  Generate the password hash with `openssl passwd -6 'yourpassword'`.
- `network-config` sets the WiFi SSID and password (netplan v2).
- `meta-data` sets the instance ID and the hostname (`doggypi`).

On Trixie, the older methods don't work with a `dd`-flashed image:
`wpa_supplicant.conf` in the boot partition is no longer read, and `custom.toml`
is ignored unless Raspberry Pi Imager also added its `cmdline.txt` hook.

## 3. Deploy

Insert the card and power on. No ethernet is needed. The first boot takes 2-3
minutes and reboots once. Then:

```sh
./scripts/deploy-to-pi.sh doggy@doggypi.local
```

The script:

- syncs the code and runs `uv sync` (CPU-only PyTorch),
- installs `ncnn` and `pnnx`, which Ultralytics can't install itself because a
  uv venv has no `pip`,
- exports YOLO26n to NCNN,
- writes `.env` (if there isn't one already),
- installs the `doggy` systemd service with `Restart=always`. The app exits
  cleanly when no camera is attached, so systemd keeps restarting it until the
  webcam shows up.

The script is safe to re-run. The dashboard is now at `http://doggypi.local:8000`.

## 4. Bluetooth speaker (optional)

Run this before hardening. It needs apt, which the firewall blocks later.

```sh
./scripts/setup-bt-speaker.sh doggy@doggypi.local AA:BB:CC:DD:EE:FF   # speaker MAC
```

Put the speaker in pairing mode when prompted. The script:

- installs `pi-bluetooth` and adds the user to the `bluetooth` group,
- configures BlueZ to always accept re-pairing and to reconnect on its own,
- sets `monitor.bluez.seat-monitoring = disabled` in WirePlumber; without this,
  the Bluetooth audio sink never appears on a headless Pi,
- installs `doggy-bt.service`, which keeps a pairing agent running and reconnects
  the speaker.

Deploy already sets `DOGGY_ALERTER_BACKEND=command`, which plays sound with
`pw-play` through PipeWire. The `sounddevice` backend can't reach a Bluetooth
sink.

After a power cycle the speaker reconnects 10-15 seconds after boot.

### Why the pairing agent runs all the time

Some cheap speakers (the JBL Go 5, for example) pair with "No Bonding", so BlueZ
never stores a link key. After every reboot the speaker shows `Paired: no` and
`connect` fails with `br-connection-unknown`. This comes from the speaker's
firmware; changing how you pair doesn't fix it. The workaround is to keep a
`NoInputNoOutput` agent registered permanently, so the fresh pairing on each boot
is accepted automatically.

The service runs as the app user, not root. As root, the A2DP connection fails
with `avdtp Permission denied`, because the media endpoint belongs to the user's
PipeWire session.

The Pi 4 shares one antenna between WiFi and Bluetooth, which can cause occasional
dropouts under load. A USB Bluetooth dongle with a CSR8510 chip (such as the
TP-Link UB400) avoids this and works on Linux without drivers.

## 5. Harden

Run this after deploy (and after the speaker setup, if you did it), because
nothing can be downloaded afterwards. Pass your LAN's CIDR:

```sh
./scripts/harden-pi.sh doggy@doggypi.local 192.168.50.0/24
```

This makes SSH key-only and installs an nftables firewall that blocks all
internet traffic. Only loopback, LAN, DHCP, and mDNS are allowed. Both settings
persist across reboots.

## 6. Cloud training and self-updates (optional)

> **What this sends off the Pi.** Live detection stays local either way. Once
> this step is done, every frame the Pi saves for training is uploaded to the
> Modal volume in your account. That includes frames from alarms, uncertain
> detections, periodic background shots, and moments when a person is in view.
> Still frames only, never the live stream. The trainer also checks
> `api.github.com` for releases and the current time. Nothing else is reachable.
> Skip this step and nothing leaves the Pi.

Run this after hardening. It needs a [Modal](https://modal.com) account and copies
your local `~/.modal.toml` to the Pi.

```sh
./scripts/setup-pi-trainer.sh doggy@doggypi.local
```

This creates a separate `trainer` user. A per-user firewall rule allows only that
user out to the internet, over DNS and HTTPS. The detector stays offline. The
trainer handles:

- scheduled cloud training, gated deployment of new models, and the
  `/training` page,
- daily self-updates from GitHub releases, with a health check and automatic
  rollback,
- clock sync. The firewall blocks NTP and the Pi has no battery-backed clock, so
  the trainer sets the time from HTTPS `Date` headers instead.

See [MANUAL.md](../../MANUAL.md) for how training and updates work.

To stop all uploads and update checks later, disable the trainer's timer. Frames
already uploaded stay in your Modal volume until you delete them there.

```sh
ssh doggy@doggypi.local 'sudo systemctl disable --now doggy-trainer.timer'
```

### Updating without the trainer

A hardened Pi can't install packages, so `deploy-to-pi.sh` won't work anymore.
To update the code, sync it and restart the service:

```sh
rsync -az --delete --exclude __pycache__ src/     doggy@doggypi.local:doggy/src/
rsync -az --delete --exclude __pycache__ scripts/ doggy@doggypi.local:doggy/scripts/
ssh doggy@doggypi.local 'sudo systemctl restart doggy'
```

Releases that change dependencies can't be installed this way, and the
self-updater refuses them too.

## 7. HTTPS (optional)

Browsers only allow the microphone (push-to-talk) and notifications on HTTPS
pages.

```sh
./scripts/setup-https.sh doggy@doggypi.local
```

This creates a private certificate authority on the Pi, issues the dashboard a
certificate, and walks each device through trusting it the first time you open
the dashboard. No internet access is needed. See [MANUAL.md](../../MANUAL.md#https-for-push-to-talk-and-notifications).

## 8. SD card longevity

```sh
./scripts/reduce-sd-writes.sh doggy@doggypi.local
```

This keeps the systemd journal in RAM, which removes most constant SD writes.
Bluetooth pairing, `settings.json`, and the dataset still persist, unlike with a
fully read-only root filesystem.

SD cards on a Pi mostly die from unstable power. Use a proper 5V/3A supply; a
USB-C wall charger is fine, but laptop USB ports and cheap chargers are not.
`vcgencmd get_throttled` should report `0x0`.

## Performance

YOLO26n through NCNN at 640px runs at about 2.4 FPS on a Pi 4 CPU (a Pi 5 is
much faster). That's enough because the trigger waits for the dog to stay in the
watch area for a while rather than reacting to a single frame. For more speed,
export at a smaller `imgsz` such as 480, at some cost in accuracy on small or
distant dogs.

Inference uses every core, so anything else using the CPU lowers the frame rate.
The trainer service runs at low priority for this reason.
