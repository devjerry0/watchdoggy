# Deploying doggy to a Raspberry Pi (as a hardened appliance)

Target: **Raspberry Pi 4B**, USB webcam (e.g. Logitech C922), current **Raspberry
Pi OS Lite 64-bit** (Debian 13 "Trixie" as of 2026). End state: boots on WiFi with
no ethernet, SSH is key-only, the detector auto-starts, and the Pi has **no
internet egress** — and all of that survives a power loss.

## 1. Flash the SD card

From a Mac (`xz` via Homebrew), targeting the SD card's raw device (find it with
`diskutil list` — it's the removable ~N GB disk, **not** `disk0`):

```sh
diskutil unmountDisk /dev/diskN
xz -dc raspios-lite-arm64.img.xz | sudo dd of=/dev/rdiskN bs=1m
sync
```

(Or use Raspberry Pi Imager, which also does the headless config below for you.)

## 2. Headless config — cloud-init (the Trixie way)

**Gotchas that cost us hours:** on Trixie the old `wpa_supplicant.conf` boot-file
is dead, and a `custom.toml` on a *raw-dd'd* image is silently ignored (it needs a
`cmdline.txt` hook that only Raspberry Pi Imager adds). The working method is
**cloud-init**: drop these three files on the FAT `bootfs` partition **before first
boot**. Templates live in [`cloud-init/`](cloud-init/) — fill in the placeholders.

- `user-data` — creates the user, installs your SSH key, enables SSH.
- `network-config` — WiFi SSID + password (netplan v2).
- `meta-data` — instance-id + hostname.

Generate the password hash with `openssl passwd -6 'yourpassword'`.

## 3. Boot + deploy

Insert the card, power on (no ethernet needed), wait ~2–3 min (it reboots once).
Then, from the repo root:

```sh
./scripts/deploy-to-pi.sh doggy@doggypi.local
```

This rsyncs the code, `uv sync`s (CPU-only Torch), installs the **NCNN** toolchain
(`ncnn` + `pnnx` — Ultralytics can't auto-install them because uv's venv has no
`pip`), exports YOLO26n to NCNN, writes `.env`, and installs the systemd service
(`Restart=always` — the app exits 0 when no camera is attached, so it must relaunch
until the webcam is present).

## 4. Bluetooth speaker — auto-reconnect across power loss

If you want alert audio on a Bluetooth speaker (e.g. a JBL Go 5 with no aux in),
run this **before hardening** (it needs apt/internet, which the firewall later
blocks):

```sh
./scripts/setup-bt-speaker.sh doggy@doggypi.local AA:BB:CC:DD:EE:FF   # <speaker MAC>
```

It installs `pi-bluetooth`, adds the user to the `bluetooth` group, sets BlueZ
`JustWorksRepairing=always` + a reconnect policy, tells WirePlumber to load the
bluez A2DP endpoints headlessly (`monitor.bluez.seat-monitoring = disabled` —
without this the sink never appears on a headless box), installs a **persistent
pairing-agent + reconnect service** (`doggy-bt.service`), then walks you through
a one-time pairing (put the speaker in pairing mode when prompted). Set
`DOGGY_ALERTER_BACKEND=command` in `.env` so playback routes through PipeWire →
Bluetooth (`pw-play`; PortAudio/`sounddevice` will NOT hit a BT sink).

**The gotcha that cost us hours — cheap speakers store no bond.** The Go 5 pairs
as *No Bonding*, so the kernel sets `store_hint=0` and BlueZ never writes a
`[LinkKey]` to `/var/lib/bluetooth`. After a reboot it's `Paired: no` and
`connect` fails with `br-connection-unknown`. This is the *speaker's* firmware,
not fixable by how you pair (one-shot, kept-alive pipe, and a real PTY all give
store_hint=0). What makes hands-off reconnect work anyway: a `NoInputNoOutput`
pairing **agent kept registered at all times** (the `doggy-bt` daemon, in a
long-lived `bluetoothctl` PTY) so the fresh Just-Works re-pair is auto-accepted
on every boot. The service must run **as the app user, not root** — as root the
A2DP connect fails with `avdtp Permission denied` because the media endpoint
belongs to the user's PipeWire. After setup, a power cycle reconnects the
speaker ~10–15 s after boot with zero touches.

Note: the Pi 4's onboard radio shares one antenna for WiFi + BT, which can cause
occasional `br-connection-unknown` drops under load; a ~$9 USB BT dongle (e.g.
TP-Link UB400, CSR8510 — truly plug-and-play on Linux) sidesteps that entirely.

## 5. Harden (LAN-only appliance)

After the model + deps are downloaded (and BT, if any, is set up):

```sh
./scripts/harden-pi.sh doggy@doggypi.local 192.168.50.0/24
```

SSH becomes key-only (a `00-` sshd drop-in overrides cloud-init's `50-` one that
enables passwords), and an nftables firewall blocks all internet egress (loopback +
LAN + DHCP + mDNS only). Both persist across reboots.

## 6. SD-card longevity (optional but recommended)

```sh
./scripts/reduce-sd-writes.sh doggy@doggypi.local
```

Switches systemd-journald to volatile (RAM) storage — the biggest source of
constant SD writes — while keeping BT pairing and `.env` tunables persistent
(unlike a full read-only/overlay root, which would make both volatile). **The
real fix for card death is stable power**: use a proper 5V/3A supply (a laptop
USB-C *wall brick* is fine; a laptop's *ports* or a flaky charger are not).
Brownout power-cycling from a flaky supply is what bricks cards — verify clean
power on the Pi with `vcgencmd get_throttled` (want `0x0`).

## Result

Dashboard: `http://doggypi.local:8000` (from any device on the LAN).
Lose power for a day → on restore it boots, rejoins WiFi, starts the service, and
re-arms the firewall — fully autonomous.

**Performance note:** NCNN YOLO26n at 640px is ~2.5 FPS on a Pi 4 CPU (the ~15 FPS
figure is a Pi 5). That's fine for a lingering-dog deterrent given the time-based
trigger; for more speed on a Pi 4, re-export at a smaller `imgsz` (e.g. 320).
