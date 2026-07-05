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

## 4. Harden (LAN-only appliance)

After the model + deps are downloaded:

```sh
./scripts/harden-pi.sh doggy@doggypi.local 192.168.50.0/24
```

SSH becomes key-only (a `00-` sshd drop-in overrides cloud-init's `50-` one that
enables passwords), and an nftables firewall blocks all internet egress (loopback +
LAN + DHCP + mDNS only). Both persist across reboots.

## Result

Dashboard: `http://doggypi.local:8000` (from any device on the LAN).
Lose power for a day → on restore it boots, rejoins WiFi, starts the service, and
re-arms the firewall — fully autonomous.

**Performance note:** NCNN YOLO26n at 640px is ~2.5 FPS on a Pi 4 CPU (the ~15 FPS
figure is a Pi 5). That's fine for a lingering-dog deterrent given the time-based
trigger; for more speed on a Pi 4, re-export at a smaller `imgsz` (e.g. 320).
