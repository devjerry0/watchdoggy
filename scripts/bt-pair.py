#!/usr/bin/env python3
"""One-shot initial pairing for a keyless BT speaker, driven in a real PTY.

Run this ONCE (with the speaker in pairing mode) to establish the trusted
device entry. The persistent agent daemon (bt-agent-daemon.py) then keeps it
reconnecting across reboots. See that file's docstring for why the link key
never persists and why the agent is what makes reconnect work anyway.

The speaker MAC comes from ``$DOGGY_BT_MAC`` or argv[1].
"""
import os
import pty
import select
import sys
import time

MAC = os.environ.get("DOGGY_BT_MAC") or (sys.argv[1] if len(sys.argv) > 1 else "")
if not MAC:
    sys.exit("usage: bt-pair.py <speaker-mac>   (put the speaker in pairing mode first)")

# (command, seconds to wait after sending). Kept deliberately generous — the
# speaker must be advertising in pairing mode for `pair` to succeed.
STEPS = [
    ("power on", 1),
    ("agent NoInputNoOutput", 1),
    ("default-agent", 1),
    ("pairable on", 1),
    ("scan on", 10),
    ("remove " + MAC, 3),
    ("scan on", 7),
    ("pair " + MAC, 15),
    ("trust " + MAC, 2),
    ("connect " + MAC, 9),
    ("scan off", 1),
    ("quit", 1),
]

def _pump(fd: int, seconds: float) -> bytes:
    """Collect whatever bluetoothctl prints for this long."""
    got = b""
    end = time.time() + seconds
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.3)
        if not r:
            continue
        try:
            d = os.read(fd, 4096)
        except OSError:
            d = b""
        if not d:
            break
        got += d
    return got


pid, fd = pty.fork()
if pid == 0:
    os.execvp("bluetoothctl", ["bluetoothctl"])

buf = b""
for cmd, delay in STEPS:
    try:
        os.write(fd, (cmd + "\n").encode())
    except OSError:
        break
    buf += _pump(fd, delay)

for line in buf.decode(errors="replace").splitlines():
    low = line.lower()
    if any(k in low for k in ("pairing successful", "failed to pair", "connection successful",
                              "failed to connect", "paired: yes", "authentication")):
        print("BT>", line.strip()[-90:])
