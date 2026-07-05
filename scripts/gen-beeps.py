#!/usr/bin/env python3
"""Generate the deterrent beep(s) the detector plays on a dog sighting.

The default output, `sounds/alert-chirp.wav`, is the alert the appliance ships
with: two rising 1.4->3.7 kHz sine sweeps with a touch of 2nd harmonic —
attention-grabbing without the buzzy harshness of a square wave.

Usage:
  python3 scripts/gen-beeps.py               # write sounds/alert-chirp.wav (the default)
  python3 scripts/gen-beeps.py --variants DIR # also dump the alternatives to DIR

Pure stdlib (wave/struct/math) — no numpy. WAV/FLAC only; pw-play/paplay (the
Linux CommandAlerter players) do not take mp3.
"""
import argparse
import math
import os
import struct
import wave

SR = 44100


def write_wav(path: str, samples) -> None:
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(b"".join(
            struct.pack("<h", int(max(-1.0, min(1.0, s)) * 32767)) for s in samples))


def _env(i: int, n: int, edge: float = 0.012) -> float:
    """Raised-cosine attack/release so beeps don't click."""
    e = int(edge * SR)
    if i < e:
        return 0.5 * (1 - math.cos(math.pi * i / e))
    if i > n - e:
        return 0.5 * (1 - math.cos(math.pi * (n - i) / e))
    return 1.0


def chirp(sweeps: int, amp: float, f0: float, f1: float, dur: float, gap: float,
          harmonic: float = 0.0):
    out = []
    n = int(dur * SR)
    for _ in range(sweeps):
        ph = 0.0
        for i in range(n):
            f = f0 + (f1 - f0) * (i / n)
            ph += 2 * math.pi * f / SR
            v = math.sin(ph) + harmonic * math.sin(2 * ph)
            out.append(amp * _env(i, n) * v)
        out += [0.0] * int(gap * SR)
    return out


def _square(f: float, t: float) -> float:
    return 1.0 if math.sin(2 * math.pi * f * t) >= 0 else -1.0


def triple_square(amp=0.9, f=3000.0, beep=0.16, gap=0.11, n=3):
    out = []
    nb = int(beep * SR)
    for _ in range(n):
        out += [amp * _env(i, nb) * _square(f, i / SR) for i in range(nb)]
        out += [0.0] * int(gap * SR)
    return out


# The shipped default: "medium" rising sine chirp.
DEFAULT = lambda: chirp(sweeps=2, amp=0.72, f0=1400, f1=3700, dur=0.32, gap=0.08, harmonic=0.18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "sounds", "alert-chirp.wav"))
    ap.add_argument("--variants", metavar="DIR", help="also write alternative beeps here")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_wav(out, DEFAULT())
    print("wrote", out)

    if args.variants:
        d = os.path.abspath(args.variants)
        os.makedirs(d, exist_ok=True)
        write_wav(os.path.join(d, "beep-chirp-soft.wav"), chirp(2, 0.60, 1300, 3400, 0.34, 0.08))
        write_wav(os.path.join(d, "beep-chirp-square.wav"), chirp(3, 0.90, 1400, 4000, 0.35, 0.06, harmonic=0.0))
        write_wav(os.path.join(d, "beep-triple.wav"), triple_square())
        write_wav(os.path.join(d, "beep-alarm.wav"),
                  [0.9 * _env(i, int(1.2 * SR), 0.01) * _square(2500 if (i // int(0.05 * SR)) % 2 == 0 else 3900, i / SR)
                   for i in range(int(1.2 * SR))])
        print("wrote variants to", d)


if __name__ == "__main__":
    main()
