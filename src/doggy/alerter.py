from __future__ import annotations

import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from doggy.config import Settings, TunableSettings
from doggy.state import RuntimeSettings

_CLIP_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def pick_clip(clips_dir: Path, rng: random.Random) -> Path | None:
    clips = sorted(p for p in Path(clips_dir).glob("*") if p.suffix.lower() in _CLIP_EXTS)
    if not clips:
        return None
    return rng.choice(clips)


class Alerter(Protocol):
    def alert(self) -> None: ...


class FakeAlerter:
    def __init__(self) -> None:
        self.calls = 0

    def alert(self) -> None:
        self.calls += 1


class _ClipAlerter:
    """Shared plumbing for clip-playing alerters: pick a random clip, then emit it.

    Subclasses implement _emit() with their playback mechanism.
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()

    def alert(self) -> None:
        cfg = self._runtime.get()
        clip = pick_clip(cfg.clips_dir, self._rng)
        if clip is None:
            return
        self._emit(clip, cfg)

    def _emit(self, clip: Path, cfg: TunableSettings) -> None:
        raise NotImplementedError


class SoundDeviceAlerter(_ClipAlerter):
    """Plays a random clip on a background thread (fire-and-forget).

    `device` selects the output (e.g. a USB speaker on the Pi); None = default.
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None,
                 device: str | None = None) -> None:
        super().__init__(runtime, rng)
        self._device = device

    def _emit(self, clip: Path, cfg: TunableSettings) -> None:
        threading.Thread(target=self._play, args=(clip, cfg.max_volume), daemon=True).start()

    def _play(self, clip: Path, volume: float) -> None:
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(str(clip), dtype="float32")
        sd.play(data * max(0.0, min(1.0, volume)), samplerate, device=self._device)
        sd.wait()


class CommandAlerter(_ClipAlerter):
    """Plays a random clip by shelling out to a system player (non-blocking).

    macOS -> afplay. Linux -> pw-play/paplay (route through PipeWire, so a
    Bluetooth sink works) falling back to aplay (raw ALSA). Player-based
    playback is more reliable than PortAudio for a headless PipeWire+BT setup.
    Note: pw-play/paplay need WAV/FLAC clips, not mp3.
    """

    def _emit(self, clip: Path, cfg: TunableSettings) -> None:
        if sys.platform == "darwin":
            player: str | None = "afplay"
        else:
            player = shutil.which("pw-play") or shutil.which("paplay") or shutil.which("aplay")
        if player:
            subprocess.Popen([player, str(clip)])


def build_alerter(settings: Settings, runtime: RuntimeSettings) -> Alerter:
    if settings.alerter_backend == "log":
        return FakeAlerter()
    if settings.alerter_backend == "command":
        return CommandAlerter(runtime)
    return SoundDeviceAlerter(runtime, device=settings.audio_device)
