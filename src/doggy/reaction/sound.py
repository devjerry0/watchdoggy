from __future__ import annotations

import random
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Protocol

from doggy.core.config import Settings, TunableSettings
from doggy.core.runtime import RuntimeSettings

_CLIP_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def pick_clip(clips_dir: Path, rng: random.Random) -> Path | None:
    clips = sorted(p for p in Path(clips_dir).glob("*") if p.suffix.lower() in _CLIP_EXTS)
    if not clips:
        return None
    return rng.choice(clips)


class Alerter(Protocol):
    def alert(self) -> None: ...


class SoundReaction:
    """Reaction (Observer): plays the deterrent clip when a dog is caught."""

    def __init__(self, alerter: Alerter) -> None:
        self._alerter = alerter

    def on_dog_caught(self, event) -> None:
        self._alerter.alert()


class FakeAlerter:
    def __init__(self) -> None:
        self.calls = 0

    def alert(self) -> None:
        self.calls += 1


class BaseAlerter:
    """Template Method base for clip-playing alerters.

    ``alert()`` owns the shared skeleton: read the current config, resolve the
    clip to play, clamp the volume once, then hand a background daemon thread to
    the subclass's ``_play`` hook. Subclasses implement only ``_play``.
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None) -> None:
        self._runtime = runtime
        self._rng = rng or random.Random()

    def alert(self) -> None:
        cfg = self._runtime.get()
        clip = self._resolve_clip(cfg)
        if clip is None:
            return
        volume = max(0.0, min(1.0, cfg.max_volume))
        threading.Thread(target=self._play, args=(clip, volume), daemon=True).start()

    def _resolve_clip(self, cfg: TunableSettings) -> Path | None:
        """Play the user-selected clip when set (and present), else a random one."""
        if cfg.selected_sound and cfg.selected_sound != "random":
            # Path(...).name strips any directory components → no path traversal.
            chosen = Path(cfg.clips_dir) / Path(cfg.selected_sound).name
            if chosen.is_file():
                return chosen
        return pick_clip(cfg.clips_dir, self._rng)

    def _play(self, clip: Path, volume: float) -> None:
        raise NotImplementedError


class SoundDeviceAlerter(BaseAlerter):
    """Plays a clip through PortAudio on a background thread (fire-and-forget).

    `device` selects the output (e.g. a USB speaker on the Pi); None = default.
    """

    def __init__(self, runtime: RuntimeSettings, rng: random.Random | None = None,
                 device: str | None = None) -> None:
        super().__init__(runtime, rng)
        self._device = device

    def _play(self, clip: Path, volume: float) -> None:
        import soundfile as sf
        import sounddevice as sd

        data, samplerate = sf.read(str(clip), dtype="float32")
        sd.play(data * volume, samplerate, device=self._device)
        sd.wait()


class CommandAlerter(BaseAlerter):
    """Plays a clip by shelling out to a system player (non-blocking).

    macOS -> afplay. Linux -> pw-play/paplay (route through PipeWire, so a
    Bluetooth sink works) falling back to aplay (raw ALSA). Player-based
    playback is more reliable than PortAudio for a headless PipeWire+BT setup.
    Note: pw-play/paplay need WAV/FLAC clips, not mp3.
    """

    def _play(self, clip: Path, volume: float) -> None:
        if sys.platform == "darwin":
            player: str | None = "afplay"
        else:
            player = shutil.which("pw-play") or shutil.which("paplay") or shutil.which("aplay")
        if player:
            subprocess.Popen([player, *self._volume_args(player, volume), str(clip)])

    @staticmethod
    def _volume_args(player: str, volume: float) -> list[str]:
        """Volume flag for the chosen player. pw-play/afplay take a 0.0-1.0 gain;
        aplay has no volume control, so it is left at the sink's level."""
        name = Path(player).name
        vol = str(volume)
        if name in ("pw-play", "paplay"):
            return ["--volume", vol]
        if name == "afplay":
            return ["-v", vol]
        return []


class LogAlerter(BaseAlerter):
    """No-op backend: resolves a clip but plays nothing (headless/dev use)."""

    def _play(self, clip: Path, volume: float) -> None:
        pass


_BACKENDS = {
    "sounddevice": lambda settings, runtime: SoundDeviceAlerter(
        runtime, device=settings.audio_device
    ),
    "command": lambda settings, runtime: CommandAlerter(runtime),
    "log": lambda settings, runtime: LogAlerter(runtime),
}


def build_alerter(settings: Settings, runtime: RuntimeSettings) -> Alerter:
    # Unknown backends fall back to the sounddevice default, matching the prior
    # behaviour where anything other than "log"/"command" built a SoundDeviceAlerter.
    factory = _BACKENDS.get(settings.alerter_backend, _BACKENDS["sounddevice"])
    return factory(settings, runtime)
