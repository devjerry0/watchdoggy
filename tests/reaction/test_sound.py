import random
import time

import numpy as np

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.events.store import EventStore
from doggy.reaction.hub import DogCaught
from doggy.reaction.sound import BaseAlerter, FakeAlerter, SoundReaction, pick_clip


def _img():
    return np.zeros((16, 16, 3), np.uint8)


def _event(record):
    return DogCaught(record=record, frame=_img(), mono_ts=record.ts)


def _wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.01)
    raise AssertionError("condition not met within timeout")


class _SyncThread:
    """Runs the target inline on start() so alert()'s daemon thread is deterministic."""

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self._target = target
        self._args = args

    def start(self):
        if self._target is not None:
            self._target(*self._args)


def test_fake_alerter_counts_calls():
    a = FakeAlerter()
    a.alert()
    a.alert()
    assert a.calls == 2


def test_alert_returns_clip_name_and_honors_override(tmp_path):
    (tmp_path / "a.wav").write_bytes(b"x")
    runtime = RuntimeSettings(TunableSettings(clips_dir=tmp_path, max_volume=0.5))
    played = []

    class Probe(BaseAlerter):
        def _play(self, clip, volume):
            played.append((clip.name, volume))

    probe = Probe(runtime)
    assert probe.alert() == "a.wav"
    assert probe.alert(volume=2.0) == "a.wav"   # clamped
    _wait_for(lambda: len(played) == 2)          # _play runs on a daemon thread
    assert played[0] == ("a.wav", 0.5)
    assert played[1] == ("a.wav", 1.0)


def test_alert_returns_none_when_no_clips(tmp_path):
    runtime = RuntimeSettings(TunableSettings(clips_dir=tmp_path))

    class Probe(BaseAlerter):
        def _play(self, clip, volume):
            raise AssertionError("should not play")

    assert Probe(runtime).alert() is None


def test_fake_alerter_records_volumes_and_names_clip():
    a = FakeAlerter()
    assert a.alert() == "fake.wav"
    assert a.alert(volume=2.0) == "fake.wav"
    assert a.volumes == [None, 2.0]  # raw values, pre-clamp
    assert a.calls == 2


def test_sound_reaction_attaches_played_sound(tmp_path):
    store = EventStore(tmp_path, 100, 0)
    r = store.add(_img(), 0.9, 1.0, 1000.0, 1.0)
    fake = FakeAlerter()
    SoundReaction(fake, store).on_dog_caught(_event(r))
    assert store.list()[0].sound == "fake.wav"


def test_pick_clip_none_when_empty(tmp_path):
    assert pick_clip(tmp_path, random.Random(0)) is None


def test_pick_clip_is_deterministic_with_seed(tmp_path):
    for name in ["a.wav", "b.wav", "c.wav"]:
        (tmp_path / name).write_bytes(b"RIFF")
    chosen = pick_clip(tmp_path, random.Random(0))
    assert chosen.suffix == ".wav"
    assert pick_clip(tmp_path, random.Random(0)) == chosen  # same seed -> same pick


def test_build_alerter_passes_audio_device():
    from doggy.core.config import Settings
    from doggy.core.runtime import RuntimeSettings
    from doggy.reaction.sound import SoundDeviceAlerter, build_alerter

    s = Settings(alerter_backend="sounddevice", audio_device="USB Speaker")
    a = build_alerter(s, RuntimeSettings(s.tunable()))
    assert isinstance(a, SoundDeviceAlerter)
    assert a._device == "USB Speaker"


def test_sounddevice_play_passes_configured_device(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np

    from doggy.core.config import TunableSettings
    from doggy.core.runtime import RuntimeSettings
    from doggy.reaction.sound import SoundDeviceAlerter

    calls = {}
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        play=lambda data, samplerate, device=None: calls.update(device=device),
        wait=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(
        read=lambda path, dtype: (np.zeros(4, dtype="float32"), 22050),
    ))
    a = SoundDeviceAlerter(RuntimeSettings(TunableSettings()), device="USB Speaker")
    a._play(tmp_path / "x.wav", 0.5)
    assert calls["device"] == "USB Speaker"


def test_sounddevice_play_scales_samples_by_volume(monkeypatch, tmp_path):
    import sys
    import types

    import numpy as np

    from doggy.core.config import TunableSettings
    from doggy.core.runtime import RuntimeSettings
    from doggy.reaction.sound import SoundDeviceAlerter

    captured = {}
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(
        play=lambda data, samplerate, device=None: captured.update(data=np.asarray(data)),
        wait=lambda: None,
    ))
    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(
        read=lambda path, dtype: (np.ones(4, dtype="float32"), 22050),
    ))
    a = SoundDeviceAlerter(RuntimeSettings(TunableSettings()))
    a._play(tmp_path / "x.wav", 0.25)
    assert np.allclose(captured["data"], 0.25)


def _command_alerter(monkeypatch, tmp_path, tunable):
    """Build a CommandAlerter with a stubbed pw-play + Popen, returning captured cmd."""
    import doggy.reaction.sound as sound_mod
    from doggy.core.runtime import RuntimeSettings
    from doggy.reaction.sound import CommandAlerter

    calls = {}
    monkeypatch.setattr(sound_mod.threading, "Thread", _SyncThread)
    monkeypatch.setattr(sound_mod.sys, "platform", "linux")
    monkeypatch.setattr(sound_mod.shutil, "which",
                        lambda name: "/usr/bin/pw-play" if name == "pw-play" else None)
    monkeypatch.setattr(sound_mod.subprocess, "Popen",
                        lambda cmd, *a, **k: calls.__setitem__("cmd", cmd))
    return CommandAlerter(RuntimeSettings(tunable), rng=random.Random(0)), calls


def test_command_alerter_plays_selected_clip(monkeypatch, tmp_path):
    from doggy.core.config import TunableSettings

    for name in ["bark.wav", "growl.wav", "whistle.wav"]:
        (tmp_path / name).write_bytes(b"RIFF")
    tun = TunableSettings(clips_dir=tmp_path, selected_sound="growl.wav", max_volume=0.3)
    a, calls = _command_alerter(monkeypatch, tmp_path, tun)
    a.alert()
    assert calls["cmd"][-1] == str(tmp_path / "growl.wav")


def test_command_alerter_applies_volume_flag(monkeypatch, tmp_path):
    from doggy.core.config import TunableSettings

    (tmp_path / "bark.wav").write_bytes(b"RIFF")
    tun = TunableSettings(clips_dir=tmp_path, selected_sound="bark.wav", max_volume=0.3)
    a, calls = _command_alerter(monkeypatch, tmp_path, tun)
    a.alert()
    cmd = calls["cmd"]
    assert "--volume" in cmd
    assert cmd[cmd.index("--volume") + 1] == "0.3"


def test_command_alerter_falls_back_to_random(monkeypatch, tmp_path):
    from doggy.core.config import TunableSettings

    for name in ["bark.wav", "growl.wav"]:
        (tmp_path / name).write_bytes(b"RIFF")
    tun = TunableSettings(clips_dir=tmp_path, selected_sound="random")
    a, calls = _command_alerter(monkeypatch, tmp_path, tun)
    a.alert()
    assert calls["cmd"][-1] in {str(tmp_path / "bark.wav"), str(tmp_path / "growl.wav")}


def test_command_alerter_missing_selected_falls_back_to_random(monkeypatch, tmp_path):
    from doggy.core.config import TunableSettings

    (tmp_path / "bark.wav").write_bytes(b"RIFF")
    tun = TunableSettings(clips_dir=tmp_path, selected_sound="does-not-exist.wav")
    a, calls = _command_alerter(monkeypatch, tmp_path, tun)
    a.alert()
    assert calls["cmd"][-1] == str(tmp_path / "bark.wav")
