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
