import random


class _SyncThread:
    """Runs the target inline on start() so alert()'s daemon thread is deterministic."""

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self._target = target
        self._args = args

    def start(self):
        if self._target is not None:
            self._target(*self._args)


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
