import logging

from tests.reaction.conftest import _backend


# -- default _spawn_player backend selection -------------------------------


def test_spawn_player_prefers_pw_play_with_volume(monkeypatch, tmp_path):
    import doggy.reaction.soothing.audio as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which",
                        lambda n: "/usr/bin/pw-play" if n == "pw-play" else None)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    proc = _backend().spawn(tmp_path / "calm.mp3", 0.3)
    assert proc is not None
    assert cmds[0] == ["/usr/bin/pw-play", "--volume", "0.3", str(tmp_path / "calm.mp3")]


def test_spawn_player_falls_back_to_pw_cat(monkeypatch, tmp_path):
    import doggy.reaction.soothing.audio as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which",
                        lambda n: "/usr/bin/pw-cat" if n == "pw-cat" else None)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    _backend().spawn(tmp_path / "calm.wav", 0.5)
    # Bare pw-cat needs an explicit --playback; pw-play (above) defaults to it.
    assert cmds[0] == [
        "/usr/bin/pw-cat", "--volume", "0.5", "--playback", str(tmp_path / "calm.wav")]


def test_spawn_player_darwin_uses_afplay(monkeypatch, tmp_path):
    import doggy.reaction.soothing.audio as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    _backend().spawn(tmp_path / "calm.mp3", 0.5)
    assert cmds[0] == ["afplay", "-v", "0.5", str(tmp_path / "calm.mp3")]


def test_spawn_player_none_when_no_player_and_logs_once(monkeypatch, tmp_path, caplog):
    import doggy.reaction.soothing.audio as mod

    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(mod.sys, "platform", "linux")
    backend = _backend()
    with caplog.at_level(logging.INFO):
        assert backend.spawn(tmp_path / "x.mp3", 0.4) is None
        assert backend.spawn(tmp_path / "y.mp3", 0.4) is None
    logged = [r for r in caplog.records if "no audio player" in r.message]
    assert len(logged) == 1  # logged once, then idles quietly
