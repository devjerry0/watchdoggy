import time

import pytest
from pydantic import ValidationError

from doggy.core.config import TunableSettings
from doggy.core.status import Status, StatusStore
from doggy.reaction.soothing import SoothingPlayer
from tests.reaction.conftest import FakeProc, FakeSpawn, _player, _runtime, _tracks, _wait_for, running


# -- config + status interfaces --------------------------------------------


def test_soothing_tunables_defaults_and_bounds():
    cfg = TunableSettings()
    assert cfg.soothing_enabled is False
    assert cfg.soothing_volume == 0.4
    assert cfg.soothing_resume_seconds == 45.0
    with pytest.raises(ValidationError):
        TunableSettings(soothing_volume=1.5)
    with pytest.raises(ValidationError):
        TunableSettings(soothing_resume_seconds=-1.0)


def test_status_has_soothing_track_default_none():
    assert Status().soothing_track is None


def test_list_tracks_is_name_sorted_and_filters_extensions(monkeypatch, tmp_path):
    # readdir order is arbitrary on the Pi (ext4), so the sort is load-bearing:
    # feed glob a deliberately shuffled order and non-audio files and assert the
    # result is name-sorted audio only (the .upload.part temp file is dropped).
    for n in ("b.mp3", "a.wav", "c.flac", "notes.txt", ".upload.part"):
        (tmp_path / n).write_bytes(b"x")
    order = ["c.flac", "notes.txt", "b.mp3", ".upload.part", "a.wav"]
    monkeypatch.setattr(
        type(tmp_path), "glob", lambda self, pat: [tmp_path / n for n in order])
    player = SoothingPlayer(_runtime(), tmp_path, StatusStore())
    assert [p.name for p in player._list_tracks()] == ["a.wav", "b.mp3", "c.flac"]


# -- loop semantics --------------------------------------------------------


def test_disabled_never_spawns(tmp_path):
    _tracks(tmp_path, "a.mp3")
    spawn = FakeSpawn()
    status = StatusStore()
    with running(_player(tmp_path, spawn, rt=_runtime(enabled=False), status=status)):
        time.sleep(0.1)
        assert spawn.calls == []
        assert status.snapshot().soothing_track is None


def test_plays_in_name_order_then_loops(tmp_path):
    _tracks(tmp_path, "b.mp3", "a.mp3", "c.mp3")  # created out of order
    spawn = FakeSpawn()  # autofinish success -> loop cycles quickly
    rt = _runtime(enabled=True)
    status = StatusStore()
    with running(_player(tmp_path, spawn, rt=rt, status=status)):
        _wait_for(lambda: len(spawn.calls) >= 6)
        assert spawn.names[:6] == ["a.mp3", "b.mp3", "c.mp3", "a.mp3", "b.mp3", "c.mp3"]
        _wait_for(lambda: status.snapshot().soothing_track in {"a.mp3", "b.mp3", "c.mp3"})
        rt.update(TunableSettings(soothing_enabled=False))
        _wait_for(lambda: status.snapshot().soothing_track is None)


def test_volume_read_fresh_per_track(tmp_path):
    _tracks(tmp_path, "a.mp3", "b.mp3")
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    rt = _runtime(enabled=True, volume=0.4)
    with running(_player(tmp_path, spawn, rt=rt)):
        _wait_for(lambda: len(spawn.calls) == 1)
        assert spawn.calls[0][1] == 0.4
        spawn.procs[0].finish(0)  # let track a end so b starts at the same volume
        _wait_for(lambda: len(spawn.calls) == 2)
        assert spawn.calls[1][0].name == "b.mp3"
        assert spawn.calls[1][1] == 0.4


def test_volume_change_applied_live_without_restart(tmp_path):
    # Preferred path: moving the loudness slider mid-track sets the running
    # stream's volume in place (via set_volume) -- no re-spawn, track keeps
    # playing from where it was.
    _tracks(tmp_path, "a.mp3", "b.mp3")
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    rt = _runtime(enabled=True, volume=0.4)
    applied: list[float] = []
    live = lambda proc, v: (applied.append(v), True)[1]  # noqa: E731
    with running(_player(tmp_path, spawn, rt=rt, set_volume=live)):
        _wait_for(lambda: len(spawn.calls) == 1)
        first = spawn.procs[0]
        assert applied == [0.4]  # asserted once on spawn
        rt.update(TunableSettings(soothing_enabled=True, soothing_volume=0.1))
        _wait_for(lambda: 0.1 in applied)
        # Still the same, un-terminated track: no re-spawn.
        assert not first.terminated
        assert len(spawn.calls) == 1


def test_volume_change_restarts_when_live_apply_unavailable(tmp_path):
    # Fallback path (no PipeWire / set_volume returns False): the SAME track is
    # re-spawned at the new volume so the change still takes effect.
    _tracks(tmp_path, "a.mp3", "b.mp3")
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    rt = _runtime(enabled=True, volume=0.4)
    with running(_player(tmp_path, spawn, rt=rt)):  # default set_volume -> False
        _wait_for(lambda: len(spawn.calls) == 1)
        assert spawn.calls[0][0].name == "a.mp3" and spawn.calls[0][1] == 0.4
        first = spawn.procs[0]
        rt.update(TunableSettings(soothing_enabled=True, soothing_volume=0.1))
        _wait_for(lambda: len(spawn.calls) == 2)
        assert first.terminated
        assert spawn.calls[1][0].name == "a.mp3"  # same track, not advanced to b
        assert spawn.calls[1][1] == 0.1


def test_soothing_schedule_gates_playback(tmp_path):
    # Monday 2026-07-06 is a Monday; a window Mon 09:00-17:00. At noon it plays;
    # outside the window it does not spawn at all.
    from datetime import datetime
    _tracks(tmp_path, "a.mp3")
    win = {"days": [0], "start": "09:00", "end": "17:00"}
    rt = _runtime(enabled=True, soothing_schedule_enabled=True, soothing_windows=[win])
    inside = datetime(2026, 7, 6, 12, 0).timestamp()
    outside = datetime(2026, 7, 6, 20, 0).timestamp()

    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    with running(_player(tmp_path, spawn, rt=rt, wall_clock=lambda: outside)):
        time.sleep(0.1)
        assert spawn.calls == []  # off-schedule: never spawns

    spawn2 = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    with running(_player(tmp_path, spawn2, rt=rt, wall_clock=lambda: inside)):
        _wait_for(lambda: len(spawn2.calls) == 1)  # in-window: plays


def test_soothing_schedule_window_close_stops_current_track(tmp_path):
    from datetime import datetime
    _tracks(tmp_path, "a.mp3")
    win = {"days": [0], "start": "09:00", "end": "17:00"}
    rt = _runtime(enabled=True, soothing_schedule_enabled=True, soothing_windows=[win])
    now = {"t": datetime(2026, 7, 6, 12, 0).timestamp()}  # inside
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    with running(_player(tmp_path, spawn, rt=rt, wall_clock=lambda: now["t"])):
        _wait_for(lambda: len(spawn.calls) == 1)
        proc = spawn.procs[0]
        now["t"] = datetime(2026, 7, 6, 20, 0).timestamp()  # window closes
        _wait_for(lambda: proc.terminated)


def test_toggle_off_terminates_current_track(tmp_path):
    _tracks(tmp_path, "a.mp3")
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    rt = _runtime(enabled=True)
    status = StatusStore()
    with running(_player(tmp_path, spawn, rt=rt, status=status)):
        _wait_for(lambda: status.snapshot().soothing_track == "a.mp3")
        proc = spawn.procs[0]
        rt.update(TunableSettings(soothing_enabled=False))
        _wait_for(lambda: proc.terminated and status.snapshot().soothing_track is None)
