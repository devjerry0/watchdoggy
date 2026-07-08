import contextlib
import logging
import subprocess
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import Status, StatusStore
from doggy.reaction.soothing import SoothingPlayer


# -- helpers ---------------------------------------------------------------


def _wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def _runtime(enabled=False, volume=0.4, resume=45.0, **extra):
    return RuntimeSettings(TunableSettings(
        soothing_enabled=enabled, soothing_volume=volume,
        soothing_resume_seconds=resume, **extra))


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


class FakeProc:
    """A controllable stand-in for subprocess.Popen. ``autofinish`` marks it
    already-exited (wait returns immediately); otherwise it "plays" until
    ``finish``/``terminate`` sets its done event, and ``wait(timeout)`` raises
    TimeoutExpired like the real thing."""

    def __init__(self, code=0, autofinish=True):
        self._done = threading.Event()
        self._code = code
        self.terminated = False
        if autofinish:
            self._done.set()

    def poll(self):
        return self._code if self._done.is_set() else None

    def wait(self, timeout=None):
        if self._done.wait(timeout):
            return self._code
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)

    def terminate(self):
        self.terminated = True
        self._code = -15
        self._done.set()

    def finish(self, code=0):
        self._code = code
        self._done.set()


class FakeSpawn:
    """Records every (path, volume) call and returns whatever ``factory``
    produces (default: an immediately-finished success proc)."""

    def __init__(self, factory=None):
        self.calls: list[tuple[Path, float]] = []
        self.procs: list[FakeProc] = []
        self._factory = factory or (lambda p, v: FakeProc(code=0, autofinish=True))

    def __call__(self, path, volume):
        self.calls.append((Path(path), volume))
        proc = self._factory(path, volume)
        if proc is not None:
            self.procs.append(proc)
        return proc

    @property
    def names(self):
        return [p.name for p, _ in self.calls]


@contextlib.contextmanager
def running(player):
    player.start()
    try:
        yield player
    finally:
        player.stop()


def _player(tmp_path, spawn, rt=None, status=None, clock=time.monotonic, poll=0.02):
    p = SoothingPlayer(rt or _runtime(enabled=True), tmp_path,
                       status or StatusStore(), clock=clock, spawn=spawn)
    p._poll = poll
    return p


def _tracks(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_bytes(b"x")


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
        rt.update(TunableSettings(soothing_enabled=True, soothing_volume=0.7))
        spawn.procs[0].finish(0)  # let track a end so b starts
        _wait_for(lambda: len(spawn.calls) == 2)
        assert spawn.calls[1][0].name == "b.mp3"
        assert spawn.calls[1][1] == 0.7


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


def test_catch_interrupts_holds_extends_and_resumes_next(tmp_path):
    _tracks(tmp_path, "a.mp3", "b.mp3", "c.mp3")
    clk = FakeClock(0.0)
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    status = StatusStore()
    player = _player(tmp_path, spawn, rt=_runtime(enabled=True, resume=45.0),
                     status=status, clock=clk)
    with running(player):
        # a is playing (a blocking proc)
        _wait_for(lambda: spawn.names[:1] == ["a.mp3"]
                  and status.snapshot().soothing_track == "a.mp3")
        proc_a = spawn.procs[0]

        # a catch cuts a instantly and arms the hold
        player.on_dog_caught(None)
        _wait_for(lambda: proc_a.terminated and status.snapshot().soothing_track is None)
        assert len(spawn.calls) == 1  # nothing new while held

        # a second catch partway through EXTENDS the hold (45 -> 75)
        clk.t = 30.0
        player.on_dog_caught(None)

        # past the first deadline (45) but not the extended one (75): still held
        clk.t = 50.0
        time.sleep(0.1)
        assert len(spawn.calls) == 1
        assert status.snapshot().soothing_track is None

        # past the extended deadline: resume with the NEXT track (b)
        clk.t = 80.0
        _wait_for(lambda: len(spawn.calls) == 2)
        assert spawn.calls[1][0].name == "b.mp3"
        _wait_for(lambda: status.snapshot().soothing_track == "b.mp3")


def test_catch_with_zero_resume_resumes_immediately_with_next(tmp_path):
    # resume_seconds == 0 is a valid (degenerate) hold: the catch still cuts the
    # current track, and playback resumes at once with the next one.
    _tracks(tmp_path, "a.mp3", "b.mp3")
    clk = FakeClock(0.0)
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    player = _player(tmp_path, spawn, rt=_runtime(enabled=True, resume=0.0), clock=clk)
    with running(player):
        _wait_for(lambda: spawn.names[:1] == ["a.mp3"])
        proc_a = spawn.procs[0]
        player.on_dog_caught(None)
        _wait_for(lambda: proc_a.terminated)
        _wait_for(lambda: spawn.names[:2] == ["a.mp3", "b.mp3"])


def test_catch_in_spawn_window_is_cut_and_stays_held(tmp_path):
    # Reviewer's race: a catch lands AFTER the loop-top hold check but BEFORE
    # _play registers the freshly spawned proc. on_dog_caught reads a null proc
    # and terminates nothing; without the fix the calm track plays to its natural
    # end THROUGH the armed hold. Reproduce it deterministically (no sleeps) by
    # firing the catch from inside spawn -- spawn runs before registration.
    _tracks(tmp_path, "a.mp3", "b.mp3")
    clk = FakeClock(0.0)
    status = StatusStore()
    caught = threading.Event()

    def factory(p, v):
        proc = FakeProc(autofinish=False)
        if not caught.is_set():  # fire once, inside the first spawn's window
            caught.set()
            player.on_dog_caught(None)  # _proc still None here -> cuts nothing
        return proc

    spawn = FakeSpawn(factory)
    player = _player(tmp_path, spawn, rt=_runtime(enabled=True, resume=45.0),
                     status=status, clock=clk)
    with running(player):
        # The track spawned into the hold must be terminated, not left playing.
        _wait_for(lambda: spawn.procs and spawn.procs[0].terminated)
        # ...and nothing new starts while the hold is armed.
        clk.t = 30.0
        time.sleep(0.05)
        assert len(spawn.calls) == 1
        assert status.snapshot().soothing_track is None
        # Past the hold: playback resumes with the NEXT track.
        clk.t = 50.0
        _wait_for(lambda: len(spawn.calls) == 2)
        assert spawn.calls[1][0].name == "b.mp3"


def test_catch_in_spawn_window_is_cut_at_registration(tmp_path):
    # Pins the registration-seam guard specifically: the track spawned into the
    # hold is cut the instant it is registered, before _await_exit's first poll
    # slice can even elapse. Poll is deliberately huge so the per-slice re-check
    # cannot be what cuts it -- only the register-time hold check can, fast.
    _tracks(tmp_path, "a.mp3")
    clk = FakeClock(0.0)
    caught = threading.Event()

    def factory(p, v):
        proc = FakeProc(autofinish=False)
        if not caught.is_set():
            caught.set()
            player.on_dog_caught(None)  # catch in the spawn window
        return proc

    spawn = FakeSpawn(factory)
    player = _player(tmp_path, spawn, rt=_runtime(enabled=True, resume=45.0),
                     clock=clk, poll=5.0)  # a slice is 5s; the per-slice check is too slow
    with running(player):
        _wait_for(lambda: spawn.procs and spawn.procs[0].terminated, timeout=1.0)


def test_await_exit_cuts_track_when_hold_armed_mid_flight(tmp_path):
    # Second seam: a hold can be armed while a track is already registered and
    # mid-flight without a catch's terminate() reaching THIS proc. _await_exit
    # must re-check the hold per slice and cut the track, or a calm track
    # outlives the hold. Arm the hold directly (no catch touches the proc) so the
    # ONLY thing that can stop the track is _await_exit's per-slice hold re-check.
    _tracks(tmp_path, "a.mp3", "b.mp3")
    clk = FakeClock(0.0)
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    player = _player(tmp_path, spawn, rt=_runtime(enabled=True, resume=45.0), clock=clk)
    with running(player):
        _wait_for(lambda: spawn.names[:1] == ["a.mp3"])  # registered, in _await_exit
        proc_a = spawn.procs[0]
        with player._lock:
            player._hold_until = 45.0  # armed after registration; proc untouched
        _wait_for(lambda: proc_a.terminated)  # only the hold re-check can do this
        clk.t = 50.0
        _wait_for(lambda: len(spawn.calls) == 2)
        assert spawn.calls[1][0].name == "b.mp3"


def test_unplayable_track_is_skipped(tmp_path):
    _tracks(tmp_path, "a.mp3", "b.mp3")

    def factory(p, v):
        return None if Path(p).name == "a.mp3" else FakeProc(autofinish=True)

    spawn = FakeSpawn(factory)
    with running(_player(tmp_path, spawn, rt=_runtime(enabled=True))):
        _wait_for(lambda: "b.mp3" in spawn.names)  # skipped a, played b, no crash


def test_all_failing_pass_backs_off(tmp_path):
    _tracks(tmp_path, "a.mp3", "b.mp3", "c.mp3")
    spawn = FakeSpawn(lambda p, v: FakeProc(code=1, autofinish=True))  # every play fails
    with running(_player(tmp_path, spawn, rt=_runtime(enabled=True), poll=0.05)):
        time.sleep(0.3)
        n = len(spawn.calls)
    # It kept retrying (not stalled) yet did NOT hot-loop: with a ~0.05s idle
    # after every failing pass of 3, ~0.3s allows only a handful of passes.
    # A hot loop would rack up many thousands of spawns here.
    assert n >= 3
    assert n < 200


def test_empty_library_idles_then_picks_up_new_file(tmp_path):
    spawn = FakeSpawn(lambda p, v: FakeProc(autofinish=False))
    status = StatusStore()
    with running(_player(tmp_path, spawn, rt=_runtime(enabled=True), status=status)):
        time.sleep(0.1)
        assert spawn.calls == []
        assert status.snapshot().soothing_track is None
        (tmp_path / "a.mp3").write_bytes(b"x")
        _wait_for(lambda: spawn.names[:1] == ["a.mp3"])


# -- default _spawn_player backend selection -------------------------------


def test_spawn_player_prefers_pw_play_with_volume(monkeypatch, tmp_path):
    import doggy.reaction.soothing as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which",
                        lambda n: "/usr/bin/pw-play" if n == "pw-play" else None)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    player = SoothingPlayer(_runtime(), tmp_path, StatusStore())
    proc = player._spawn_player(tmp_path / "calm.mp3", 0.3)
    assert proc is not None
    assert cmds[0] == ["/usr/bin/pw-play", "--volume", "0.3", str(tmp_path / "calm.mp3")]


def test_spawn_player_falls_back_to_pw_cat(monkeypatch, tmp_path):
    import doggy.reaction.soothing as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which",
                        lambda n: "/usr/bin/pw-cat" if n == "pw-cat" else None)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    player = SoothingPlayer(_runtime(), tmp_path, StatusStore())
    player._spawn_player(tmp_path / "calm.wav", 0.5)
    # Bare pw-cat needs an explicit --playback; pw-play (above) defaults to it.
    assert cmds[0] == [
        "/usr/bin/pw-cat", "--volume", "0.5", "--playback", str(tmp_path / "calm.wav")]


def test_spawn_player_darwin_uses_afplay(monkeypatch, tmp_path):
    import doggy.reaction.soothing as mod

    cmds = []
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(mod.sys, "platform", "darwin")
    monkeypatch.setattr(mod.subprocess, "Popen", lambda cmd: cmds.append(cmd) or object())
    player = SoothingPlayer(_runtime(), tmp_path, StatusStore())
    player._spawn_player(tmp_path / "calm.mp3", 0.5)
    assert cmds[0] == ["afplay", "-v", "0.5", str(tmp_path / "calm.mp3")]


def test_spawn_player_none_when_no_player_and_logs_once(monkeypatch, tmp_path, caplog):
    import doggy.reaction.soothing as mod

    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(mod.sys, "platform", "linux")
    player = SoothingPlayer(_runtime(), tmp_path, StatusStore())
    with caplog.at_level(logging.INFO):
        assert player._spawn_player(tmp_path / "x.mp3", 0.4) is None
        assert player._spawn_player(tmp_path / "y.mp3", 0.4) is None
    logged = [r for r in caplog.records if "no audio player" in r.message]
    assert len(logged) == 1  # logged once, then idles quietly
