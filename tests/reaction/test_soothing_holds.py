import threading
import time
from pathlib import Path

from doggy.core.status import StatusStore
from tests.reaction.conftest import FakeClock, FakeProc, FakeSpawn, _player, _runtime, _tracks, _wait_for, running


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
        with player._state._lock:
            player._state._hold_until = 45.0  # armed after registration; proc untouched
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
