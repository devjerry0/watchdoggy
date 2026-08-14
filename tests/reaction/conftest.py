import contextlib
import subprocess
import threading
import time
from pathlib import Path

from doggy.core.config import TunableSettings
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import StatusStore
from doggy.reaction.soothing import SoothingPlayer


# -- shared helpers for tests/reaction/test_soothing_*.py -------------------
#
# These are used across most of the soothing tests (loop, holds, backend
# selection), so they live here rather than in any single test module.


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


def _player(tmp_path, spawn, rt=None, status=None, clock=time.monotonic, poll=0.02,
            set_volume=None, wall_clock=time.time):
    # Default set_volume is a no-op returning False (no live apply, no real
    # subprocess to pw-dump/wpctl): keeps tests fast and host-independent, and
    # exercises the re-spawn fallback. Live-volume tests pass their own.
    return SoothingPlayer(rt or _runtime(enabled=True), tmp_path,
                          status or StatusStore(), clock=clock, spawn=spawn,
                          set_volume=set_volume or (lambda proc, v: False),
                          wall_clock=wall_clock, poll=poll)


def _tracks(tmp_path, *names):
    for n in names:
        (tmp_path / n).write_bytes(b"x")


def _backend():
    from doggy.reaction.soothing.audio import AudioBackend
    return AudioBackend(threading.Event().wait)
