import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from doggy.reaction.sound import FakeAlerter
from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.web import create_app
from doggy.web.routers import talk


def _client(tmp_path):
    settings = Settings(event_log_dir=tmp_path)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app)


@pytest.fixture(autouse=True)
def _release_talk_lock():
    # Safety net: if a handler ever leaks the one-talker lock, free it here so a
    # single failure can't cascade into every later test getting turned away
    # with 1013 (someone is talking).
    yield
    try:
        talk._busy.release()
    except RuntimeError:
        pass


def test_spawn_player_uses_raw_pcm_and_volume(monkeypatch):
    # --raw is the bug fix: without it pw-cat routes stdin through libsndfile,
    # which rejects headerless PCM ("Format not recognised") and no voice plays.
    captured = {}

    class _P:
        def __init__(self, cmd, **kw):
            captured["cmd"] = cmd

    monkeypatch.setattr(talk.shutil, "which",
                        lambda name: "/usr/bin/pw-cat" if name == "pw-cat" else None)
    monkeypatch.setattr(talk.subprocess, "Popen", _P)
    talk._spawn_player(0.5)
    cmd = captured["cmd"]
    assert "--raw" in cmd and "--playback" in cmd
    assert cmd[cmd.index("--volume") + 1] == "0.500"
    assert cmd[cmd.index("--rate") + 1] == "16000"
    assert cmd[cmd.index("--format") + 1] == "s16"


class FakeProc:
    def __init__(self):
        self.stdin = self
        self.written = []

    def write(self, b):
        self.written.append(bytes(b))

    def flush(self):
        pass

    def close(self):
        pass

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


class DyingProc:
    """A player that dies mid-stream: the first frame lands, then the pipe is
    broken (the Bluetooth speaker dropped, an expected event on this appliance).
    close() re-flushes buffered bytes and so ALSO raises, mirroring a real
    BufferedWriter over a dead pipe."""

    def __init__(self):
        self.stdin = self
        self.written = []

    def write(self, b):
        if self.written:
            raise BrokenPipeError(32, "Broken pipe")
        self.written.append(bytes(b))

    def flush(self):
        pass

    def close(self):
        raise BrokenPipeError(32, "Broken pipe")

    def terminate(self):
        pass

    def wait(self, timeout=None):
        pass


def test_player_dying_mid_stream_frees_the_lock(tmp_path, monkeypatch):
    # The speaker drops mid-call: the second frame's write raises BrokenPipeError
    # and so does the close() in the finally. The handler must end the session
    # gracefully and, above all, release the one-talker lock so push-to-talk is
    # not bricked until the next service restart.
    proc = DyingProc()
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: proc)
    c = _client(tmp_path)

    # TestClient re-raises any exception the handler propagates on context exit,
    # so a clean exit here proves the handler swallowed the broken pipe rather
    # than letting BrokenPipeError escape.
    with c.websocket_connect("/ws/talk") as ws:
        ws.send_bytes(b"\x01\x02")  # lands
        ws.send_bytes(b"\x03\x04")  # player is dead: write raises
    assert proc.written == [b"\x01\x02"]

    # Proof the lock was released: a fresh talker with a working player must be
    # accepted and served, not turned away with 1013.
    fresh = FakeProc()
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: fresh)
    with c.websocket_connect("/ws/talk") as ws2:
        ws2.send_bytes(b"\x05")
    assert fresh.written == [b"\x05"]


def test_spawn_failure_frees_the_lock(tmp_path, monkeypatch):
    # Spawning the player fails outright (FileNotFoundError race / fork failure).
    # The lock is acquired before the spawn, so it must still be released.
    def boom(volume):
        raise OSError("fork failed")

    monkeypatch.setattr(talk, "_spawn_player", boom)
    c = _client(tmp_path)

    try:
        with c.websocket_connect("/ws/talk"):
            pass
    except Exception:
        pass

    fresh = FakeProc()
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: fresh)
    with c.websocket_connect("/ws/talk") as ws2:
        ws2.send_bytes(b"\x05")
    assert fresh.written == [b"\x05"]


def test_talk_pipes_frames_to_player(tmp_path, monkeypatch):
    proc = FakeProc()
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: proc)
    c = _client(tmp_path)
    with c.websocket_connect("/ws/talk") as ws:
        ws.send_bytes(b"\x01\x02")
    assert proc.written == [b"\x01\x02"]


def test_second_talker_is_rejected(tmp_path, monkeypatch):
    # First talker holds the lock (blocked on receive); the second must be
    # accepted then closed with 1013 (try again later: someone is talking).
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: None)
    c = _client(tmp_path)
    with c.websocket_connect("/ws/talk"):
        with c.websocket_connect("/ws/talk") as ws2:
            with pytest.raises(WebSocketDisconnect) as exc:
                ws2.receive_bytes()
    assert exc.value.code == 1013


def test_index_has_talk_button(tmp_path):
    html = _client(tmp_path).get("/").text
    assert 'id="ptt"' in html
    assert "Hold to talk" in html
    assert "/ws/talk" in html


def test_missing_player_accepts_and_discards(tmp_path, monkeypatch):
    # No pw-cat on this host (Mac dev): the socket still opens and frames are
    # dropped without error, and the lock is released so a fresh talker works.
    monkeypatch.setattr(talk, "_spawn_player", lambda volume: None)
    c = _client(tmp_path)
    with c.websocket_connect("/ws/talk") as ws:
        ws.send_bytes(b"\x01\x02")
        ws.send_bytes(b"\x03\x04")
    with c.websocket_connect("/ws/talk") as ws:
        ws.send_bytes(b"\x05")
