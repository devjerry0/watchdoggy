from fastapi.testclient import TestClient

from doggy.reaction.sound import FakeAlerter
from doggy.core.config import Settings
from doggy.events.store import EventStore
from doggy.decision.gate import FireGate
from doggy.core.runtime import RuntimeSettings
from doggy.core.status import FrameBuffer, StatusStore
from doggy.web import create_app


def _client(tmp_path, limit=None):
    soothing = tmp_path / "soothing"
    kwargs = {"event_log_dir": tmp_path, "soothing_dir": soothing}
    if limit is not None:
        kwargs["soothing_limit_bytes"] = limit
    settings = Settings(**kwargs)
    runtime = RuntimeSettings(settings.tunable())
    store = EventStore(tmp_path, 100, 0)
    app = create_app(settings, runtime, FrameBuffer(), StatusStore(), FakeAlerter(),
                     store, FireGate(runtime))
    return TestClient(app), soothing


def test_soothing_list_empty(tmp_path):
    c, _ = _client(tmp_path)
    body = c.get("/api/soothing").json()
    assert body["tracks"] == []
    assert body["total_bytes"] == 0
    assert body["limit_bytes"] == 1_073_741_824


def test_soothing_upload_appears_with_totals(tmp_path):
    c, soothing = _client(tmp_path)
    payload = b"RIFF" + b"\x00" * 100
    r = c.post("/api/soothing", files={"file": ("calm.wav", payload, "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"name": "calm.wav"}
    assert (soothing / "calm.wav").read_bytes() == payload
    body = c.get("/api/soothing").json()
    assert body["tracks"] == [{"name": "calm.wav", "size": len(payload)}]
    assert body["total_bytes"] == len(payload)


def test_soothing_list_is_name_sorted_and_excludes_non_audio(tmp_path):
    c, soothing = _client(tmp_path)
    soothing.mkdir()
    (soothing / "zebra.mp3").write_bytes(b"ID3")
    (soothing / "apple.wav").write_bytes(b"RIFF")
    (soothing / "notes.txt").write_bytes(b"nope")      # non-audio: excluded
    (soothing / ".upload.part").write_bytes(b"partial")  # dotfile: excluded
    body = c.get("/api/soothing").json()
    assert [t["name"] for t in body["tracks"]] == ["apple.wav", "zebra.mp3"]


def test_soothing_upload_over_limit_413_and_cleans_up(tmp_path):
    c, soothing = _client(tmp_path, limit=1000)
    payload = b"\x00" * 2048  # 2 KB > 1000-byte limit
    r = c.post("/api/soothing", files={"file": ("big.wav", payload, "audio/wav")})
    assert r.status_code == 413
    assert r.json() == {"detail": "That would go over the 1 GB limit. Delete a track first."}
    # partial file cleaned up and library unchanged (nothing landed). The temp file is
    # Unlike the glob MODULE, pathlib's Path.glob does match dotfiles, so
    # "*.part" would work too; ".upload.*" is used as the most specific pattern.
    assert list(soothing.glob(".upload.*")) == []
    assert list(soothing.glob("*")) == []
    assert c.get("/api/soothing").json()["tracks"] == []


def test_soothing_replace_near_limit_ok(tmp_path):
    # Re-uploading a track that already exists must not count its old bytes twice:
    # os.replace overwrites it, so the same 600-byte file fits under a 1000-byte cap.
    c, soothing = _client(tmp_path, limit=1000)
    payload = b"\x00" * 600
    assert c.post("/api/soothing",
                  files={"file": ("calm.wav", payload, "audio/wav")}).status_code == 200
    r = c.post("/api/soothing", files={"file": ("calm.wav", payload, "audio/wav")})
    assert r.status_code == 200
    body = c.get("/api/soothing").json()
    assert body["tracks"] == [{"name": "calm.wav", "size": 600}]
    assert body["total_bytes"] == 600
    assert list(soothing.glob(".upload.*")) == []


def test_soothing_over_limit_keeps_existing_tracks(tmp_path):
    # A rejected upload must leave the pre-existing library completely untouched.
    c, soothing = _client(tmp_path, limit=1000)
    soothing.mkdir(parents=True)
    (soothing / "aaa.wav").write_bytes(b"\x00" * 300)
    (soothing / "bbb.wav").write_bytes(b"\x00" * 300)
    r = c.post("/api/soothing",
               files={"file": ("ccc.wav", b"\x00" * 2048, "audio/wav")})
    assert r.status_code == 413
    body = c.get("/api/soothing").json()
    assert body["tracks"] == [
        {"name": "aaa.wav", "size": 300},
        {"name": "bbb.wav", "size": 300},
    ]
    assert body["total_bytes"] == 600
    assert (soothing / "aaa.wav").read_bytes() == b"\x00" * 300
    assert (soothing / "bbb.wav").read_bytes() == b"\x00" * 300
    assert list(soothing.glob(".upload.*")) == []


def test_soothing_oversize_content_length_rejected_early(tmp_path):
    # A body far bigger than the remaining cap trips the Content-Length precheck
    # (TestClient sets Content-Length from the body). A pre-existing track survives
    # and no part file is written.
    c, soothing = _client(tmp_path, limit=1000)
    soothing.mkdir(parents=True)
    (soothing / "seed.wav").write_bytes(b"\x00" * 200)
    r = c.post("/api/soothing",
               files={"file": ("big.wav", b"\x00" * 20000, "audio/wav")})
    assert r.status_code == 413
    assert r.json() == {"detail": "That would go over the 1 GB limit. Delete a track first."}
    assert list(soothing.glob(".upload.*")) == []
    body = c.get("/api/soothing").json()
    assert body["tracks"] == [{"name": "seed.wav", "size": 200}]
    assert (soothing / "seed.wav").read_bytes() == b"\x00" * 200


def test_soothing_upload_rejects_bad_extension(tmp_path):
    c, _ = _client(tmp_path)
    r = c.post("/api/soothing", files={"file": ("x.txt", b"x", "text/plain")})
    assert r.status_code == 415


def test_soothing_delete_then_404(tmp_path):
    c, soothing = _client(tmp_path)
    c.post("/api/soothing", files={"file": ("calm.wav", b"RIFF", "audio/wav")})
    assert (soothing / "calm.wav").is_file()
    assert c.delete("/api/soothing/calm.wav").status_code == 200
    assert not (soothing / "calm.wav").exists()
    assert c.delete("/api/soothing/calm.wav").status_code == 404


def test_soothing_upload_sanitizes_path_traversal(tmp_path):
    c, soothing = _client(tmp_path)
    r = c.post("/api/soothing", files={"file": ("../evil.mp3", b"ID3", "audio/mpeg")})
    assert r.status_code == 200
    assert r.json()["name"] == "evil.mp3"
    assert (soothing / "evil.mp3").is_file()
    # nothing escaped the soothing dir (e.g. into tmp_path itself)
    assert not (tmp_path / "evil.mp3").exists()
