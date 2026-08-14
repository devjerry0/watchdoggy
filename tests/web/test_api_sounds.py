from .conftest import _sounds_client


def test_sounds_lists_files_and_selected(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    (sounds / "bark.wav").write_bytes(b"RIFF")
    (sounds / "growl.mp3").write_bytes(b"ID3")
    (sounds / "notes.txt").write_bytes(b"nope")  # non-audio: excluded
    body = c.get("/api/sounds").json()
    assert body["sounds"] == ["bark.wav", "growl.mp3"]
    assert body["selected"] == "random"


def test_sounds_reflects_selected(tmp_path):
    c, sounds, runtime = _sounds_client(tmp_path)
    (sounds / "bark.wav").write_bytes(b"RIFF")
    c.patch("/api/settings", json={"selected_sound": "bark.wav"})
    assert c.get("/api/sounds").json()["selected"] == "bark.wav"


def test_upload_saves_wav(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("new.wav", b"RIFFDATA", "audio/wav")})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "name": "new.wav"}
    assert (sounds / "new.wav").read_bytes() == b"RIFFDATA"


def test_upload_sanitizes_path_traversal(tmp_path):
    c, sounds, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("../evil.wav", b"RIFF", "audio/wav")})
    assert r.status_code == 200
    assert r.json()["name"] == "evil.wav"
    assert (sounds / "evil.wav").is_file()


def test_upload_rejects_non_audio(tmp_path):
    c, _, _ = _sounds_client(tmp_path)
    r = c.post("/api/sounds", files={"file": ("x.txt", b"x", "text/plain")})
    assert r.status_code == 422
