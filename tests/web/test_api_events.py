from .conftest import _app_with_events, _app_with_store, _seeded_store


def test_events_route_serves_thumbnail(tmp_path):
    (tmp_path / "fire_1.jpg").write_bytes(b"\xff\xd8\xff")
    c = _app_with_events(tmp_path)
    r = c.get("/events/fire_1.jpg")
    assert r.status_code == 200
    assert r.content == b"\xff\xd8\xff"


def test_events_route_404_for_missing(tmp_path):
    c = _app_with_events(tmp_path)
    assert c.get("/events/nope.jpg").status_code == 404


def test_events_list_and_delete(tmp_path):
    store, ids = _seeded_store(tmp_path, 2)
    c = _app_with_store(tmp_path, store)
    r = c.get("/api/events").json()
    assert len(r["events"]) == 2 and "age_seconds" in r["events"][0]
    assert c.delete(f"/api/events/{ids[0]}").status_code == 200
    assert len(c.get("/api/events").json()["events"]) == 1
    assert c.delete("/api/events/nope").status_code == 404


def test_stats_endpoint(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    assert "today" in c.get("/api/stats").json()


def test_clear(tmp_path):
    store, _ = _seeded_store(tmp_path, 2)
    c = _app_with_store(tmp_path, store)
    c.post("/api/events/clear")
    assert c.get("/api/events").json()["events"] == []


def test_events_age_prefers_wall_time(tmp_path):
    store, _ = _seeded_store(tmp_path, 1)
    c = _app_with_store(tmp_path, store)
    ev = c.get("/api/events").json()["events"][0]
    # wall_time is a real (old) epoch -> age comes from the wall clock and is large.
    assert ev["wall_time"] is not None and ev["age_seconds"] > 0


def test_events_limit(tmp_path):
    store, _ = _seeded_store(tmp_path, 3)
    c = _app_with_store(tmp_path, store)
    assert len(c.get("/api/events", params={"limit": 1}).json()["events"]) == 1


def test_lab_endpoint_shape(tmp_path):
    import time
    store, ids = _seeded_store(tmp_path, 1)
    store.attach_sound(ids[0], "chirp.wav")
    store.attach_outcome(ids[0], clear_seconds=4.0, taken=[], wall_time=time.time())
    c = _app_with_store(tmp_path, store)
    body = c.get("/api/lab").json()
    assert "thefts_this_week" in body
    assert len(body["sounds"]) == 1
    row = body["sounds"][0]
    assert set(row) == {"sound", "plays", "completed", "deterred_rate",
                        "avg_clear_s", "wearing_off"}
    assert row["sound"] == "chirp.wav" and row["plays"] == 1


def test_index_has_deterrence_card(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    html = c.get("/").text
    assert "Deterrence" in html
    assert "/api/lab" in html


def test_index_has_soothing_card(tmp_path):
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    html = c.get("/").text
    assert "Soothing sounds" in html
    assert "Play soothing sounds" in html
    assert 'id="soothing_enabled"' in html
    assert "/api/soothing" in html


def test_clips_route_serves_and_404(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"data")
    store, _ = _seeded_store(tmp_path, 0)
    c = _app_with_store(tmp_path, store)
    assert c.get("/clips/clip.mp4").content == b"data"
    assert c.get("/clips/missing.mp4").status_code == 404
