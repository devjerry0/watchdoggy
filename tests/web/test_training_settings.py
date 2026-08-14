from .conftest import _training_client as _client


def test_settings_roundtrip_and_validation(tmp_path):
    c, _ = _client(tmp_path)
    d = c.get("/api/training/settings").json()
    assert d["epochs"] == 200 and d["augment"] is True and d["batch"] == "auto"
    r = c.post("/api/training/settings",
               json={"epochs": 120, "augment": False, "nightly_prelabel_hour": 3})
    assert r.json()["settings"]["epochs"] == 120
    d = c.get("/api/training/settings").json()
    assert d["epochs"] == 120 and d["augment"] is False and d["batch"] == "auto"
    assert c.post("/api/training/settings",
                  json={"epochs": 9999}).status_code == 422
    assert c.post("/api/training/settings",
                  json={"augment": "yes"}).status_code == 422


def test_request_carries_validated_params(tmp_path):
    c, root = _client(tmp_path)
    d = c.post("/api/training/request",
               json={"kind": "train",
                     "params": {"epochs": 80, "augment": False}}).json()
    assert d["job"]["params"] == {"epochs": 80, "augment": False}
    assert c.post("/api/training/request",
                  json={"kind": "prelabel",
                        "params": {"epochs": 0}}).status_code == 422


def test_gpu_setting_validated(tmp_path):
    c, _ = _client(tmp_path)
    assert c.get("/api/training/settings").json()["gpu"] == "auto"
    assert c.post("/api/training/settings",
                  json={"gpu": "A10G"}).json()["settings"]["gpu"] == "A10G"
    assert c.post("/api/training/settings",
                  json={"gpu": "H9000"}).status_code == 422


def test_batch_setting_accepts_auto_and_tiers_only(tmp_path):
    c, _ = _client(tmp_path)
    assert c.post("/api/training/settings",
                  json={"batch": 32}).json()["settings"]["batch"] == 32
    assert c.post("/api/training/settings",
                  json={"batch": "auto"}).json()["settings"]["batch"] == "auto"
    assert c.post("/api/training/settings", json={"batch": 13}).status_code == 422
    assert c.post("/api/training/request",
                  json={"kind": "train",
                        "params": {"batch": "auto"}}).json()["job"]["params"]["batch"] == "auto"
