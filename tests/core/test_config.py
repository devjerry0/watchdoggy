import pytest
from pydantic import ValidationError

from doggy.core.config import Settings, TunableSettings, load_settings


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # avoid picking up a real .env
    s = load_settings()
    assert s.confidence == 0.55
    assert s.window_m == 4 and s.window_n == 6
    assert s.camera_index == 0
    assert s.web_host == "127.0.0.1"
    assert s.web_port == 8000


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DOGGY_CONFIDENCE", "0.7")
    monkeypatch.setenv("DOGGY_CAMERA_INDEX", "1")
    s = load_settings()
    assert s.confidence == 0.7
    assert s.camera_index == 1


def test_target_labels_default_and_parsing():
    assert TunableSettings().target_labels == ("dog",)
    assert TunableSettings().alert_labels == ("dog",)
    got = TunableSettings(target_labels="dog,cat", alert_labels="dog,cat")
    assert got.target_labels == ("dog", "cat") and got.alert_labels == ("dog", "cat")
    assert TunableSettings(target_labels='["cat"]', alert_labels='["cat"]').target_labels == ("cat",)
    assert TunableSettings(target_labels=["bird", "dog"], alert_labels=["dog"]).alert_labels == ("dog",)


def test_target_labels_rejects_unknown_and_empty():
    with pytest.raises(ValidationError):
        TunableSettings(target_labels="dragon")
    with pytest.raises(ValidationError):
        TunableSettings(target_labels=[])


def test_target_labels_rejects_non_list_garbage():
    # Bad PATCH payloads must surface as ValidationError (-> HTTP 422), never a
    # bare TypeError from inside the validator (-> HTTP 500).
    with pytest.raises(ValidationError):
        TunableSettings(target_labels=5)
    with pytest.raises(ValidationError):
        TunableSettings(target_labels=None)


def test_alert_labels_subset_rule():
    # detect-only birds: valid; alerting on an undetected class: not.
    ok = TunableSettings(target_labels="dog,bird", alert_labels="dog")
    assert ok.alert_labels == ("dog",)
    assert TunableSettings(target_labels="dog", alert_labels=[]).alert_labels == ()
    with pytest.raises(ValidationError):
        TunableSettings(target_labels="dog", alert_labels="cat")


def test_window_validation():
    with pytest.raises(ValidationError):
        TunableSettings(window_m=7, window_n=6)


def test_cooldown_validation():
    with pytest.raises(ValidationError):
        TunableSettings(cooldown_min_seconds=30, cooldown_max_seconds=10)


def test_confidence_range():
    with pytest.raises(ValidationError):
        TunableSettings(confidence=1.5)


def test_tunable_subset_extracted():
    s = Settings(confidence=0.6)
    t = s.tunable()
    assert isinstance(t, TunableSettings)
    assert t.confidence == 0.6
    assert not hasattr(t, "camera_index")


def test_zone_defaults_disabled():
    from doggy.core.config import Settings
    s = Settings()
    assert s.zone_enabled is False
    assert s.zone_points == []
    assert s.detect_interval_seconds == 0.7


def test_zone_points_parsed_from_env(monkeypatch):
    from doggy.core.config import Settings
    monkeypatch.setenv("DOGGY_ZONE_ENABLED", "true")
    monkeypatch.setenv("DOGGY_ZONE_POINTS", "[[0.1,0.2],[0.3,0.4],[0.5,0.1]]")
    s = Settings()
    assert s.zone_enabled is True
    assert s.zone_points == [(0.1, 0.2), (0.3, 0.4), (0.5, 0.1)]


def test_thermal_defaults():
    from doggy.core.config import Settings
    s = Settings()
    assert s.thermal_enabled is True
    assert s.thermal_target_c == 74.0
    assert s.thermal_max_c == 82.0
    assert s.thermal_cooldown_interval_seconds == 1.5

def test_thermal_target_must_be_le_max():
    import pytest
    from pydantic import ValidationError
    from doggy.core.config import TunableSettings
    with pytest.raises(ValidationError):
        TunableSettings(thermal_target_c=90.0, thermal_max_c=80.0)
