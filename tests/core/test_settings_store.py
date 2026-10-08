"""settings.json is the tunables' home: migrated from .env on first boot,
written atomically, every change logged with who made it."""
import json
import os

from doggy.core.settings_store import (load_tunables, read_changelog,
                                       read_confidence, save_tunables)
from doggy.core.tunables import TunableSettings


def test_first_boot_migrates_from_fallback_and_persists(tmp_path):
    path = tmp_path / "settings.json"
    fallback = TunableSettings(confidence=0.6, window_m=2, window_n=4)
    loaded = load_tunables(fallback, path=path)
    assert loaded == fallback
    payload = json.loads(path.read_text())
    assert payload["version"] == 1
    assert payload["tunables"]["confidence"] == 0.6
    # The migration is itself logged, attributed, one line per key.
    log = read_changelog(limit=1000, changelog=tmp_path / "settings-changes.jsonl")
    assert log and all(r["by"] == "migration" for r in log)
    assert {r["key"] for r in log} == set(TunableSettings.model_fields)


def test_settings_json_wins_over_fallback(tmp_path):
    path = tmp_path / "settings.json"
    save_tunables(TunableSettings(confidence=0.7), changed_by="test",
                  old=None, path=path)
    stale_env_derived = TunableSettings(confidence=0.95 - 0.2)  # .env says 0.75
    assert load_tunables(stale_env_derived, path=path).confidence == 0.7


def test_unreadable_file_is_set_aside_not_overwritten(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"version": 1, "tunables": {"confidence": 9}}')  # invalid
    fallback = TunableSettings(confidence=0.6)
    assert load_tunables(fallback, path=path).confidence == 0.6
    assert not path.exists()
    aside = [p for p in tmp_path.iterdir() if p.name.startswith("settings.json.bad-")]
    assert len(aside) == 1 and '"confidence": 9' in aside[0].read_text()


def test_save_is_atomic_and_logs_only_changed_keys(tmp_path):
    path = tmp_path / "settings.json"
    old = TunableSettings(confidence=0.6, confirm_seconds=1.0)
    new = TunableSettings(confidence=0.7, confirm_seconds=1.0)
    changes = save_tunables(new, changed_by="192.168.50.16", old=old, path=path)
    assert [(c["key"], c["old"], c["new"], c["by"]) for c in changes] == [
        ("confidence", 0.6, 0.7, "192.168.50.16")]
    assert not (tmp_path / ".settings.json.tmp").exists()  # renamed into place
    newest = read_changelog(limit=5, changelog=tmp_path / "settings-changes.jsonl")
    assert newest[0]["key"] == "confidence" and newest[0]["new"] == 0.7


def test_changelog_reads_newest_first_and_skips_torn_lines(tmp_path):
    log = tmp_path / "settings-changes.jsonl"
    log.write_text('{"ts": "a", "key": "k1"}\nnot json\n{"ts": "b", "key": "k2"}\n')
    assert [r["key"] for r in read_changelog(limit=10, changelog=log)] == ["k2", "k1"]
    assert [r["key"] for r in read_changelog(limit=1, changelog=log)] == ["k2"]


def test_read_confidence_for_out_of_process_readers(tmp_path):
    path = tmp_path / "settings.json"
    assert read_confidence(path) is None
    save_tunables(TunableSettings(confidence=0.65), changed_by="t", old=None, path=path)
    assert read_confidence(path) == 0.65
    path.write_text("garbage")
    assert read_confidence(path) is None


def test_list_valued_tunables_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    t = TunableSettings(zone_points=[[0.1, 0.2], [0.9, 0.2], [0.9, 0.8]],
                        target_labels=["dog", "cat"], alert_labels=["dog"])
    save_tunables(t, changed_by="t", old=None, path=path)
    back = load_tunables(TunableSettings(), path=path)
    assert back.target_labels == t.target_labels
    assert [list(p) for p in back.zone_points] == [list(p) for p in t.zone_points]


def test_concurrent_saves_use_distinct_temp_names(tmp_path, monkeypatch):
    # Two PATCHes racing must not promote each other's half-written temp
    # file: the temp path is per-writer.
    import doggy.core.settings_store as ss
    seen = []
    real_replace = ss.os.replace
    def spy(src, dst):
        seen.append(str(src)); return real_replace(src, dst)
    monkeypatch.setattr(ss.os, "replace", spy)
    path = tmp_path / "settings.json"
    save_tunables(TunableSettings(confidence=0.6), changed_by="a", old=None, path=path)
    assert ".settings.json." in seen[0] and seen[0].endswith(".tmp")
    assert seen[0] != str(tmp_path / ".settings.json.tmp")   # not the shared name
