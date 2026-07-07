from doggy.core.config import TunableSettings
from doggy.thermal import ThermalGovernor

CFG = TunableSettings(detect_interval_seconds=0.5, thermal_enabled=True,
                      thermal_target_c=74.0, thermal_max_c=82.0,
                      thermal_cooldown_interval_seconds=1.5)

def test_read_temp_c_parses_millidegrees(tmp_path):
    f = tmp_path / "temp"; f.write_text("78123\n")
    assert ThermalGovernor(str(f)).read_temp_c() == 78.123

def test_read_temp_c_missing_file_returns_none(tmp_path):
    assert ThermalGovernor(str(tmp_path / "nope")).read_temp_c() is None

def test_interval_none_temp_is_normal():
    assert ThermalGovernor().effective_interval(None, CFG) == 0.5

def test_interval_below_target_is_normal():
    assert ThermalGovernor().effective_interval(70.0, CFG) == 0.5

def test_interval_at_or_above_max_is_cooldown():
    assert ThermalGovernor().effective_interval(82.0, CFG) == 1.5
    assert ThermalGovernor().effective_interval(90.0, CFG) == 1.5

def test_interval_ramps_linearly_between():
    # midpoint 78 of [74,82] -> halfway between 0.5 and 1.5 = 1.0
    assert ThermalGovernor().effective_interval(78.0, CFG) == 1.0

def test_interval_disabled_is_normal():
    cfg = CFG.model_copy(update={"thermal_enabled": False})
    assert ThermalGovernor().effective_interval(90.0, cfg) == 0.5

def test_interval_never_faster_than_normal():
    # cooldown accidentally set below normal -> guard keeps it at normal
    cfg = CFG.model_copy(update={"detect_interval_seconds": 2.0,
                                 "thermal_cooldown_interval_seconds": 1.5})
    assert ThermalGovernor().effective_interval(90.0, cfg) == 2.0

def test_governor_picks_cooldown_when_hot(tmp_path):
    from doggy.core.config import TunableSettings
    from doggy.thermal import ThermalGovernor
    f = tmp_path / "temp"; f.write_text("83000\n")
    g = ThermalGovernor(str(f))
    cfg = TunableSettings(detect_interval_seconds=0.5, thermal_cooldown_interval_seconds=1.5)
    assert g.effective_interval(g.read_temp_c(), cfg) == 1.5
