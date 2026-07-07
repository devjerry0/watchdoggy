from __future__ import annotations

from doggy.core.config import TunableSettings

_SYSFS_TEMP = "/sys/class/thermal/thermal_zone0/temp"


class ThermalGovernor:
    """Map CPU temperature to a detect interval (stateless, proportional).

    Cool → normal interval; hotter → longer interval (less load) up to a cap;
    unreadable temp or disabled → normal interval (inert). Never returns an
    interval faster than the configured normal one.
    """

    def __init__(self, temp_path: str = _SYSFS_TEMP) -> None:
        self._temp_path = temp_path

    def read_temp_c(self) -> float | None:
        try:
            with open(self._temp_path) as fh:
                return int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            return None

    def effective_interval(self, temp_c: float | None, cfg: TunableSettings) -> float:
        normal = cfg.detect_interval_seconds
        if temp_c is None or not cfg.thermal_enabled:
            return normal
        if temp_c <= cfg.thermal_target_c:
            return normal
        if temp_c >= cfg.thermal_max_c:
            ramped = cfg.thermal_cooldown_interval_seconds
        else:
            span = cfg.thermal_max_c - cfg.thermal_target_c
            frac = (temp_c - cfg.thermal_target_c) / span if span > 0 else 1.0
            ramped = normal + frac * (cfg.thermal_cooldown_interval_seconds - normal)
        return max(normal, ramped)
