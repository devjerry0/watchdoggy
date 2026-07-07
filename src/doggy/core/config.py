from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TunableSettings(BaseModel):
    """The subset of config that can be changed live via the web UI."""

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(0.55, ge=0.0, le=1.0)
    confirm_seconds: float = Field(1.2, ge=0.0)
    window_m: int = Field(4, ge=1)
    window_n: int = Field(6, ge=1)
    cooldown_min_seconds: float = Field(12.0, ge=0.0)
    cooldown_max_seconds: float = Field(20.0, ge=0.0)
    max_volume: float = Field(0.8, ge=0.0, le=1.0)
    selected_sound: str = "random"  # filename in clips_dir, or "random"
    safety_enabled: bool = True
    max_fires_per_hour: int = Field(6, ge=0)
    clips_dir: Path = Path("sounds")
    log_level: str = "INFO"
    zone_enabled: bool = False
    zone_points: list[tuple[float, float]] = Field(default_factory=list)
    detect_interval_seconds: float = Field(0.7, ge=0.0)
    thermal_enabled: bool = True
    thermal_target_c: float = Field(74.0, ge=0.0)
    thermal_max_c: float = Field(82.0, ge=0.0)
    thermal_cooldown_interval_seconds: float = Field(1.5, ge=0.0)
    event_retention_max: int = Field(500, ge=0)   # 0 = unlimited
    event_retention_days: int = Field(30, ge=0)    # 0 = no age limit
    person_suppression_enabled: bool = True
    # A "dog" whose box overlaps a person's by at least this IoU is treated as a
    # misclassified person and suppressed (not fired on). High by design.
    person_iou_threshold: float = Field(0.85, ge=0.0, le=1.0)
    # Opt-in short video clips per catch, cut from a rolling in-memory JPEG buffer
    # (no continuous SD writes -- a clip is written only when a catch happens).
    clips_enabled: bool = False
    clip_window_seconds: float = 20
    clip_preroll_seconds: float = 5
    clip_postroll_seconds: float = 3
    clip_fps: int = Field(6, ge=1)
    clip_retention: int = Field(10, ge=0)   # 0 = unlimited

    @model_validator(mode="after")
    def _check_ranges(self) -> "TunableSettings":
        if self.window_m > self.window_n:
            raise ValueError("window_m must be <= window_n")
        if self.cooldown_min_seconds > self.cooldown_max_seconds:
            raise ValueError("cooldown_min_seconds must be <= cooldown_max_seconds")
        if self.thermal_target_c > self.thermal_max_c:
            raise ValueError("thermal_target_c must be <= thermal_max_c")
        return self


class Settings(TunableSettings, BaseSettings):
    """Full config: structural (restart-required) fields + the tunable subset."""

    model_config = SettingsConfigDict(
        env_prefix="DOGGY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    camera_backend: str = "opencv"  # opencv | file
    camera_index: int = 0
    camera_path: Path | None = None
    model_path: Path = Path("models/yolo26n.pt")
    alerter_backend: str = "sounddevice"  # sounddevice | command | log
    audio_device: str | None = None
    event_log_dir: Path = Path("events")
    web_enabled: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8000

    def tunable(self) -> TunableSettings:
        fields = TunableSettings.model_fields
        return TunableSettings(**{name: getattr(self, name) for name in fields})


def load_settings() -> Settings:
    return Settings()
