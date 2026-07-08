from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# "HH:MM" on a 24-hour clock (00:00 .. 23:59).
_HHMM = re.compile(r"([01]\d|2[0-3]):[0-5]\d")


class ArmedWindow(BaseModel):
    """One weekly arming window: on the given days, react from ``start`` to
    ``end`` ("HH:MM"). When ``end <= start`` the window wraps past midnight and
    belongs to its start day (e.g. 21:00-07:00 covers that night into the next
    morning)."""

    model_config = ConfigDict(frozen=True)

    days: tuple[int, ...]  # 0 = Monday .. 6 = Sunday
    start: str
    end: str

    @field_validator("start", "end")
    @classmethod
    def _valid_time(cls, v: str) -> str:
        if not _HHMM.fullmatch(v):
            raise ValueError(f"time must be HH:MM (00:00-23:59), got {v!r}")
        return v

    @field_validator("days")
    @classmethod
    def _valid_days(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("pick at least one day for the window")
        if any(d < 0 or d > 6 for d in v):
            raise ValueError("day must be 0..6 (0=Monday)")
        return v


class TunableSettings(BaseModel):
    """The subset of config that can be changed live via the web UI."""

    model_config = ConfigDict(frozen=True)

    confidence: float = Field(0.55, ge=0.0, le=1.0)
    # Which animals are detected (drawn + counted), comma-separated in .env
    # ("dog,cat"), and which of those may fire the deterrent. alert_labels
    # must be a subset of target_labels; empty alert_labels = monitor mode.
    # NoDecode: hand the raw .env string to _parse_labels (else pydantic-settings
    # insists on JSON and rejects the comma form).
    target_labels: Annotated[tuple[str, ...], NoDecode] = ("dog",)
    alert_labels: Annotated[tuple[str, ...], NoDecode] = ("dog",)
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
    # Least fraction of an animal's box that must lie inside the watch area
    # before it can trigger; stops a box that only scrapes the edge. Inventory
    # ignores this (see ZoneInclusionFilter.apply).
    zone_overlap: float = Field(0.4, ge=0.0, le=1.0)
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
    # Counter inventory: food/tableware detection for the readout + theft diffs.
    inventory_enabled: bool = True
    # Overhead food shots score low; independent, laxer threshold.
    inventory_confidence: float = Field(0.4, ge=0.0, le=1.0)
    show_inventory_boxes: bool = False
    # Opt-in short video clips per catch, cut from a rolling in-memory JPEG buffer
    # (no continuous SD writes -- a clip is written only when a catch happens).
    clips_enabled: bool = False
    clip_window_seconds: float = 20
    clip_preroll_seconds: float = 5
    clip_postroll_seconds: float = 3
    clip_fps: int = Field(6, ge=1)
    clip_retention: int = Field(10, ge=0)   # 0 = unlimited
    # Escalation: fire again, louder, while the animal stands its ground.
    escalation_enabled: bool = False
    escalation_seconds: float = Field(8.0, ge=1)
    escalation_max_strikes: int = Field(3, ge=1)
    escalation_volume_step: float = Field(0.2, ge=0, le=1)
    # Soothing sounds: loop uploaded calm audio through the speaker. A confirmed
    # catch cuts the current track and holds playback for soothing_resume_seconds
    # (long by default so one catch covers a whole escalation sequence, whose
    # strikes publish no hub event -- otherwise music would resume between strikes).
    soothing_enabled: bool = False
    soothing_volume: float = Field(0.4, ge=0.0, le=1.0)
    soothing_resume_seconds: float = Field(45.0, ge=0.0)
    # Weekly arming schedule: when on, reactions only happen inside armed_windows
    # (detection keeps running around the clock). Empty windows = always armed.
    # NoDecode: the .env form is a JSON string; _parse_windows json.loads it (the
    # same trick the label fields use for their comma form).
    schedule_enabled: bool = False
    armed_windows: Annotated[tuple[ArmedWindow, ...], NoDecode] = ()

    # Mirrors doggy.vision.detection.ANIMAL_TARGETS (importing it would create
    # a core -> vision cycle).
    _ALLOWED_TARGETS: ClassVar[tuple[str, ...]] = ("dog", "cat", "bird")

    @field_validator("target_labels", "alert_labels", mode="before")
    @classmethod
    def _parse_labels(cls, v):
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                v = json.loads(s)
            else:
                v = [part.strip() for part in s.split(",") if part.strip()]
        if not isinstance(v, (list, tuple)):
            # ValueError -> ValidationError -> 422; a bare TypeError would 500.
            raise ValueError("watch classes must be a list of class names")
        labels = tuple(dict.fromkeys(v))  # de-dupe, keep order
        unknown = [x for x in labels if x not in cls._ALLOWED_TARGETS]
        if unknown:
            raise ValueError(f"unknown watch classes: {unknown}")
        return labels

    @field_validator("armed_windows", mode="before")
    @classmethod
    def _parse_windows(cls, v):
        # NoDecode hands the raw .env string here; other callers pass a list of
        # dicts/models straight through for pydantic to coerce into ArmedWindow.
        if isinstance(v, str):
            s = v.strip()
            v = json.loads(s) if s else []
        return v

    @model_validator(mode="after")
    def _check_ranges(self) -> "TunableSettings":
        if self.window_m > self.window_n:
            raise ValueError("window_m must be <= window_n")
        if self.cooldown_min_seconds > self.cooldown_max_seconds:
            raise ValueError("cooldown_min_seconds must be <= cooldown_max_seconds")
        if self.thermal_target_c > self.thermal_max_c:
            raise ValueError("thermal_target_c must be <= thermal_max_c")
        if not self.target_labels:
            raise ValueError("select at least one animal to watch for")
        extra = [x for x in self.alert_labels if x not in self.target_labels]
        if extra:
            raise ValueError(f"alert classes must also be detected: {extra}")
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
    # Soothing sounds library: calm audio users upload for the looping player.
    # soothing_limit_bytes caps the whole library (1 GiB) and each single file.
    soothing_dir: Path = Path("soothing")
    soothing_limit_bytes: int = 1_073_741_824
    web_enabled: bool = True
    web_host: str = "127.0.0.1"
    web_port: int = 8000
    # Optional TLS: set both to serve https; needed for mic + notifications.
    # With TLS on, the dashboard moves to ssl_port and web_port serves the
    # onboarding door (see web/door.py).
    ssl_cert: Path | None = None
    ssl_key: Path | None = None
    ssl_port: int = 8443
    ca_cert: Path | None = None  # served at /ca.pem so devices can trust the home CA

    def tunable(self) -> TunableSettings:
        fields = TunableSettings.model_fields
        return TunableSettings(**{name: getattr(self, name) for name in fields})


def load_settings() -> Settings:
    return Settings()
