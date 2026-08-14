from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from doggy.core.tunables import ArmedWindow, TunableSettings

__all__ = ["ArmedWindow", "Settings", "TunableSettings", "load_settings"]


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
    # Training-data capture storage (raw frames + JSON sidecars), oldest pruned
    # past the cap (2 GiB).
    dataset_dir: Path = Path("dataset")
    dataset_cap_bytes: int = 2_147_483_648
    # Training job queue: the web UI writes job requests here; the trainer
    # daemon (separate user, the one with cloud egress) consumes them.
    jobs_dir: Path = Path("jobs")
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
