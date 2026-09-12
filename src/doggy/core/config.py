from __future__ import annotations

import logging
from pathlib import Path

from pydantic import ValidationError
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
    """Structural config from .env/environment. A tunable left in .env from
    before the settings.json migration must not be able to stop boot: a
    stale DOGGY_CONFIDENCE=0.95 (the Sep 2026 value) now fails the
    certainty ceiling. Drop offending tunable keys and retry with defaults --
    settings.json (or its migration) decides the live value anyway."""
    try:
        return Settings()
    except ValidationError as exc:
        # Field-level errors name the key; model-level ones (the ceiling,
        # window_m <= window_n, ...) have an empty loc. Either way: if the
        # .env is valid once every TUNABLE is reset to its default, the fault
        # is a legacy tunable and boot proceeds; a structural fault re-raises.
        structural_bad = {str(err["loc"][0]) for err in exc.errors()
                          if err.get("loc")
                          and str(err["loc"][0]) not in TunableSettings.model_fields}
        if structural_bad:
            raise
        defaults = {k: f.default for k, f in TunableSettings.model_fields.items()}
        try:
            settings = Settings(**defaults)
        except ValidationError:
            raise exc from None
        logging.getLogger("doggy").warning(
            "settings: legacy .env tunables are invalid (%s); using defaults "
            "for them -- settings.json holds the live values",
            exc.errors()[0].get("msg", ""))
        return settings
