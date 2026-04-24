"""Configuration management using Pydantic Settings with YAML + env overrides."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelsConfig(BaseModel):
    artifacts_dir: str = "artifacts"
    model_a_path: str = "artifacts/model_a.joblib"
    model_b_path: str = "artifacts/model_b.joblib"
    baseline_metrics_path: str = "artifacts/baseline_metrics.json"


class DataConfig(BaseModel):
    dataset_path: str = "data/adult.csv"
    target_column: str = "income"
    test_size: float = 0.2
    random_state: int = 42


class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"


class DatabaseConfig(BaseModel):
    path: str = "artifacts/platform.db"


class RoutingConfig(BaseModel):
    default_split: float = 0.5
    canary_initial_split: float = 0.05
    canary_promotion_threshold: float = 0.95
    canary_min_samples: int = 200
    canary_degradation_threshold: float = 0.05


class StatisticsConfig(BaseModel):
    alpha: float = 0.05
    power: float = 0.8
    minimum_effect_size: float = 0.02
    minimum_sample_size: int = 200
    obrien_fleming_peeks: int = 5


class BanditConfig(BaseModel):
    prior_alpha: float = 1.0
    prior_beta: float = 1.0


class DashboardConfig(BaseModel):
    refresh_seconds: int = 5
    port: int = 8501


class SimulationConfig(BaseModel):
    default_requests: int = 2000
    default_delay_ms: int = 5
    feedback_delay_ms: int = 50
    noise: float = 0.05


class Settings(BaseSettings):
    """Top-level application settings."""

    models: ModelsConfig = Field(default_factory=ModelsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    statistics: StatisticsConfig = Field(default_factory=StatisticsConfig)
    bandit: BanditConfig = Field(default_factory=BanditConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        env_prefix="MLAB_",
        extra="ignore",
    )


def _find_config_file() -> Path | None:
    """Find the YAML config file, checking env var then default locations."""
    if env_path := os.environ.get("MLAB_CONFIG"):
        p = Path(env_path)
        if p.exists():
            return p

    here = Path.cwd()
    for candidate in [
        here / "configs" / "default.yaml",
        here.parent / "configs" / "default.yaml",
        Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml",
    ]:
        if candidate.exists():
            return candidate
    return None


def _collect_env_overrides(prefix: str = "MLAB_") -> dict[str, Any]:
    """Walk ``os.environ`` and build a nested dict mirroring the Settings shape.

    ``MLAB_DATABASE__PATH=/x`` becomes ``{"database": {"path": "/x"}}``. This
    mirrors Pydantic's own nested-env parsing, but we do it here so we can
    cleanly *merge* env-based overrides on top of YAML values — otherwise the
    YAML kwargs passed to ``Settings(...)`` would win over env vars, which is
    the opposite of what we want.
    """
    out: dict[str, Any] = {}
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        path = key[len(prefix):].lower().split("__")
        if not path or not path[0]:
            continue
        cursor = out
        for segment in path[:-1]:
            cursor = cursor.setdefault(segment, {})
        cursor[path[-1]] = value
    return out


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — ``override`` wins on conflicts."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML and allow env vars to override.

    Env var scheme: MLAB_<SECTION>__<FIELD> (e.g. MLAB_API__PORT=9000).
    """
    yaml_data: dict[str, Any] = {}
    path = Path(config_path) if config_path else _find_config_file()
    if path and path.exists():
        with open(path) as f:
            yaml_data = yaml.safe_load(f) or {}
    merged = _deep_merge(yaml_data, _collect_env_overrides())
    return Settings(**merged)


_settings: Settings | None = None


def get_settings() -> Settings:
    """Get cached application settings."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings() -> None:
    """Clear the cached settings — mostly for tests."""
    global _settings
    _settings = None
