"""Shared pytest fixtures."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from ml_ab_platform import config as config_module
from ml_ab_platform.storage import reset_db


@pytest.fixture(autouse=True)
def _seed_rng() -> None:
    """Deterministic seeds — critical for statistical convergence tests."""
    random.seed(0)
    np.random.seed(0)


@pytest.fixture
def tmp_db(monkeypatch, tmp_path) -> Path:
    """Redirect the platform DB to a temp path for each test."""
    db_path = tmp_path / "platform.db"
    monkeypatch.setenv("MLAB_DATABASE__PATH", str(db_path))
    config_module.reset_settings()
    reset_db(str(db_path))
    return db_path


@pytest.fixture
def tmp_config(monkeypatch, tmp_path):
    """Isolated settings instance (temp DB + small sample-size thresholds)."""
    db_path = tmp_path / "platform.db"
    monkeypatch.setenv("MLAB_DATABASE__PATH", str(db_path))
    monkeypatch.setenv("MLAB_STATISTICS__MINIMUM_SAMPLE_SIZE", "50")
    monkeypatch.setenv("MLAB_ROUTING__CANARY_MIN_SAMPLES", "50")
    config_module.reset_settings()
    reset_db(str(db_path))
    yield config_module.get_settings()
    config_module.reset_settings()
