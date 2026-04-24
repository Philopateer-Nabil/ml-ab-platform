"""Model registry — lazy-loads joblib artifacts and wraps them for serving."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_ab_platform.config import get_settings
from ml_ab_platform.logging_ import get_logger

logger = get_logger(__name__)


@dataclass
class ModelBundle:
    """A loaded model with its human-readable version label."""

    version: str
    model: Any

    def predict(self, features: dict[str, Any] | pd.DataFrame) -> tuple[int, float]:
        """Predict class + probability-of-positive for a single row."""
        x = features if isinstance(features, pd.DataFrame) else pd.DataFrame([features])
        proba = float(self.model.predict_proba(x)[0, 1])
        pred = int(proba >= 0.5)
        return pred, proba

    def predict_batch(self, df: pd.DataFrame) -> tuple[list[int], list[float]]:
        """Batch predictions for a DataFrame of rows."""
        proba = self.model.predict_proba(df)[:, 1]
        preds = (proba >= 0.5).astype(int)
        return preds.tolist(), proba.tolist()


_cache: dict[str, ModelBundle] = {}
_cache_lock = threading.Lock()


def load_model_bundle(version: str, path: str | None = None) -> ModelBundle:
    """Load a model by version label (``A`` or ``B``), caching the result."""
    with _cache_lock:
        if version in _cache:
            return _cache[version]

        settings = get_settings()
        if path is None:
            if version == "A":
                path = settings.models.model_a_path
            elif version == "B":
                path = settings.models.model_b_path
            else:
                raise ValueError(f"Unknown model version: {version}")

        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model artifact missing at {path}. Run `mlab train` first."
            )

        logger.info("model.load", version=version, path=path)
        bundle = ModelBundle(version=version, model=joblib.load(path))
        _cache[version] = bundle
        return bundle


def clear_cache() -> None:
    """Drop cached models — used in tests."""
    with _cache_lock:
        _cache.clear()
