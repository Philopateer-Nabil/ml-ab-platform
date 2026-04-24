"""Model training pipeline — swap any sklearn-compatible classifier.

Both models are wrapped in a ``Pipeline`` with a ``ColumnTransformer`` so they
accept raw DataFrames at serve time. The pipeline is the unit that gets pickled
with joblib.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from ml_ab_platform.config import get_settings
from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.models.data import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    load_adult,
    split_xy,
)

logger = get_logger(__name__)


def _make_preprocessor() -> ColumnTransformer:
    """ColumnTransformer that encodes categoricals and scales numerics."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_model_a(random_state: int = 42) -> Pipeline:
    """Model A: RandomForest — shallow, fast, good baseline."""
    return Pipeline([
        ("preprocessor", _make_preprocessor()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
    ])


def build_model_b(random_state: int = 42) -> Pipeline:
    """Model B: XGBoost — gradient boosted trees, typically higher AUC."""
    return Pipeline([
        ("preprocessor", _make_preprocessor()),
        ("classifier", XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Compute standard classification metrics on a held-out set."""
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "auc": float(roc_auc_score(y_test, probs)),
        "n_test": int(len(y_test)),
    }


def train_and_save_models(
    model_a: Pipeline | None = None,
    model_b: Pipeline | None = None,
    data_path: str | None = None,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Train Model A and Model B, save joblib artifacts + baseline metrics JSON.

    Any sklearn-compatible classifier wrapped in a Pipeline can be passed in,
    making it easy to swap the default RF / XGB choice without touching the
    platform code.
    """
    settings = get_settings()
    data_path = data_path or settings.data.dataset_path
    random_state = random_state if random_state is not None else settings.data.random_state

    logger.info("training.start", data_path=data_path, random_state=random_state)
    df = load_adult(data_path)
    x, y = split_xy(df, target=settings.data.target_column)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=settings.data.test_size, random_state=random_state, stratify=y,
    )

    mdl_a = model_a or build_model_a(random_state)
    mdl_b = model_b or build_model_b(random_state)

    logger.info("training.fit_model_a")
    mdl_a.fit(x_train, y_train)
    logger.info("training.fit_model_b")
    mdl_b.fit(x_train, y_train)

    metrics_a = evaluate_model(mdl_a, x_test, y_test)
    metrics_b = evaluate_model(mdl_b, x_test, y_test)
    logger.info("training.metrics", model_a=metrics_a, model_b=metrics_b)

    artifacts_dir = Path(settings.models.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(mdl_a, settings.models.model_a_path)
    joblib.dump(mdl_b, settings.models.model_b_path)

    baseline = {
        "model_a": {
            "name": "A",
            "class": type(mdl_a.named_steps["classifier"]).__name__,
            "path": settings.models.model_a_path,
            "metrics": metrics_a,
        },
        "model_b": {
            "name": "B",
            "class": type(mdl_b.named_steps["classifier"]).__name__,
            "path": settings.models.model_b_path,
            "metrics": metrics_b,
        },
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(np.mean(y_train)),
        "random_state": random_state,
    }
    Path(settings.models.baseline_metrics_path).write_text(json.dumps(baseline, indent=2))
    logger.info("training.done", baseline_path=settings.models.baseline_metrics_path)
    return baseline
