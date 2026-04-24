"""Canary mode — promotion when healthy, degradation detection when not."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ml_ab_platform.experiments import ExperimentCreate, ExperimentStore
from ml_ab_platform.routing import CanaryRouter, RoutingContext
from ml_ab_platform.storage import get_conn


def _seed_canary(exp_id: str, n_a: int, n_b: int, acc_a: float, acc_b: float) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for i in range(n_a):
            rid = str(uuid.uuid4())
            gt = 1 if i < int(acc_a * n_a) else 0
            conn.execute(
                """INSERT INTO predictions
                   (experiment_id, timestamp, model_version, user_id, input_hash,
                    prediction, probability, latency_ms, request_id)
                   VALUES (?, ?, 'A', NULL, 'h', 1, 0.9, 5.0, ?)""",
                (exp_id, now, rid),
            )
            conn.execute(
                """INSERT INTO feedback (request_id, experiment_id, model_version,
                       ground_truth, received_at) VALUES (?, ?, 'A', ?, ?)""",
                (rid, exp_id, gt, now),
            )
        for i in range(n_b):
            rid = str(uuid.uuid4())
            gt = 1 if i < int(acc_b * n_b) else 0
            conn.execute(
                """INSERT INTO predictions
                   (experiment_id, timestamp, model_version, user_id, input_hash,
                    prediction, probability, latency_ms, request_id)
                   VALUES (?, ?, 'B', NULL, 'h', 1, 0.9, 6.0, ?)""",
                (exp_id, now, rid),
            )
            conn.execute(
                """INSERT INTO feedback (request_id, experiment_id, model_version,
                       ground_truth, received_at) VALUES (?, ?, 'B', ?, ?)""",
                (rid, exp_id, gt, now),
            )


def test_canary_promotes_when_healthy(tmp_config):
    store = ExperimentStore()
    exp = store.create(ExperimentCreate(
        name="canary-promote", routing_strategy="canary",
        routing_config={"initial_split": 0.05, "promoted_split": 0.5,
                        "min_samples": 100, "degradation_threshold": 0.05},
    ))
    _seed_canary(exp.id, n_a=200, n_b=200, acc_a=0.80, acc_b=0.82)
    router = CanaryRouter(exp.id, ["A", "B"], exp.routing_config)
    # first choose triggers evaluation & promotion
    router.choose(RoutingContext(experiment_id=exp.id))
    with get_conn() as conn:
        row = conn.execute(
            "SELECT promoted, degraded, current_split FROM canary_state WHERE experiment_id=?",
            (exp.id,),
        ).fetchone()
    assert row["promoted"] == 1
    assert row["degraded"] == 0
    assert row["current_split"] == 0.5


def test_canary_degrades_when_worse(tmp_config):
    store = ExperimentStore()
    exp = store.create(ExperimentCreate(
        name="canary-degrade", routing_strategy="canary",
        routing_config={"initial_split": 0.05, "promoted_split": 0.5,
                        "min_samples": 100, "degradation_threshold": 0.05},
    ))
    _seed_canary(exp.id, n_a=200, n_b=200, acc_a=0.85, acc_b=0.55)
    router = CanaryRouter(exp.id, ["A", "B"], exp.routing_config)
    router.choose(RoutingContext(experiment_id=exp.id))
    with get_conn() as conn:
        row = conn.execute(
            "SELECT promoted, degraded, current_split FROM canary_state WHERE experiment_id=?",
            (exp.id,),
        ).fetchone()
    assert row["degraded"] == 1
    assert row["current_split"] == 0.0
