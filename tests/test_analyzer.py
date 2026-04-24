"""Integration tests for the StatisticalAnalyzer against a fake DB."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from ml_ab_platform.analysis import StatisticalAnalyzer
from ml_ab_platform.experiments import ExperimentCreate, ExperimentStore
from ml_ab_platform.storage import get_conn


def _seed_experiment(n_a: int, n_b: int, acc_a: float, acc_b: float) -> str:
    """Create an experiment + N predictions + feedback with given per-arm accuracy."""
    store = ExperimentStore()
    exp = store.create(ExperimentCreate(name="unit", routing_strategy="fixed"))
    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i in range(n_a):
        rid = str(uuid.uuid4())
        pred = 1
        gt = 1 if i < int(acc_a * n_a) else 0  # controlled accuracy
        rows.append(("A", rid, pred, gt, 5.0 + (i % 10)))
    for i in range(n_b):
        rid = str(uuid.uuid4())
        pred = 1
        gt = 1 if i < int(acc_b * n_b) else 0
        rows.append(("B", rid, pred, gt, 7.0 + (i % 10)))
    with get_conn() as conn:
        for ver, rid, pred, gt, lat in rows:
            conn.execute(
                """INSERT INTO predictions (experiment_id, timestamp, model_version,
                       user_id, input_hash, prediction, probability, latency_ms, request_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp.id, now, ver, None, "h", pred, 0.9, lat, rid),
            )
            conn.execute(
                """INSERT INTO feedback (request_id, experiment_id, model_version,
                       ground_truth, received_at) VALUES (?, ?, ?, ?, ?)""",
                (rid, exp.id, ver, gt, now),
            )
    return exp.id


def test_analyzer_detects_clear_winner(tmp_config):
    exp_id = _seed_experiment(n_a=1000, n_b=1000, acc_a=0.7, acc_b=0.85)
    result = StatisticalAnalyzer().analyze(exp_id).to_dict()
    assert result["verdict"] == "significant"
    assert result["accuracy_test"]["p_value"] < 1e-5
    assert result["accuracy_test"]["diff"] > 0.1


def test_analyzer_no_difference(tmp_config):
    exp_id = _seed_experiment(n_a=1000, n_b=1000, acc_a=0.80, acc_b=0.80)
    result = StatisticalAnalyzer().analyze(exp_id).to_dict()
    assert result["verdict"] in ("no_difference", "not_yet")
    assert result["accuracy_test"]["p_value"] > 0.05


def test_analyzer_warns_on_small_sample(tmp_config):
    exp_id = _seed_experiment(n_a=10, n_b=10, acc_a=0.7, acc_b=0.9)
    result = StatisticalAnalyzer().analyze(exp_id).to_dict()
    assert result["verdict"] == "not_enough_data"
    assert any("sample size" in w.lower() for w in result["warnings"])


def test_analyzer_latency_welch(tmp_config):
    exp_id = _seed_experiment(n_a=500, n_b=500, acc_a=0.8, acc_b=0.8)
    result = StatisticalAnalyzer().analyze(exp_id).to_dict()
    assert result["latency_test"] is not None
    # Model B latencies are +2 ms shift — should be detectable
    assert result["latency_test"]["diff"] > 1.5
    assert result["latency_test"]["p_value"] < 0.01
