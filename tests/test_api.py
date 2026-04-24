"""Full-lifecycle API tests against an in-memory FastAPI + temp DB."""
from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from ml_ab_platform.api.app import create_app
from ml_ab_platform.models.registry import ModelBundle, clear_cache


class _DummyClassifier:
    """Deterministic fake classifier for API tests — always predicts with
    prob 0.8 if age >= 40, else 0.3."""

    classes_ = [0, 1]

    def predict_proba(self, df: Any):  # type: ignore[no-untyped-def]
        import numpy as np
        probs = np.where(df["age"].to_numpy() >= 40, 0.8, 0.3)
        return np.stack([1 - probs, probs], axis=1)


class _FakePipeline:
    def __init__(self):
        self.named_steps = {"classifier": _DummyClassifier()}

    def predict_proba(self, df):  # type: ignore[no-untyped-def]
        return self.named_steps["classifier"].predict_proba(df)

    def predict(self, df):  # type: ignore[no-untyped-def]
        p = self.predict_proba(df)[:, 1]
        return (p >= 0.5).astype(int)


@pytest.fixture
def client(tmp_config, monkeypatch):
    """FastAPI test client with models stubbed out."""
    clear_cache()

    def fake_loader(version: str, path: str | None = None) -> ModelBundle:
        return ModelBundle(version=version, model=_FakePipeline())

    monkeypatch.setattr("ml_ab_platform.api.app.load_model_bundle", fake_loader)
    app = create_app()
    return TestClient(app)


def _features() -> dict[str, Any]:
    return {
        "age": 45, "workclass": "Private", "fnlwgt": 200000,
        "education": "Bachelors", "education_num": 13,
        "marital_status": "Married-civ-spouse", "occupation": "Exec-managerial",
        "relationship": "Husband", "race": "White", "sex": "Male",
        "capital_gain": 0, "capital_loss": 0, "hours_per_week": 40,
        "native_country": "United-States",
    }


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_requires_active_experiment(client):
    r = client.post("/predict", json={"features": _features()})
    assert r.status_code == 409


def test_full_lifecycle(client):
    # create
    r = client.post("/experiments", json={
        "name": "demo", "routing_strategy": "fixed",
        "routing_config": {"split": 0.5}, "minimum_sample_size": 10,
    })
    assert r.status_code == 201, r.text
    exp_id = r.json()["id"]

    # start
    r = client.post(f"/experiments/{exp_id}/start")
    assert r.status_code == 200
    assert r.json()["status"] == "running"

    # cannot start a second one
    r2 = client.post("/experiments", json={"name": "other", "routing_strategy": "fixed"})
    other_id = r2.json()["id"]
    r3 = client.post(f"/experiments/{other_id}/start")
    assert r3.status_code == 409

    # predict
    r = client.post("/predict", json={"features": _features(), "user_id": "u1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_version"] in ("A", "B")
    assert body["experiment_id"] == exp_id
    req_id = body["request_id"]

    # feedback
    r = client.post("/feedback", json={"request_id": req_id, "ground_truth": 1})
    assert r.status_code == 200

    # feedback on unknown request
    r = client.post("/feedback", json={"request_id": "nope", "ground_truth": 1})
    assert r.status_code == 404

    # analysis
    r = client.get(f"/experiments/{exp_id}/analysis")
    assert r.status_code == 200
    body = r.json()
    assert "verdict" in body

    # stop
    r = client.post(f"/experiments/{exp_id}/stop")
    assert r.status_code == 200
    assert r.json()["status"] == "stopped"

    # conclude
    r = client.post(f"/experiments/{exp_id}/conclude")
    assert r.status_code == 200
    assert r.json()["status"] == "concluded"


def test_fixed_split_actually_distributes(client):
    r = client.post("/experiments", json={
        "name": "split-check", "routing_strategy": "fixed",
        "routing_config": {"split": 0.5},
    })
    exp_id = r.json()["id"]
    client.post(f"/experiments/{exp_id}/start")
    counts = {"A": 0, "B": 0}
    for _ in range(400):
        r = client.post("/predict", json={"features": _features()})
        counts[r.json()["model_version"]] += 1
    share = counts["A"] / sum(counts.values())
    assert 0.4 < share < 0.6, counts


def test_sticky_routing_consistent_per_user(client):
    r = client.post("/experiments", json={
        "name": "sticky-check", "routing_strategy": "sticky",
        "routing_config": {"split": 0.5},
    })
    exp_id = r.json()["id"]
    client.post(f"/experiments/{exp_id}/start")
    for uid in ["alice", "bob", "carol"]:
        seen = set()
        for _ in range(15):
            r = client.post("/predict", json={"features": _features(), "user_id": uid})
            seen.add(r.json()["model_version"])
        assert len(seen) == 1
