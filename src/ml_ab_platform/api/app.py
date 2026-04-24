"""FastAPI gateway — `/predict`, `/feedback`, `/experiments/...`."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from ml_ab_platform.analysis import StatisticalAnalyzer
from ml_ab_platform.api.schemas import (
    ExperimentSummary,
    FeedbackRequest,
    FeedbackResponse,
    PredictRequest,
    PredictResponse,
)
from ml_ab_platform.experiments import (
    ExperimentCreate,
    ExperimentManager,
    ExperimentStore,
)
from ml_ab_platform.experiments.manager import ExperimentAlreadyRunningError
from ml_ab_platform.experiments.models import ExperimentStatus
from ml_ab_platform.logging_ import configure_logging, get_logger
from ml_ab_platform.models.registry import load_model_bundle
from ml_ab_platform.routing import RoutingContext
from ml_ab_platform.storage import get_conn

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
def _hash_features(features: dict[str, Any]) -> str:
    blob = json.dumps(features, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()  # noqa: S324 — content hash, not auth


def _counts(experiment_id: str) -> dict[str, int]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT model_version, COUNT(*) AS n FROM predictions
               WHERE experiment_id = ? GROUP BY model_version""",
            (experiment_id,),
        ).fetchall()
    out = {r["model_version"]: r["n"] for r in rows}
    with get_conn() as conn:
        fb = conn.execute(
            """SELECT model_version, COUNT(*) AS n FROM feedback
               WHERE experiment_id = ? GROUP BY model_version""",
            (experiment_id,),
        ).fetchall()
    for r in fb:
        out[f"{r['model_version']}_feedback"] = r["n"]
    return out


def _to_summary(exp) -> ExperimentSummary:  # type: ignore[no-untyped-def]
    return ExperimentSummary(
        id=exp.id, name=exp.name, description=exp.description,
        status=exp.status.value, routing_strategy=exp.routing_strategy,
        routing_config=exp.routing_config,
        model_a=exp.model_a, model_b=exp.model_b,
        target_metric=exp.target_metric,
        minimum_sample_size=exp.minimum_sample_size,
        created_at=exp.created_at.isoformat(),
        started_at=exp.started_at.isoformat() if exp.started_at else None,
        stopped_at=exp.stopped_at.isoformat() if exp.stopped_at else None,
        concluded_at=exp.concluded_at.isoformat() if exp.concluded_at else None,
        winner=exp.winner,
        counts=_counts(exp.id),
    )


# --------------------------------------------------------------------------- #
def create_app() -> FastAPI:
    """FastAPI app factory. Keeps tests hermetic by building a fresh app each time."""
    configure_logging()
    app = FastAPI(
        title="ML A/B Testing Platform",
        description="Production-quality gateway for model A/B experiments.",
        version="0.1.0",
    )
    store = ExperimentStore()
    manager = ExperimentManager(store=store)

    # -------------------------- health ---------------------------------- #
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # -------------------------- predict --------------------------------- #
    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        active = store.active()
        if active is None:
            raise HTTPException(
                status_code=409,
                detail="No running experiment. Start one with POST /experiments/{id}/start.",
            )

        router = manager.get_router(active)
        ctx = RoutingContext(experiment_id=active.id, user_id=req.user_id)
        decision = router.choose(ctx)
        bundle = load_model_bundle(decision.model_version)

        t0 = time.perf_counter()
        try:
            pred, proba = bundle.predict(pd.DataFrame([req.features]))
        except Exception as exc:
            logger.exception("predict.error", error=str(exc))
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        request_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO predictions
                   (experiment_id, timestamp, model_version, user_id, input_hash,
                    prediction, probability, latency_ms, request_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (active.id, now, decision.model_version, req.user_id,
                 _hash_features(req.features), pred, proba, latency_ms, request_id),
            )

        logger.info(
            "predict.routed", experiment_id=active.id,
            model_version=decision.model_version,
            strategy=decision.strategy, reason=decision.reason,
            latency_ms=round(latency_ms, 3), request_id=request_id,
        )
        return PredictResponse(
            request_id=request_id,
            experiment_id=active.id,
            model_version=decision.model_version,
            prediction=int(pred),
            probability=float(proba),
            latency_ms=latency_ms,
            routing_strategy=decision.strategy,
            routing_reason=decision.reason,
        )

    # -------------------------- feedback -------------------------------- #
    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest) -> FeedbackResponse:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT experiment_id, model_version FROM predictions WHERE request_id = ?",
                (req.request_id,),
            ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No prediction found for request_id={req.request_id}",
            )
        experiment_id = row["experiment_id"]
        model_version = row["model_version"]
        now = datetime.now(timezone.utc).isoformat()
        with get_conn() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO feedback
                   (request_id, experiment_id, model_version, ground_truth, received_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (req.request_id, experiment_id, model_version, req.ground_truth, now),
            )

        # For bandit routing, we also update the posterior: reward = 1 if the
        # prediction was correct. We pull the original prediction to compute that.
        with get_conn() as conn:
            pred_row = conn.execute(
                "SELECT prediction FROM predictions WHERE request_id = ?",
                (req.request_id,),
            ).fetchone()
        reward = int(pred_row["prediction"] == req.ground_truth)
        exp = store.get(experiment_id)
        if exp and exp.routing_strategy in ("bandit", "canary"):
            router = manager.get_router(exp)
            router.observe_feedback(req.request_id, model_version, reward)

        logger.info(
            "feedback.received",
            request_id=req.request_id,
            experiment_id=experiment_id,
            model_version=model_version,
            reward=reward,
        )
        return FeedbackResponse(
            request_id=req.request_id,
            recorded=True,
            experiment_id=experiment_id,
            model_version=model_version,
        )

    # -------------------------- experiments ----------------------------- #
    @app.post("/experiments", response_model=ExperimentSummary, status_code=201)
    def create_experiment(data: ExperimentCreate) -> ExperimentSummary:
        exp = manager.create(data)
        return _to_summary(exp)

    @app.get("/experiments", response_model=list[ExperimentSummary])
    def list_experiments() -> list[ExperimentSummary]:
        return [_to_summary(e) for e in store.list()]

    @app.get("/experiments/{experiment_id}", response_model=ExperimentSummary)
    def get_experiment(experiment_id: str) -> ExperimentSummary:
        exp = store.get(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return _to_summary(exp)

    @app.post("/experiments/{experiment_id}/start", response_model=ExperimentSummary)
    def start_experiment(experiment_id: str) -> ExperimentSummary:
        try:
            exp = manager.start(experiment_id)
        except ExperimentAlreadyRunningError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Experiment not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_summary(exp)

    @app.post("/experiments/{experiment_id}/stop", response_model=ExperimentSummary)
    def stop_experiment(experiment_id: str) -> ExperimentSummary:
        try:
            exp = manager.stop(experiment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Experiment not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return _to_summary(exp)

    @app.post("/experiments/{experiment_id}/conclude", response_model=ExperimentSummary)
    def conclude_experiment(experiment_id: str, winner: str | None = None) -> ExperimentSummary:
        # Force the experiment to be stopped first (safer state transition).
        exp = store.get(experiment_id)
        if exp is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        if exp.status == ExperimentStatus.RUNNING:
            manager.stop(experiment_id)
        return _to_summary(manager.conclude(experiment_id, winner=winner))

    @app.get("/experiments/{experiment_id}/analysis")
    def experiment_analysis(experiment_id: str) -> dict:
        if store.get(experiment_id) is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return StatisticalAnalyzer().analyze(experiment_id).to_dict()

    return app


app = create_app()
