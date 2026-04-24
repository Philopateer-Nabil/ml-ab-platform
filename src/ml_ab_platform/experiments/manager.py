"""Experiment lifecycle orchestrator."""
from __future__ import annotations

from datetime import datetime, timezone

from ml_ab_platform.analysis import StatisticalAnalyzer
from ml_ab_platform.experiments.models import (
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
)
from ml_ab_platform.experiments.store import ExperimentStore
from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.routing import Router, build_router

logger = get_logger(__name__)


class ExperimentAlreadyRunningError(RuntimeError):
    """Raised if a second experiment is started while one is already live."""


class ExperimentManager:
    """Lifecycle operations: create → start → stop → conclude.

    Enforces the single-active-experiment invariant and constructs the right
    :class:`Router` subclass on demand.
    """

    def __init__(self, store: ExperimentStore | None = None):
        self.store = store or ExperimentStore()

    def create(self, data: ExperimentCreate) -> Experiment:
        exp = self.store.create(data)
        logger.info("experiment.created", id=exp.id, strategy=exp.routing_strategy)
        return exp

    def start(self, experiment_id: str) -> Experiment:
        active = self.store.active()
        if active and active.id != experiment_id:
            raise ExperimentAlreadyRunningError(
                f"Experiment {active.id} ({active.name!r}) is already running. "
                "Stop it before starting another."
            )
        exp = self.store.get(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)
        if exp.status == ExperimentStatus.CONCLUDED:
            raise ValueError("Concluded experiments cannot be restarted.")
        self.store.update_status(
            experiment_id,
            ExperimentStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        logger.info("experiment.started", id=experiment_id)
        return self.store.get(experiment_id)  # type: ignore[return-value]

    def stop(self, experiment_id: str) -> Experiment:
        exp = self.store.get(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)
        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError(f"Experiment is not running (status={exp.status}).")
        self.store.update_status(
            experiment_id,
            ExperimentStatus.STOPPED,
            stopped_at=datetime.now(timezone.utc),
        )
        logger.info("experiment.stopped", id=experiment_id)
        return self.store.get(experiment_id)  # type: ignore[return-value]

    def conclude(self, experiment_id: str, winner: str | None = None) -> Experiment:
        """Mark winner and archive the full analysis result."""
        exp = self.store.get(experiment_id)
        if exp is None:
            raise KeyError(experiment_id)
        analysis = StatisticalAnalyzer().analyze(experiment_id).to_dict()
        chosen = winner
        if chosen is None:
            acc = analysis.get("accuracy_test")
            if acc and analysis["verdict"] == "significant":
                chosen = exp.model_b if acc["diff"] > 0 else exp.model_a
        self.store.update_status(
            experiment_id,
            ExperimentStatus.CONCLUDED,
            concluded_at=datetime.now(timezone.utc),
            winner=chosen,
            conclusion=analysis,
        )
        logger.info("experiment.concluded", id=experiment_id, winner=chosen)
        return self.store.get(experiment_id)  # type: ignore[return-value]

    def get_router(self, exp: Experiment) -> Router:
        """Instantiate the router for this experiment from its stored config."""
        return build_router(
            exp.routing_strategy,
            exp.id,
            [exp.model_a, exp.model_b],
            exp.routing_config,
        )
