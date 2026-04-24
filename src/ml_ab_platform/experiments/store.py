"""SQLite-backed experiment storage."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from ml_ab_platform.experiments.models import (
    Experiment,
    ExperimentCreate,
    ExperimentStatus,
)
from ml_ab_platform.storage import get_conn


def _row_to_experiment(row) -> Experiment:  # type: ignore[no-untyped-def]
    config = json.loads(row["config_json"]) if row["config_json"] else {}
    conclusion = json.loads(row["conclusion_json"]) if row["conclusion_json"] else None
    return Experiment(
        id=row["id"],
        name=row["name"],
        description=row["description"] or "",
        status=ExperimentStatus(row["status"]),
        routing_strategy=row["routing_strategy"],
        routing_config=config,
        model_a=row["model_a"],
        model_b=row["model_b"],
        target_metric=row["target_metric"],
        minimum_sample_size=row["minimum_sample_size"],
        created_at=datetime.fromisoformat(row["created_at"]),
        started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
        stopped_at=datetime.fromisoformat(row["stopped_at"]) if row["stopped_at"] else None,
        concluded_at=datetime.fromisoformat(row["concluded_at"]) if row["concluded_at"] else None,
        winner=row["winner"],
        conclusion=conclusion,
    )


class ExperimentStore:
    """CRUD for experiments on the shared SQLite database."""

    def create(self, data: ExperimentCreate) -> Experiment:
        exp = Experiment(
            id=str(uuid.uuid4()),
            name=data.name,
            description=data.description,
            status=ExperimentStatus.DRAFT,
            routing_strategy=data.routing_strategy,
            routing_config=data.routing_config,
            model_a=data.model_a,
            model_b=data.model_b,
            target_metric=data.target_metric,
            minimum_sample_size=data.minimum_sample_size,
            created_at=datetime.now(timezone.utc),
        )
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO experiments
                   (id, name, description, status, routing_strategy, config_json,
                    model_a, model_b, target_metric, minimum_sample_size, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (exp.id, exp.name, exp.description, exp.status.value,
                 exp.routing_strategy, json.dumps(exp.routing_config),
                 exp.model_a, exp.model_b, exp.target_metric,
                 exp.minimum_sample_size, exp.created_at.isoformat()),
            )
        return exp

    def get(self, experiment_id: str) -> Experiment | None:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,),
            ).fetchone()
        return _row_to_experiment(row) if row else None

    def list(self) -> list[Experiment]:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC",
            ).fetchall()
        return [_row_to_experiment(r) for r in rows]

    def active(self) -> Experiment | None:
        """The currently-running experiment, if any."""
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE status = 'running' LIMIT 1",
            ).fetchone()
        return _row_to_experiment(row) if row else None

    def update_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        started_at: datetime | None = None,
        stopped_at: datetime | None = None,
        concluded_at: datetime | None = None,
        winner: str | None = None,
        conclusion: dict | None = None,
    ) -> None:
        assignments = ["status = ?"]
        params: list = [status.value]
        if started_at:
            assignments.append("started_at = ?")
            params.append(started_at.isoformat())
        if stopped_at:
            assignments.append("stopped_at = ?")
            params.append(stopped_at.isoformat())
        if concluded_at:
            assignments.append("concluded_at = ?")
            params.append(concluded_at.isoformat())
        if winner is not None:
            assignments.append("winner = ?")
            params.append(winner)
        if conclusion is not None:
            assignments.append("conclusion_json = ?")
            params.append(json.dumps(conclusion))
        params.append(experiment_id)
        sql = f"UPDATE experiments SET {', '.join(assignments)} WHERE id = ?"
        with get_conn() as conn:
            conn.execute(sql, params)
