"""Synthetic production traffic simulator.

Sends POST /predict requests to a running gateway and submits synthetic
ground-truth via POST /feedback with a configurable delay. Four scenarios
produce qualitatively different outcomes:

    * ``equal`` — both models have identical underlying accuracy
    * ``clear-winner`` — Model B clearly beats Model A (~8pp)
    * ``subtle-winner`` — Model B has a small edge (~1.5pp) — needs large n
    * ``degradation`` — Model B starts fine then degrades mid-run

We parameterise the simulation by assigning each model a *true* accuracy and
drawing the ground-truth label as "prediction matches GT w.p. accuracy". That
decouples the scenario from the actual trained model — so the same simulator
works regardless of which classifier you trained.
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests
from rich.console import Console
from rich.live import Live
from rich.table import Table

from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.models.data import load_adult, split_xy

logger = get_logger(__name__)
console = Console()


@dataclass
class ScenarioParams:
    """Configuration for a scenario: true accuracy per model + optional drift."""

    name: str
    description: str
    true_accuracy: dict[str, float]
    drift: Callable[[str, int, int], float] | None = None  # (version, step, total) → delta


def _clear_winner_drift() -> None:
    return None


SCENARIOS: dict[str, ScenarioParams] = {
    "equal": ScenarioParams(
        name="equal",
        description="Both models perform identically — test should say 'no difference'.",
        true_accuracy={"A": 0.85, "B": 0.85},
    ),
    "clear-winner": ScenarioParams(
        name="clear-winner",
        description="Model B is noticeably better — test should detect quickly.",
        true_accuracy={"A": 0.80, "B": 0.88},
    ),
    "subtle-winner": ScenarioParams(
        name="subtle-winner",
        description="Model B is slightly better — needs large sample size.",
        true_accuracy={"A": 0.850, "B": 0.865},
    ),
    "degradation": ScenarioParams(
        name="degradation",
        description="Model B starts fine, then degrades — canary should catch it.",
        true_accuracy={"A": 0.85, "B": 0.85},
        drift=lambda v, t, total: (-0.25 * (t / total)) if v == "B" and t / total > 0.3 else 0.0,
    ),
}


@dataclass
class SimulatorStats:
    """Live counters used by the Rich progress panel."""

    requests_sent: int = 0
    per_model: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "A": {"requests": 0, "correct": 0, "feedback": 0},
        "B": {"requests": 0, "correct": 0, "feedback": 0},
    })
    started_at: float = field(default_factory=time.time)


class Simulator:
    """Drives synthetic traffic against a running gateway.

    ``api_url`` is the base URL of the FastAPI gateway (no trailing slash).
    ``scenario`` is a key of :data:`SCENARIOS`.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:8000",
        scenario: str = "clear-winner",
        requests_per_run: int = 2000,
        delay_ms: int = 5,
        feedback_delay_ms: int = 50,
        seed: int = 0,
        timeout: float = 5.0,
        data_path: str | None = None,
    ):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario {scenario!r}. Known: {list(SCENARIOS)}")
        self.api_url = api_url.rstrip("/")
        self.scenario = SCENARIOS[scenario]
        self.requests_per_run = requests_per_run
        self.delay_ms = delay_ms
        self.feedback_delay_ms = feedback_delay_ms
        self.seed = seed
        self.timeout = timeout
        self.rng = random.Random(seed)
        self.stats = SimulatorStats()

        df = load_adult(data_path or "data/adult.csv")
        self.x, self.y = split_xy(df)

    # ------------------------------------------------------------------ #
    def _sample_row(self) -> tuple[dict[str, Any], int]:
        idx = self.rng.randrange(len(self.x))
        row = self.x.iloc[idx].to_dict()
        # Cast numpy types to plain python for JSON-serialisability
        clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
        return clean, int(self.y.iloc[idx])

    def _effective_accuracy(self, version: str, step: int) -> float:
        base = self.scenario.true_accuracy[version]
        if self.scenario.drift is None:
            return base
        return max(0.0, min(1.0, base + self.scenario.drift(version, step, self.requests_per_run)))

    def _generate_ground_truth(
        self, version: str, prediction: int, step: int,
    ) -> int:
        """Return a synthetic ground-truth label consistent with the scenario.

        With probability = effective_accuracy(version), GT matches the
        prediction; otherwise it's flipped. This way observed accuracy per
        model converges on the scenario's configured true accuracy.
        """
        acc = self._effective_accuracy(version, step)
        if self.rng.random() < acc:
            return prediction
        return 1 - prediction

    # ------------------------------------------------------------------ #
    def _send_feedback_later(self, request_id: str, model_version: str, gt: int) -> None:
        def _task() -> None:
            time.sleep(self.feedback_delay_ms / 1000.0)
            try:
                requests.post(
                    f"{self.api_url}/feedback",
                    json={"request_id": request_id, "ground_truth": gt},
                    timeout=self.timeout,
                )
                self.stats.per_model[model_version]["feedback"] += 1
            except Exception as exc:  # pragma: no cover
                logger.warning("sim.feedback_failed", error=str(exc))
        threading.Thread(target=_task, daemon=True).start()

    # ------------------------------------------------------------------ #
    def _render_panel(self) -> Table:
        t = Table(title=f"Simulation — scenario={self.scenario.name}")
        t.add_column("Model")
        t.add_column("Requests", justify="right")
        t.add_column("Feedback", justify="right")
        t.add_column("Correct", justify="right")
        t.add_column("Observed acc", justify="right")
        for v in ("A", "B"):
            s = self.stats.per_model[v]
            acc = s["correct"] / s["feedback"] if s["feedback"] else 0.0
            t.add_row(v, str(s["requests"]), str(s["feedback"]),
                      str(s["correct"]), f"{acc:.3f}")
        elapsed = max(time.time() - self.stats.started_at, 1e-6)
        t.caption = (f"{self.stats.requests_sent}/{self.requests_per_run} requests • "
                     f"{self.stats.requests_sent / elapsed:.1f} req/s")
        return t

    # ------------------------------------------------------------------ #
    def run(self, live_display: bool = True) -> dict[str, Any]:
        """Drive the scenario. Returns a summary dict when done."""
        def _iter() -> Iterator[int]:
            with Live(self._render_panel(), console=console, refresh_per_second=4,
                      transient=False) as live:
                for i in range(self.requests_per_run):
                    self._tick(i)
                    if i % 25 == 0:
                        live.update(self._render_panel())
                    yield i
                live.update(self._render_panel())

        if live_display:
            for _ in _iter():
                pass
        else:
            for i in range(self.requests_per_run):
                self._tick(i)

        # Drain any in-flight feedback threads
        time.sleep((self.feedback_delay_ms + 50) / 1000.0)
        return {
            "scenario": self.scenario.name,
            "requests_sent": self.stats.requests_sent,
            "per_model": self.stats.per_model,
            "finished_at": datetime.utcnow().isoformat(),
        }

    def _tick(self, step: int) -> None:
        features, true_label = self._sample_row()
        user_id = f"user-{self.rng.randrange(5000)}"
        try:
            resp = requests.post(
                f"{self.api_url}/predict",
                json={"features": features, "user_id": user_id},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("sim.predict_failed", error=str(exc), step=step)
            return
        body = resp.json()
        version = body["model_version"]
        request_id = body["request_id"]
        prediction = int(body["prediction"])
        self.stats.requests_sent += 1
        self.stats.per_model[version]["requests"] += 1

        # Generate synthetic GT consistent with the scenario's true accuracy.
        # NOTE: we ignore ``true_label`` from the dataset and instead use the
        # scenario-defined distribution — this is what lets us produce
        # "clear-winner" vs "degradation" scenarios without retraining.
        gt = self._generate_ground_truth(version, prediction, step)
        _ = true_label  # unused; kept for reference
        if gt == prediction:
            self.stats.per_model[version]["correct"] += 1

        self._send_feedback_later(request_id, version, gt)
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
