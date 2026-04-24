"""Thompson Sampling for Beta-Bernoulli multi-armed bandits.

Each arm (model version) maintains a Beta(alpha, beta) posterior over its true
reward probability. On each decision:

    1. Sample theta_i ~ Beta(alpha_i, beta_i) for every arm.
    2. Pick the arm with the largest sampled theta.

After observing reward r (0/1), update the chosen arm:

    alpha_i <- alpha_i + r
    beta_i  <- beta_i  + (1 - r)

Thompson Sampling is asymptotically optimal for the Bernoulli bandit and
trivially handles delayed feedback (we just update when the label arrives).
We keep a separate DB table ``bandit_state`` per (experiment, model) so state
survives process restarts and dashboard views.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from ml_ab_platform.config import get_settings
from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.storage import get_conn

logger = get_logger(__name__)


@dataclass
class ArmState:
    """Beta posterior for a single bandit arm."""

    version: str
    alpha: float = 1.0
    beta: float = 1.0
    trials: int = 0
    successes: int = 0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    def update(self, reward: int) -> None:
        self.alpha += reward
        self.beta += 1 - reward
        self.trials += 1
        self.successes += reward


@dataclass
class ThompsonSampler:
    """Thompson Sampling over an arbitrary set of arms.

    Deterministic when a seed is supplied — important for reproducible tests
    that verify convergence toward the better arm.
    """

    arms: dict[str, ArmState] = field(default_factory=dict)
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())

    @classmethod
    def from_versions(
        cls,
        versions: Iterable[str],
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        seed: int | None = None,
    ) -> ThompsonSampler:
        arms = {v: ArmState(version=v, alpha=prior_alpha, beta=prior_beta) for v in versions}
        rng = np.random.default_rng(seed)
        return cls(arms=arms, rng=rng)

    def choose(self) -> str:
        """Sample one theta per arm and return the arm with the max."""
        samples = {v: self.rng.beta(a.alpha, a.beta) for v, a in self.arms.items()}
        return max(samples, key=samples.get)

    def update(self, version: str, reward: int) -> None:
        if version not in self.arms:
            raise KeyError(f"No arm named {version!r}")
        self.arms[version].update(reward)

    def snapshot(self) -> dict[str, dict[str, float]]:
        return {
            v: {
                "alpha": a.alpha, "beta": a.beta,
                "trials": a.trials, "successes": a.successes, "mean": a.mean,
            }
            for v, a in self.arms.items()
        }


# ------------------------------ persistence --------------------------------- #


def persist_bandit_state(experiment_id: str, sampler: ThompsonSampler) -> None:
    """Upsert all arms for an experiment into the ``bandit_state`` table."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for arm in sampler.arms.values():
            conn.execute(
                """
                INSERT INTO bandit_state(experiment_id, model_version, alpha, beta,
                                         trials, successes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id, model_version) DO UPDATE SET
                    alpha=excluded.alpha, beta=excluded.beta,
                    trials=excluded.trials, successes=excluded.successes,
                    updated_at=excluded.updated_at
                """,
                (experiment_id, arm.version, arm.alpha, arm.beta,
                 arm.trials, arm.successes, now),
            )


def load_bandit_state(
    experiment_id: str,
    versions: Iterable[str],
    seed: int | None = None,
) -> ThompsonSampler:
    """Load (or initialise) the bandit sampler for an experiment."""
    settings = get_settings()
    sampler = ThompsonSampler.from_versions(
        versions,
        prior_alpha=settings.bandit.prior_alpha,
        prior_beta=settings.bandit.prior_beta,
        seed=seed,
    )
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM bandit_state WHERE experiment_id = ?", (experiment_id,),
        ).fetchall()
    for row in rows:
        v = row["model_version"]
        if v in sampler.arms:
            sampler.arms[v] = ArmState(
                version=v,
                alpha=row["alpha"], beta=row["beta"],
                trials=row["trials"], successes=row["successes"],
            )
    return sampler


def record_bandit_choice(
    experiment_id: str,
    chosen: str,
    chosen_prob: float,
    cumulative_regret: float,
) -> None:
    """Append a row to ``bandit_history`` for later visualisation."""
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO bandit_history
               (experiment_id, timestamp, model_version, chosen_prob, cumulative_regret)
               VALUES (?, ?, ?, ?, ?)""",
            (experiment_id, now, chosen, chosen_prob, cumulative_regret),
        )


def compute_regret(
    rewards: list[tuple[str, int]],
    true_means: dict[str, float],
) -> list[float]:
    """Cumulative regret = sum over t of (max_mean - mean(chosen_t)).

    ``rewards`` is a list of (chosen_arm, observed_reward) — we only use the
    chosen arm; the observed reward is irrelevant for regret given true means.
    ``true_means`` is the ground-truth reward probability per arm.
    """
    best = max(true_means.values())
    cumulative = 0.0
    out: list[float] = []
    for version, _ in rewards:
        cumulative += best - true_means.get(version, 0.0)
        out.append(cumulative)
    return out
