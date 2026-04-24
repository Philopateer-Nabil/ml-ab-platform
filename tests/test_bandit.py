"""Bandit tests — Thompson Sampling must converge toward the better arm."""
from __future__ import annotations

import numpy as np

from ml_ab_platform.bandit import ThompsonSampler, compute_regret


def test_thompson_converges_to_best_arm(tmp_config):
    """Over many iterations, TS should pick the best arm most of the time."""
    sampler = ThompsonSampler.from_versions(["A", "B"], seed=42)
    true_means = {"A": 0.3, "B": 0.6}
    rng = np.random.default_rng(0)
    for _ in range(2000):
        chosen = sampler.choose()
        r = int(rng.random() < true_means[chosen])
        sampler.update(chosen, r)
    snap = sampler.snapshot()
    # After convergence, B should have gotten far more pulls than A
    assert snap["B"]["trials"] > 5 * snap["A"]["trials"], snap


def test_thompson_updates_beta():
    s = ThompsonSampler.from_versions(["A", "B"], seed=1)
    s.update("A", 1)
    s.update("A", 0)
    s.update("A", 1)
    arm = s.arms["A"]
    assert arm.alpha == 1 + 2
    assert arm.beta == 1 + 1
    assert arm.trials == 3
    assert arm.successes == 2


def test_regret_monotonic_and_zero_for_optimal():
    rewards = [("A", 1)] * 10  # always pick the best arm
    true_means = {"A": 0.9, "B": 0.1}
    regret = compute_regret(rewards, true_means)
    assert all(r == 0 for r in regret)

    rewards_bad = [("B", 1)] * 10  # always pick the worse arm
    regret_bad = compute_regret(rewards_bad, true_means)
    assert regret_bad[-1] > 7.9  # 10 * (0.9 - 0.1)
    # monotonically non-decreasing
    assert all(regret_bad[i] <= regret_bad[i + 1] for i in range(len(regret_bad) - 1))
