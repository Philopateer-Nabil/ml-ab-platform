"""Tests for routing strategies."""
from __future__ import annotations

import random

from ml_ab_platform.routing import (
    FixedSplitRouter,
    RoutingContext,
    StickyRouter,
    build_router,
)


def test_fixed_split_distribution(tmp_config):
    random.seed(123)
    router = FixedSplitRouter("exp1", ["A", "B"], {"split": 0.7})
    counts = {"A": 0, "B": 0}
    for _ in range(5000):
        d = router.choose(RoutingContext(experiment_id="exp1"))
        counts[d.model_version] += 1
    total = counts["A"] + counts["B"]
    share_a = counts["A"] / total
    assert 0.65 < share_a < 0.75, f"share_a={share_a}"


def test_sticky_is_consistent_per_user(tmp_config):
    router = StickyRouter("exp1", ["A", "B"], {"split": 0.5})
    for uid in ["alice", "bob", "carol", "dan", "erin"]:
        first = router.choose(RoutingContext(experiment_id="exp1", user_id=uid)).model_version
        for _ in range(20):
            assert router.choose(
                RoutingContext(experiment_id="exp1", user_id=uid)
            ).model_version == first


def test_sticky_respects_split(tmp_config):
    router = StickyRouter("exp1", ["A", "B"], {"split": 0.3})
    counts = {"A": 0, "B": 0}
    for i in range(2000):
        d = router.choose(RoutingContext(experiment_id="exp1", user_id=f"u-{i}"))
        counts[d.model_version] += 1
    share_a = counts["A"] / (counts["A"] + counts["B"])
    assert 0.25 < share_a < 0.35


def test_factory_unknown_strategy(tmp_config):
    import pytest
    with pytest.raises(ValueError, match="Unknown routing strategy"):
        build_router("magic", "exp1", ["A", "B"], {})


def test_router_requires_two_versions(tmp_config):
    import pytest
    with pytest.raises(ValueError, match="at least two"):
        FixedSplitRouter("exp1", ["A"], {"split": 0.5})
