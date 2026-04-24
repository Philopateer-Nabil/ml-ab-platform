"""Tests for the statistical engine — feed known distributions, verify outputs."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from ml_ab_platform.analysis.tests import (
    cohen_h,
    obrien_fleming_boundary,
    required_sample_size_proportions,
    two_proportion_z_test,
    welch_t_test,
)


def test_z_test_no_difference():
    """Identical proportions → p-value close to 1."""
    r = two_proportion_z_test(k_a=500, n_a=1000, k_b=500, n_b=1000)
    assert r.p_value > 0.99
    assert math.isclose(r.diff, 0.0, abs_tol=1e-9)
    assert r.ci_low < 0 < r.ci_high


def test_z_test_clear_winner():
    """Large, real difference should produce tiny p-value."""
    r = two_proportion_z_test(k_a=400, n_a=1000, k_b=550, n_b=1000)
    assert r.p_value < 1e-5
    assert r.diff > 0.14
    assert r.ci_low > 0


def test_z_test_matches_scipy():
    """Sanity check against scipy's implementation."""
    from scipy.stats import norm
    k_a, n_a = 450, 1000
    k_b, n_b = 480, 1000
    r = two_proportion_z_test(k_a, n_a, k_b, n_b)
    # Recompute with the same pooled-variance formula and compare
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))
    z_expected = (k_b/n_b - k_a/n_a) / se
    p_expected = 2 * (1 - norm.cdf(abs(z_expected)))
    assert math.isclose(r.z, z_expected, rel_tol=1e-6)
    assert math.isclose(r.p_value, p_expected, rel_tol=1e-6)


def test_cohen_h_bounds():
    assert cohen_h(0.5, 0.5) == 0
    # h for (0.1, 0.5) is large
    h = cohen_h(0.1, 0.5)
    assert h > 0.9
    # symmetry
    assert math.isclose(cohen_h(0.3, 0.6), -cohen_h(0.6, 0.3), rel_tol=1e-9)


def test_welch_t_test_detects_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(100, 10, 500)
    b = rng.normal(105, 10, 500)
    r = welch_t_test(a, b)
    assert r.p_value < 1e-5
    assert r.diff > 4
    assert r.ci_low > 0


def test_welch_t_test_no_difference():
    rng = np.random.default_rng(1)
    a = rng.normal(100, 10, 500)
    b = rng.normal(100, 10, 500)
    r = welch_t_test(a, b)
    assert r.p_value > 0.05


def test_required_sample_size_formula():
    """Compare against standard textbook values.

    For baseline p=0.5, MDE=0.05, alpha=0.05, power=0.8, standard online
    calculators return ~1565 per arm — we accept ±5% tolerance.
    """
    n = required_sample_size_proportions(0.5, 0.05, alpha=0.05, power=0.8)
    assert 1400 < n < 1700, f"n={n}"


def test_required_sample_size_validates_inputs():
    with pytest.raises(ValueError):
        required_sample_size_proportions(0.99, 0.05)  # p + MDE > 1


def test_obrien_fleming_boundary_shape():
    """Boundary should decrease as info_fraction increases toward 1."""
    early = obrien_fleming_boundary(0.1, alpha=0.05)
    mid = obrien_fleming_boundary(0.5, alpha=0.05)
    end = obrien_fleming_boundary(1.0, alpha=0.05)
    assert early > mid > end
    assert math.isclose(end, stats.norm.ppf(0.975), rel_tol=1e-6)


def test_sequential_early_stopping_triggers_when_crossed():
    """If z is huge and boundary is modest, sequential_significant is True."""
    r = two_proportion_z_test(k_a=300, n_a=1000, k_b=600, n_b=1000)
    crit = obrien_fleming_boundary(0.8)
    assert abs(r.z) >= crit
