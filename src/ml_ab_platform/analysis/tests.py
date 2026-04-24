"""Pure-math statistical tests and sample-size calculators.

All functions are stateless and operate on raw counts / arrays, making them
easy to unit-test against textbook examples.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

# ------------------------------ frequentist --------------------------------- #

@dataclass
class ZTestResult:
    """Two-proportion z-test output with CI and effect size (Cohen's h)."""

    p_a: float
    p_b: float
    diff: float
    z: float
    p_value: float
    ci_low: float
    ci_high: float
    cohen_h: float
    n_a: int
    n_b: int


def two_proportion_z_test(
    k_a: int, n_a: int, k_b: int, n_b: int, alpha: float = 0.05,
) -> ZTestResult:
    """Unpooled two-proportion z-test with (1 - alpha) CI for the difference.

    Uses the pooled variance for the z statistic (standard textbook form) and
    the *unpooled* variance for the confidence interval, which is what most
    references recommend for reporting CIs.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError("Both sample sizes must be positive.")
    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)

    pooled_se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = 0.0 if pooled_se == 0 else (p_b - p_a) / pooled_se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    unpooled_se = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    ci_low = diff - crit * unpooled_se
    ci_high = diff + crit * unpooled_se

    return ZTestResult(
        p_a=p_a, p_b=p_b, diff=diff,
        z=z, p_value=p_value,
        ci_low=ci_low, ci_high=ci_high,
        cohen_h=cohen_h(p_a, p_b),
        n_a=n_a, n_b=n_b,
    )


@dataclass
class WelchResult:
    """Welch's t-test result for continuous metrics (e.g. latency)."""

    mean_a: float
    mean_b: float
    diff: float
    t: float
    df: float
    p_value: float
    ci_low: float
    ci_high: float
    n_a: int
    n_b: int


def welch_t_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> WelchResult:
    """Welch's t-test (unequal variances) with a (1 - alpha) CI for the difference."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        raise ValueError("Need at least two observations per group.")

    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a, var_b = float(a.var(ddof=1)), float(b.var(ddof=1))
    n_a, n_b = len(a), len(b)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    diff = mean_b - mean_a
    t = 0.0 if se == 0 else diff / se
    df_num = (var_a / n_a + var_b / n_b) ** 2
    df_den = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = df_num / df_den if df_den > 0 else float(n_a + n_b - 2)
    p_value = 2 * (1 - stats.t.cdf(abs(t), df=df))
    crit = stats.t.ppf(1 - alpha / 2, df=df)
    return WelchResult(
        mean_a=mean_a, mean_b=mean_b, diff=diff,
        t=t, df=df, p_value=p_value,
        ci_low=diff - crit * se, ci_high=diff + crit * se,
        n_a=n_a, n_b=n_b,
    )


# ------------------------------ effect sizes -------------------------------- #

def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h for the difference between two proportions.

        h = 2 * arcsin(sqrt(p2)) - 2 * arcsin(sqrt(p1))

    Interpretation: |h| ≈ 0.2 small, ≈ 0.5 medium, ≈ 0.8 large.
    """
    return 2 * math.asin(math.sqrt(max(0.0, min(1.0, p2)))) - \
           2 * math.asin(math.sqrt(max(0.0, min(1.0, p1))))


# --------------------------- sample size & power ---------------------------- #

def required_sample_size_proportions(
    p_baseline: float,
    min_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Required *per-arm* sample size to detect an absolute-difference
    ``min_detectable_effect`` on top of ``p_baseline``.

    Formula (standard normal-approximation sample size for two proportions):

        n = (z_{α/2} * sqrt(2 p̄ (1-p̄)) + z_β * sqrt(p1(1-p1) + p2(1-p2)))^2 / δ^2

    Returns ``n`` rounded up.
    """
    p1 = p_baseline
    p2 = p_baseline + min_detectable_effect
    if not 0 < p1 < 1 or not 0 < p2 < 1:
        raise ValueError("Both baseline and baseline+MDE must be in (0, 1).")
    z_alpha = stats.norm.ppf(1 - alpha / (2 if two_sided else 1))
    z_beta = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    num = (
        z_alpha * math.sqrt(2 * pbar * (1 - pbar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    delta = p2 - p1
    return int(math.ceil(num / (delta ** 2)))


def power_for_proportions(
    p_a: float, p_b: float, n_per_arm: int, alpha: float = 0.05,
) -> float:
    """Achieved power given observed / hypothesised proportions and sample size."""
    if n_per_arm <= 0:
        return 0.0
    pbar = (p_a + p_b) / 2
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se_null = math.sqrt(2 * pbar * (1 - pbar) / n_per_arm)
    se_alt = math.sqrt((p_a * (1 - p_a) + p_b * (1 - p_b)) / n_per_arm)
    if se_alt == 0:
        return 1.0 if p_a != p_b else alpha
    z = (abs(p_b - p_a) - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z))


# ------------------------- sequential (O'Brien-Fleming) --------------------- #

def obrien_fleming_boundary(
    info_fraction: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Lan-DeMets approximation to the O'Brien-Fleming alpha-spending boundary.

    At information fraction ``t`` in (0, 1], the critical z-value is::

        z_OF(t) = z_{α/2} / sqrt(t)   (standard O'Brien-Fleming form)

    This is the classic closed-form boundary. Early looks have huge critical
    values (hard to reject), so cumulative Type-I error stays below α even
    across many peeks — the *peeking problem* is controlled.
    """
    if not 0 < info_fraction <= 1:
        raise ValueError("info_fraction must be in (0, 1].")
    alpha_side = alpha / 2 if two_sided else alpha
    z_alpha = stats.norm.ppf(1 - alpha_side)
    return z_alpha / math.sqrt(info_fraction)
