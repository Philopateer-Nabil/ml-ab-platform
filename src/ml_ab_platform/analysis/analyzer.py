"""High-level analyzer that consumes experiment logs and returns a verdict."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ml_ab_platform.analysis.tests import (
    ZTestResult,
    obrien_fleming_boundary,
    power_for_proportions,
    required_sample_size_proportions,
    two_proportion_z_test,
    welch_t_test,
)
from ml_ab_platform.config import get_settings
from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.storage import get_conn

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Observed metrics for a single model on an experiment."""

    version: str
    n_predictions: int = 0
    n_feedback: int = 0
    n_correct: int = 0
    accuracy: float = 0.0
    positive_rate: float = 0.0
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_p99: float = 0.0
    latency_mean: float = 0.0
    throughput_qps: float = 0.0
    error_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    """Structured verdict from :meth:`StatisticalAnalyzer.analyze`."""

    experiment_id: str
    model_a: ModelMetrics
    model_b: ModelMetrics
    accuracy_test: dict[str, Any] | None = None
    latency_test: dict[str, Any] | None = None
    verdict: str = "not_enough_data"
    verdict_reason: str = ""
    required_sample_size: int | None = None
    current_power: float | None = None
    obrien_fleming_critical_z: float | None = None
    sequential_significant: bool = False
    warnings: list[str] = field(default_factory=list)
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model_a": self.model_a.to_dict(),
            "model_b": self.model_b.to_dict(),
            "accuracy_test": self.accuracy_test,
            "latency_test": self.latency_test,
            "verdict": self.verdict,
            "verdict_reason": self.verdict_reason,
            "required_sample_size": self.required_sample_size,
            "current_power": self.current_power,
            "obrien_fleming_critical_z": self.obrien_fleming_critical_z,
            "sequential_significant": self.sequential_significant,
            "warnings": self.warnings,
            "alpha": self.alpha,
        }


class StatisticalAnalyzer:
    """Compute per-model metrics and render a significance verdict.

    Accuracy is the primary metric (two-proportion z-test). Latency is
    evaluated as a secondary metric via Welch's t-test.

    The analyzer also checks power, minimum sample size, and the
    O'Brien-Fleming sequential boundary. Known pitfalls are surfaced as
    free-text ``warnings`` on the result so callers (dashboard/CLI) can
    display them verbatim.
    """

    def __init__(self, alpha: float | None = None, min_samples: int | None = None):
        settings = get_settings()
        self.alpha = alpha if alpha is not None else settings.statistics.alpha
        self.min_samples = (
            min_samples if min_samples is not None else settings.statistics.minimum_sample_size
        )
        self.min_effect = settings.statistics.minimum_effect_size
        self.power = settings.statistics.power

    # ------------------------------------------------------------------ #
    def _load_experiment(self, experiment_id: str) -> dict[str, Any]:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"No experiment {experiment_id!r}")
        return dict(row)

    def _load_metrics(self, experiment_id: str, version: str) -> ModelMetrics:
        with get_conn() as conn:
            # Predictions
            preds = conn.execute(
                """SELECT latency_ms, prediction, timestamp FROM predictions
                   WHERE experiment_id = ? AND model_version = ?""",
                (experiment_id, version),
            ).fetchall()
            # Feedback joined with predictions
            fb = conn.execute(
                """SELECT p.prediction, f.ground_truth
                   FROM feedback f JOIN predictions p USING(request_id)
                   WHERE f.experiment_id = ? AND f.model_version = ?""",
                (experiment_id, version),
            ).fetchall()

        metrics = ModelMetrics(version=version, n_predictions=len(preds), n_feedback=len(fb))
        if preds:
            latencies = np.array([r["latency_ms"] for r in preds], dtype=float)
            metrics.latency_p50 = float(np.percentile(latencies, 50))
            metrics.latency_p95 = float(np.percentile(latencies, 95))
            metrics.latency_p99 = float(np.percentile(latencies, 99))
            metrics.latency_mean = float(np.mean(latencies))
            metrics.positive_rate = float(
                np.mean([r["prediction"] for r in preds])
            )
            # rough throughput over the active window
            ts = [r["timestamp"] for r in preds]
            if len(ts) >= 2:
                import datetime as _dt
                t0 = _dt.datetime.fromisoformat(min(ts))
                t1 = _dt.datetime.fromisoformat(max(ts))
                span = max((t1 - t0).total_seconds(), 1e-6)
                metrics.throughput_qps = len(ts) / span
        if fb:
            correct = sum(1 for r in fb if r["prediction"] == r["ground_truth"])
            metrics.n_correct = correct
            metrics.accuracy = correct / len(fb)
        return metrics

    def _collect_latencies(
        self, experiment_id: str, version: str,
    ) -> np.ndarray:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT latency_ms FROM predictions
                   WHERE experiment_id = ? AND model_version = ?""",
                (experiment_id, version),
            ).fetchall()
        return np.array([r["latency_ms"] for r in rows], dtype=float)

    # ------------------------------------------------------------------ #
    def analyze(self, experiment_id: str) -> AnalysisResult:
        """Full analysis of an experiment: metrics + hypothesis tests + verdict."""
        exp = self._load_experiment(experiment_id)
        a_ver, b_ver = exp["model_a"], exp["model_b"]

        m_a = self._load_metrics(experiment_id, a_ver)
        m_b = self._load_metrics(experiment_id, b_ver)
        result = AnalysisResult(
            experiment_id=experiment_id, model_a=m_a, model_b=m_b, alpha=self.alpha,
        )

        # ---- warnings -------------------------------------------------- #
        if m_a.n_feedback < self.min_samples or m_b.n_feedback < self.min_samples:
            result.warnings.append(
                f"Sample size too small: A={m_a.n_feedback}, B={m_b.n_feedback}, "
                f"min={self.min_samples}. Results are not reliable yet."
            )
        total = max(m_a.n_predictions + m_b.n_predictions, 1)
        share_a = m_a.n_predictions / total
        if share_a < 0.1 or share_a > 0.9:
            result.warnings.append(
                f"Traffic split is heavily imbalanced (A={share_a:.0%}). "
                "Statistical power suffers at extreme splits."
            )

        # ---- accuracy z-test ------------------------------------------ #
        z_result: ZTestResult | None = None
        if m_a.n_feedback > 0 and m_b.n_feedback > 0:
            z_result = two_proportion_z_test(
                k_a=m_a.n_correct, n_a=m_a.n_feedback,
                k_b=m_b.n_correct, n_b=m_b.n_feedback,
                alpha=self.alpha,
            )
            result.accuracy_test = asdict(z_result)

        # ---- latency welch test --------------------------------------- #
        la = self._collect_latencies(experiment_id, a_ver)
        lb = self._collect_latencies(experiment_id, b_ver)
        if len(la) >= 2 and len(lb) >= 2:
            w = welch_t_test(la, lb, alpha=self.alpha)
            result.latency_test = asdict(w)

        # ---- sample size + power + sequential ------------------------- #
        if m_a.n_feedback > 0 and m_b.n_feedback > 0:
            baseline_p = max(0.01, min(0.99, m_a.accuracy))
            try:
                required = required_sample_size_proportions(
                    p_baseline=baseline_p,
                    min_detectable_effect=self.min_effect,
                    alpha=self.alpha,
                    power=self.power,
                )
                result.required_sample_size = required
            except ValueError:
                result.required_sample_size = None
            n_per_arm = min(m_a.n_feedback, m_b.n_feedback)
            if n_per_arm > 0 and z_result is not None:
                result.current_power = power_for_proportions(
                    m_a.accuracy, m_b.accuracy, n_per_arm, alpha=self.alpha,
                )

        # O'Brien-Fleming: treat required_sample_size (per arm) as "information total"
        if result.required_sample_size and z_result is not None:
            info_frac = min(
                1.0,
                min(m_a.n_feedback, m_b.n_feedback) / max(result.required_sample_size, 1),
            )
            info_frac = max(info_frac, 0.01)
            crit = obrien_fleming_boundary(info_frac, alpha=self.alpha)
            result.obrien_fleming_critical_z = crit
            result.sequential_significant = abs(z_result.z) >= crit

        # ---- verdict -------------------------------------------------- #
        result.verdict, result.verdict_reason = self._verdict(result, z_result)
        logger.info("analysis.done", experiment_id=experiment_id, verdict=result.verdict)
        return result

    # ------------------------------------------------------------------ #
    def _verdict(
        self,
        result: AnalysisResult,
        z: ZTestResult | None,
    ) -> tuple[str, str]:
        if z is None:
            return ("not_enough_data",
                    "No feedback labels yet — submit ground truth to begin analysis.")
        if result.model_a.n_feedback < self.min_samples or \
                result.model_b.n_feedback < self.min_samples:
            needed = max(
                self.min_samples - result.model_a.n_feedback,
                self.min_samples - result.model_b.n_feedback,
                0,
            )
            return (
                "not_enough_data",
                f"Below minimum sample size. Need ~{needed} more feedback labels.",
            )
        # Sequential path: prefer the O'Brien-Fleming boundary when we have one.
        if result.obrien_fleming_critical_z is not None:
            if result.sequential_significant:
                winner = "B" if z.diff > 0 else "A"
                return (
                    "significant",
                    f"Model {winner} is significantly better (O'Brien-Fleming boundary "
                    f"z≥{result.obrien_fleming_critical_z:.2f}, observed z={z.z:.2f}, "
                    f"Δ={z.diff*100:.2f}pp).",
                )
            if z.p_value < self.alpha:
                return (
                    "promising",
                    f"Fixed-horizon p={z.p_value:.4f} < α, but sequential boundary "
                    f"(z≥{result.obrien_fleming_critical_z:.2f}) not yet crossed — "
                    "keep collecting data to guard against peeking.",
                )
        else:
            if z.p_value < self.alpha:
                winner = "B" if z.diff > 0 else "A"
                return (
                    "significant",
                    f"Model {winner} is significantly better (p={z.p_value:.4f}, "
                    f"Δ={z.diff*100:.2f}pp, Cohen's h={z.cohen_h:.3f}).",
                )
        if result.required_sample_size:
            remaining = max(
                result.required_sample_size - min(result.model_a.n_feedback,
                                                  result.model_b.n_feedback),
                0,
            )
            if remaining > 0:
                return (
                    "not_yet",
                    f"No significant difference detected yet (p={z.p_value:.4f}). "
                    f"Need ~{remaining} more per-arm samples for target power.",
                )
        return (
            "no_difference",
            f"No significant difference at α={self.alpha} (p={z.p_value:.4f}).",
        )
