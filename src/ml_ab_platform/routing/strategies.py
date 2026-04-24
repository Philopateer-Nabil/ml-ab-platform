"""Concrete routing strategies: fixed split, sticky, bandit, canary."""
from __future__ import annotations

import hashlib
import random
from datetime import datetime, timezone
from typing import Any

from ml_ab_platform.bandit import (
    load_bandit_state,
    persist_bandit_state,
    record_bandit_choice,
)
from ml_ab_platform.config import get_settings
from ml_ab_platform.logging_ import get_logger
from ml_ab_platform.routing.base import Router, RoutingContext, RoutingDecision
from ml_ab_platform.storage import get_conn

logger = get_logger(__name__)


class FixedSplitRouter(Router):
    """Random assignment respecting a configured probability split.

    ``config['split']`` is the share of traffic going to ``model_versions[0]``
    (i.e. Model A). Remaining traffic goes to model B.
    """

    name = "fixed"

    def choose(self, ctx: RoutingContext) -> RoutingDecision:
        split = float(self.config.get("split", 0.5))
        pick_a = random.random() < split  # noqa: S311 — not security-sensitive
        version = self.model_versions[0] if pick_a else self.model_versions[1]
        return RoutingDecision(
            model_version=version,
            strategy=self.name,
            reason=f"random draw below {split}" if pick_a else f"random draw above {split}",
            split_at_decision=split,
        )


class StickyRouter(Router):
    """User-sticky hash-based routing.

    A stable hash of ``user_id`` is mapped to [0, 1) and compared against the
    configured split. This guarantees the *same* user is always routed to the
    *same* model — critical for clean A/B analysis: otherwise a single user's
    experience swings between models and their behaviour becomes un-attributable.
    """

    name = "sticky"

    def _bucket(self, user_id: str) -> float:
        # md5 is fine here — we only need a uniform hash, not a cryptographic one.
        digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()  # noqa: S324
        return int(digest[:8], 16) / 0xFFFFFFFF

    def choose(self, ctx: RoutingContext) -> RoutingDecision:
        split = float(self.config.get("split", 0.5))
        if ctx.user_id is None:
            # fall back to fixed random routing
            pick_a = random.random() < split  # noqa: S311
            version = self.model_versions[0] if pick_a else self.model_versions[1]
            return RoutingDecision(
                model_version=version,
                strategy=self.name,
                reason="no user_id provided — fell back to random",
                split_at_decision=split,
            )
        bucket = self._bucket(ctx.user_id)
        version = self.model_versions[0] if bucket < split else self.model_versions[1]
        return RoutingDecision(
            model_version=version,
            strategy=self.name,
            reason=f"hash bucket {bucket:.4f} vs split {split}",
            split_at_decision=split,
            extras={"bucket": bucket},
        )


class BanditRouter(Router):
    """Thompson Sampling bandit router.

    Samples one posterior draw per arm and picks the max; updates the DB-backed
    posterior when feedback arrives.
    """

    name = "bandit"

    def choose(self, ctx: RoutingContext) -> RoutingDecision:
        sampler = load_bandit_state(self.experiment_id, self.model_versions)
        chosen = sampler.choose()
        snap = sampler.snapshot()
        return RoutingDecision(
            model_version=chosen,
            strategy=self.name,
            reason="Thompson sample",
            extras={"posteriors": snap},
        )

    def observe_feedback(self, request_id: str, model_version: str, reward: int) -> None:
        sampler = load_bandit_state(self.experiment_id, self.model_versions)
        sampler.update(model_version, reward)
        persist_bandit_state(self.experiment_id, sampler)
        # approximate "chosen_prob" as posterior mean of the observed arm
        snap = sampler.snapshot()
        mean = snap[model_version]["mean"]
        # cumulative regret needs a baseline — we use the max posterior mean as
        # a proxy for "optimal arm so far", yielding a well-defined monotonic trace
        best_mean = max(s["mean"] for s in snap.values())
        delta = best_mean - mean
        # pull existing cumulative regret and extend it
        with get_conn() as conn:
            row = conn.execute(
                """SELECT cumulative_regret FROM bandit_history
                   WHERE experiment_id = ? ORDER BY id DESC LIMIT 1""",
                (self.experiment_id,),
            ).fetchone()
        prior = float(row["cumulative_regret"]) if row else 0.0
        record_bandit_choice(self.experiment_id, model_version, mean, prior + delta)


class CanaryRouter(Router):
    """Canary router.

    Routes ``initial_split`` of traffic to the *new* model (Model B) while the
    rest stays on the stable model (A). State lives in ``canary_state``:

    - If Model B's empirical performance is within threshold of A *and* we
      have at least ``min_samples`` feedback labels for B, promote to a
      ``promoted_split`` (configurable).
    - If B degrades below A by ``degradation_threshold``, freeze all traffic
      back to A.
    """

    name = "canary"

    def _load_state(self) -> dict[str, Any]:
        settings = get_settings()
        with get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM canary_state WHERE experiment_id = ?", (self.experiment_id,),
            ).fetchone()
        if row is None:
            init = float(self.config.get("initial_split",
                                         settings.routing.canary_initial_split))
            now = datetime.now(timezone.utc).isoformat()
            with get_conn() as conn:
                conn.execute(
                    """INSERT INTO canary_state(experiment_id, current_split, promoted,
                                                degraded, updated_at, notes)
                       VALUES (?, ?, 0, 0, ?, ?)""",
                    (self.experiment_id, init, now, "initial canary"),
                )
            return {"current_split": init, "promoted": 0, "degraded": 0}
        return dict(row)

    def _save_state(self, split: float, promoted: int, degraded: int, notes: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with get_conn() as conn:
            conn.execute(
                """UPDATE canary_state SET current_split = ?, promoted = ?,
                       degraded = ?, updated_at = ?, notes = ?
                   WHERE experiment_id = ?""",
                (split, promoted, degraded, now, notes, self.experiment_id),
            )

    def _evaluate_canary(self, state: dict[str, Any]) -> dict[str, Any]:
        """Decide whether to promote the canary or mark it degraded.

        Compares success rate (positive feedback / total feedback) between the
        baseline (A) and canary (B) models for this experiment.
        """
        settings = get_settings()
        min_samples = int(self.config.get("min_samples",
                                          settings.routing.canary_min_samples))
        promote_split = float(self.config.get("promoted_split", 0.5))
        degradation_threshold = float(self.config.get(
            "degradation_threshold", settings.routing.canary_degradation_threshold))

        if state["promoted"] or state["degraded"]:
            return state

        with get_conn() as conn:
            rows = conn.execute(
                """SELECT f.model_version AS model_version,
                          COUNT(*) AS n,
                          SUM(CASE WHEN f.ground_truth = p.prediction THEN 1 ELSE 0 END) AS k
                   FROM feedback f
                   JOIN predictions p USING(request_id)
                   WHERE f.experiment_id = ?
                   GROUP BY f.model_version""",
                (self.experiment_id,),
            ).fetchall()
        per_model = {r["model_version"]: {"n": r["n"], "k": r["k"] or 0} for r in rows}

        a_ver, b_ver = self.model_versions
        n_b = per_model.get(b_ver, {"n": 0, "k": 0})["n"]
        if n_b < min_samples:
            return state
        p_a = (per_model.get(a_ver, {"n": 0, "k": 0})["k"]
               / max(per_model.get(a_ver, {"n": 1})["n"], 1))
        p_b = per_model[b_ver]["k"] / max(per_model[b_ver]["n"], 1)

        if p_b + degradation_threshold < p_a:
            self._save_state(0.0, 0, 1, f"canary degraded (A={p_a:.3f}, B={p_b:.3f})")
            state.update({"current_split": 0.0, "degraded": 1})
            logger.warning("canary.degraded", experiment_id=self.experiment_id,
                           p_a=p_a, p_b=p_b)
        elif p_b >= p_a - degradation_threshold:
            self._save_state(promote_split, 1, 0,
                             f"canary promoted (A={p_a:.3f}, B={p_b:.3f})")
            state.update({"current_split": promote_split, "promoted": 1})
            logger.info("canary.promoted", experiment_id=self.experiment_id,
                        p_a=p_a, p_b=p_b, new_split=promote_split)
        return state

    def choose(self, ctx: RoutingContext) -> RoutingDecision:
        state = self._load_state()
        state = self._evaluate_canary(state)

        # canary_split = share going to the NEW model (B), i.e. the second arm
        canary_split = float(state["current_split"])
        pick_b = random.random() < canary_split  # noqa: S311
        version = self.model_versions[1] if pick_b else self.model_versions[0]
        reason = (
            "canary degraded — all traffic to baseline"
            if state.get("degraded")
            else ("canary promoted" if state.get("promoted") else "canary initial split")
        )
        return RoutingDecision(
            model_version=version,
            strategy=self.name,
            reason=reason,
            split_at_decision=1.0 - canary_split,  # share going to A
            extras={"canary_split_b": canary_split, **{k: v for k, v in state.items()
                                                       if k in ("promoted", "degraded")}},
        )


# ----------------------------- factory ---------------------------------- #


def build_router(
    strategy: str,
    experiment_id: str,
    model_versions: list[str],
    config: dict[str, Any],
) -> Router:
    """Factory returning the right :class:`Router` for a strategy name."""
    mapping = {
        "fixed": FixedSplitRouter,
        "sticky": StickyRouter,
        "bandit": BanditRouter,
        "canary": CanaryRouter,
    }
    if strategy not in mapping:
        raise ValueError(f"Unknown routing strategy: {strategy!r} (known: {list(mapping)})")
    return mapping[strategy](experiment_id, model_versions, config)
