"""Abstract base class for all routing strategies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingContext:
    """Per-request context available to a router."""

    experiment_id: str
    user_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """The outcome of a routing decision — which model, plus diagnostics."""

    model_version: str
    strategy: str
    reason: str
    split_at_decision: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class Router(ABC):
    """Abstract router. Subclasses override :meth:`choose` and can hook
    :meth:`observe_feedback` to react to ground-truth labels.

    Subclasses are stateless w.r.t. the database where possible — any state
    that must persist across requests is loaded at decision time and written
    back, so multiple gateway workers converge on the same view.
    """

    name: str = "base"

    def __init__(self, experiment_id: str, model_versions: list[str], config: dict[str, Any]):
        if len(model_versions) < 2:
            raise ValueError("Router requires at least two model versions.")
        self.experiment_id = experiment_id
        self.model_versions = model_versions
        self.config = config

    @abstractmethod
    def choose(self, ctx: RoutingContext) -> RoutingDecision:
        """Select a model version for this request."""

    def observe_feedback(
        self,
        request_id: str,
        model_version: str,
        reward: int,
    ) -> None:  # pragma: no cover - default no-op
        """React to an arriving ground-truth label (bandit/canary override)."""
        return
