"""Pydantic request/response schemas for the FastAPI gateway."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Prediction request. ``features`` is a raw mapping of column → value
    matching the UCI Adult schema. ``user_id`` is optional but required for
    sticky routing."""

    features: dict[str, Any]
    user_id: str | None = None


class PredictResponse(BaseModel):
    """Prediction response with routing + latency metadata."""

    request_id: str
    experiment_id: str
    model_version: str
    prediction: int
    probability: float
    latency_ms: float
    routing_strategy: str
    routing_reason: str


class FeedbackRequest(BaseModel):
    """Ground-truth label submitted for a past prediction."""

    request_id: str
    ground_truth: int = Field(..., ge=0, le=1)


class FeedbackResponse(BaseModel):
    request_id: str
    recorded: bool
    experiment_id: str | None = None
    model_version: str | None = None


class ExperimentSummary(BaseModel):
    """Experiment record enriched with basic live counters."""

    id: str
    name: str
    description: str
    status: str
    routing_strategy: str
    routing_config: dict[str, Any]
    model_a: str
    model_b: str
    target_metric: str
    minimum_sample_size: int
    created_at: str
    started_at: str | None = None
    stopped_at: str | None = None
    concluded_at: str | None = None
    winner: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)

    model_config = ConfigDict(protected_namespaces=())
