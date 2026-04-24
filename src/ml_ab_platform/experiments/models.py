"""Pydantic models for experiments."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ExperimentStatus(str, Enum):
    """Lifecycle state of an experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    STOPPED = "stopped"
    CONCLUDED = "concluded"


class ExperimentCreate(BaseModel):
    """Request body to create a new experiment."""

    name: str
    description: str = ""
    model_a: str = "A"
    model_b: str = "B"
    routing_strategy: str = Field(
        "fixed", pattern="^(fixed|sticky|bandit|canary)$"
    )
    routing_config: dict[str, Any] = Field(default_factory=dict)
    target_metric: str = "accuracy"
    minimum_sample_size: int = 500

    model_config = ConfigDict(protected_namespaces=())


class Experiment(BaseModel):
    """Full experiment record as stored."""

    id: str
    name: str
    description: str = ""
    status: ExperimentStatus = ExperimentStatus.DRAFT
    routing_strategy: str
    routing_config: dict[str, Any] = Field(default_factory=dict)
    model_a: str
    model_b: str
    target_metric: str = "accuracy"
    minimum_sample_size: int = 500
    created_at: datetime
    started_at: datetime | None = None
    stopped_at: datetime | None = None
    concluded_at: datetime | None = None
    winner: str | None = None
    conclusion: dict[str, Any] | None = None

    model_config = ConfigDict(protected_namespaces=())
