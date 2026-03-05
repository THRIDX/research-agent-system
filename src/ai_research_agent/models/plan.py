"""Pydantic models for the planning stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class AblationStudy(BaseModel):
    """A configuration for ablation study."""
    name: str
    description: str
    config_changes: dict[str, Any] = Field(default_factory=dict)


class ExperimentStep(BaseModel):
    step_id: str
    description: str
    code_template: str = ""
    expected_output: str = ""
    dependencies: list[str] = Field(default_factory=list)


class EvaluationMetric(BaseModel):
    name: str
    description: str
    higher_is_better: bool = True
    baseline_value: Optional[float] = None


class ExperimentPlan(BaseModel):
    title: str
    objective: str
    methodology: str
    datasets: list[str] = Field(default_factory=list)
    baseline_methods: list[str] = Field(default_factory=list)
    proposed_method: str = ""
    steps: list[ExperimentStep] = Field(default_factory=list)
    metrics: list[EvaluationMetric] = Field(default_factory=list)
    compute_requirements: str = ""
    estimated_duration: str = ""


class PlanningOutput(BaseModel):
    idea_title: str
    plan: ExperimentPlan
    risks: list[str] = Field(default_factory=list)
    mitigations: list[str] = Field(default_factory=list)
    ablation_studies: list[AblationStudy] = Field(default_factory=list)
    rejected: bool = False
    rejection_reason: Optional[str] = None
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    random_seeds: list[int] = Field(default=[42, 123, 456])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
