"""Pydantic models for the planning stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
