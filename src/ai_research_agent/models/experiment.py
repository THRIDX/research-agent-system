"""Pydantic models for experiment execution stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class ExperimentRun(BaseModel):
    run_id: str
    step_id: str
    status: RunStatus = RunStatus.PENDING
    stdout: str = ""
    stderr: str = ""
    return_code: Optional[int] = None
    duration_seconds: Optional[float] = None
    artifacts: list[str] = Field(default_factory=list)  # file paths
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class StepResult(BaseModel):
    """Result of a single experiment step execution."""
    step_id: str
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    duration_seconds: float = 0.0
    output_files: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class MetricResult(BaseModel):
    metric_name: str
    value: float
    unit: str = ""
    run_id: str = ""
    notes: str = ""


class ExperimentResult(BaseModel):
    """Aggregated result for a metric across random seeds."""
    metric_name: str
    mean: float
    std: float
    values: list[float] = Field(default_factory=list)
    baseline_mean: Optional[float] = None
    p_value: Optional[float] = None
    significant: bool = False


class ExperimentOutput(BaseModel):
    plan_title: str
    runs: list[ExperimentRun] = Field(default_factory=list)
    step_results: list[StepResult] = Field(default_factory=list)
    results: list[ExperimentResult] = Field(default_factory=list)
    ablation_results: list[ExperimentResult] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)
    summary: str = ""
    conclusions: list[str] = Field(default_factory=list)
    raw_data: dict[str, Any] = Field(default_factory=dict)
    completed: bool = False
    failed: bool = False
    failure_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
