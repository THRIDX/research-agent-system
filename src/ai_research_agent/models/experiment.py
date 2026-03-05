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


class MetricResult(BaseModel):
    metric_name: str
    value: float
    unit: str = ""
    run_id: str = ""
    notes: str = ""


class ExperimentOutput(BaseModel):
    plan_title: str
    runs: list[ExperimentRun] = Field(default_factory=list)
    metric_results: list[MetricResult] = Field(default_factory=list)
    summary: str = ""
    conclusions: list[str] = Field(default_factory=list)
    raw_data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
