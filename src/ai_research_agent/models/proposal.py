"""Pydantic models for ideation/proposal stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class RelatedWork(BaseModel):
    title: str
    authors: list[str] = Field(default_factory=list)
    arxiv_id: Optional[str] = None
    year: Optional[int] = None
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.0)
    summary: str = ""


class ResearchIdea(BaseModel):
    title: str
    hypothesis: str
    motivation: str
    novelty: str
    related_work: list[RelatedWork] = Field(default_factory=list)
    feasibility_score: float = Field(ge=0.0, le=1.0, default=0.5)
    impact_score: float = Field(ge=0.0, le=1.0, default=0.5)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IdeationOutput(BaseModel):
    topic: str
    ideas: list[ResearchIdea] = Field(default_factory=list)
    selected_idea: Optional[ResearchIdea] = None
    selection_rationale: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
