"""Pydantic models for ideation/proposal stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class FailureMode(BaseModel):
    """A potential failure scenario for a research idea."""

    scenario: str
    likelihood: str = Field(description="HIGH, MEDIUM, or LOW")
    mitigation: str = ""


class SuccessCriteria(BaseModel):
    """Quantitative success criteria locked before experiment."""

    primary_metric: str
    target_value: float
    statistical_threshold: str = "p < 0.05"
    min_effect_size: Optional[float] = None
    num_random_seeds: int = 3


class ClosestPaper(BaseModel):
    """A paper found during novelty kill search."""

    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    similarity: str = Field(description="How similar to our hypothesis")
    differentiation: str = Field(description="Why our approach is different")


class NoveltyCheckResult(BaseModel):
    """Result of novelty kill search for a hypothesis."""

    hypothesis: str
    search_queries: list[str] = Field(default_factory=list)
    closest_papers: list[ClosestPaper] = Field(default_factory=list)
    novelty_score: float = Field(ge=0.0, le=1.0, description="0 = not novel, 1 = highly novel")
    is_novel: bool = Field(description="Whether passes novelty threshold")
    rejection_reason: Optional[str] = None


class CandidateIdea(BaseModel):
    """A candidate research idea with full evaluation."""

    title: str
    hypothesis: str
    hypothesis_binary: str = Field(description="Binary falsifiable statement (YES/NO testable)")
    motivation: str
    novelty_justification: str
    methodology_sketch: str
    failure_modes: list[FailureMode] = Field(default_factory=list)
    success_criteria: SuccessCriteria
    feasibility_score: float = Field(ge=0.0, le=1.0)
    impact_score: float = Field(ge=0.0, le=1.0)
    novelty_score: float = Field(ge=0.0, le=1.0)
    combined_score: float = Field(ge=0.0, le=1.0, description="feasibility * impact * novelty")
    novelty_check: Optional[NoveltyCheckResult] = None
    passed_filters: bool = True


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
    candidate_ideas: list[CandidateIdea] = Field(default_factory=list)
    selected_idea: Optional[CandidateIdea] = None
    selection_rationale: str = ""
    rejected: bool = False
    rejection_reason: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Legacy compatibility
    @property
    def ideas(self) -> list[ResearchIdea]:
        """Convert candidates to legacy ResearchIdea format."""
        return [
            ResearchIdea(
                title=c.title,
                hypothesis=c.hypothesis,
                motivation=c.motivation,
                novelty=c.novelty_justification,
                feasibility_score=c.feasibility_score,
                impact_score=c.impact_score,
            )
            for c in self.candidate_ideas
        ]
