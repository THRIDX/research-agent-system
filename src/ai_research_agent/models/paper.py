"""Pydantic models for paper writing stage outputs."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PaperSection(BaseModel):
    name: str
    content: str = ""
    latex: str = ""
    word_count: int = 0


class Figure(BaseModel):
    figure_id: str
    caption: str
    file_path: str
    latex_label: str = ""


class Citation(BaseModel):
    cite_key: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: str = ""
    arxiv_id: Optional[str] = None
    bibtex: str = ""


class PaperDraftStatus(str, Enum):
    OUTLINE = "outline"
    DRAFT = "draft"
    REVISED = "revised"
    FINAL = "final"


class PaperDraft(BaseModel):
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    sections: list[PaperSection] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    draft_status: PaperDraftStatus = PaperDraftStatus.OUTLINE
    latex_source: str = ""
    pdf_path: Optional[str] = None


class WritingOutput(BaseModel):
    idea_title: str
    draft: PaperDraft
    review_comments: list[str] = Field(default_factory=list)
    revision_notes: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
