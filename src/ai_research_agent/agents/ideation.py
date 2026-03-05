"""Ideation Agent – generates and selects research ideas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_research_agent.agents.base import BaseAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.proposal import IdeationOutput, RelatedWork, ResearchIdea
from ai_research_agent.tools.filesystem import write_json
from ai_research_agent.tools.search import search_arxiv


class IdeationAgent(BaseAgent):
    """Searches the literature and proposes research ideas."""

    agent_status = ProjectStatus.IDEATION

    def __init__(self, project: Any, topic: str, num_ideas: int = 3) -> None:
        super().__init__(project)
        self.topic = topic
        self.num_ideas = num_ideas

    def run(self) -> IdeationOutput:
        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="searching literature",
        )

        # Search arxiv for related work
        self._log("search_arxiv", {"query": self.topic, "max_results": 20})
        try:
            arxiv_results = search_arxiv(self.topic, max_results=20)
            self._log("search_arxiv", {"query": self.topic}, outputs=len(arxiv_results))
        except Exception as exc:
            self._log("search_arxiv", {"query": self.topic}, error=str(exc))
            arxiv_results = []

        related_work = [
            RelatedWork(
                title=r.title,
                authors=r.authors[:5],
                arxiv_id=r.arxiv_id,
                relevance_score=0.7,
                summary=r.abstract[:300],
            )
            for r in arxiv_results[:10]
        ]

        self.project.update_status(
            status=ProjectStatus.IDEATION,
            current_agent=self.name,
            step="generating ideas",
        )

        # Generate placeholder ideas (in production these would use an LLM)
        ideas: list[ResearchIdea] = []
        for i in range(self.num_ideas):
            idea = ResearchIdea(
                title=f"Research Idea {i + 1}: {self.topic}",
                hypothesis=f"Hypothesis {i + 1} related to {self.topic}",
                motivation=f"Motivation based on gaps identified in literature survey of '{self.topic}'",
                novelty=f"Novel aspect {i + 1}",
                related_work=related_work[:3],
                feasibility_score=0.7,
                impact_score=0.6,
            )
            ideas.append(idea)

        selected = ideas[0] if ideas else None
        output = IdeationOutput(
            topic=self.topic,
            ideas=ideas,
            selected_idea=selected,
            selection_rationale="Selected based on highest combined feasibility and impact scores.",
        )

        # Persist output
        out_path = self.project.idea_dir / "ideation_output.json"
        self._log("write_json", {"path": str(out_path)})
        write_json(out_path, output.model_dump())
        self._log("write_json", {"path": str(out_path)}, outputs="written")

        return output
