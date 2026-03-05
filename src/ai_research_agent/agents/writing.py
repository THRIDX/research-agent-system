"""Writing Agent – composes a research paper from experiment results."""

from __future__ import annotations

from typing import Any

from ai_research_agent.agents.base import BaseAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.experiment import ExperimentOutput
from ai_research_agent.models.paper import (
    Citation,
    PaperDraft,
    PaperDraftStatus,
    PaperSection,
    WritingOutput,
)
from ai_research_agent.models.proposal import IdeationOutput
from ai_research_agent.tools.filesystem import atomic_write, write_json


_LATEX_TEMPLATE = r"""\documentclass{{article}}
\usepackage{{amsmath,amssymb}}
\usepackage{{hyperref}}
\usepackage{{booktabs}}

\title{{{title}}}
\author{{{authors}}}
\date{{\today}}

\begin{{document}}
\maketitle

\begin{{abstract}}
{abstract}
\end{{abstract}}

{body}

\bibliographystyle{{plain}}
\bibliography{{references}}
\end{{document}}
"""


class WritingAgent(BaseAgent):
    """Composes a full research paper draft from ideation and experiment outputs."""

    agent_status = ProjectStatus.WRITING

    def __init__(
        self,
        project: Any,
        ideation_output: IdeationOutput,
        experiment_output: ExperimentOutput,
    ) -> None:
        super().__init__(project)
        self.ideation_output = ideation_output
        self.experiment_output = experiment_output

    def run(self) -> WritingOutput:
        idea = self.ideation_output.selected_idea
        if idea is None and self.ideation_output.ideas:
            idea = self.ideation_output.ideas[0]

        title = idea.title if idea else self.ideation_output.topic
        abstract = (
            f"{idea.hypothesis} {idea.motivation}" if idea else self.ideation_output.topic
        )

        sections = [
            PaperSection(
                name="Introduction",
                content=f"This paper investigates {title}. {idea.motivation if idea else ''}",
                latex=f"\\section{{Introduction}}\n{title} is an important research direction.",
            ),
            PaperSection(
                name="Related Work",
                content="Survey of relevant prior work.",
                latex="\\section{Related Work}\nWe review the relevant literature.",
            ),
            PaperSection(
                name="Methodology",
                content=f"Proposed approach: {idea.novelty if idea else 'TBD'}",
                latex="\\section{Methodology}\nWe propose the following approach.",
            ),
            PaperSection(
                name="Experiments",
                content=self.experiment_output.summary,
                latex="\\section{Experiments}\n" + self.experiment_output.summary,
            ),
            PaperSection(
                name="Conclusion",
                content="; ".join(self.experiment_output.conclusions),
                latex="\\section{Conclusion}\n" + "; ".join(self.experiment_output.conclusions),
            ),
        ]

        citations: list[Citation] = []
        if idea:
            for rw in idea.related_work[:5]:
                cite_key = "".join(rw.authors[0].split()[-1:]) + str(rw.year or "")
                citations.append(
                    Citation(
                        cite_key=cite_key or rw.arxiv_id or "ref",
                        title=rw.title,
                        authors=rw.authors,
                        year=rw.year,
                        arxiv_id=rw.arxiv_id,
                    )
                )

        body_latex = "\n\n".join(s.latex for s in sections)
        latex_source = _LATEX_TEMPLATE.format(
            title=title,
            authors="AI Research Agent",
            abstract=abstract,
            body=body_latex,
        )

        draft = PaperDraft(
            title=title,
            abstract=abstract,
            authors=["AI Research Agent"],
            sections=sections,
            citations=citations,
            draft_status=PaperDraftStatus.DRAFT,
            latex_source=latex_source,
        )

        output = WritingOutput(
            idea_title=title,
            draft=draft,
            review_comments=["Expand related work section", "Add quantitative results table"],
            revision_notes="Initial draft – requires human review.",
        )

        # Write LaTeX source
        latex_path = self.project.writing_dir / "paper.tex"
        self._log("atomic_write", {"path": str(latex_path)})
        atomic_write(latex_path, latex_source)
        self._log("atomic_write", {"path": str(latex_path)}, outputs="written")

        # Write JSON output
        out_path = self.project.writing_dir / "writing_output.json"
        self._log("write_json", {"path": str(out_path)})
        write_json(out_path, output.model_dump())
        self._log("write_json", {"path": str(out_path)}, outputs="written")

        return output
