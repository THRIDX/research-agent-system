"""Main orchestrator – runs the full research pipeline."""

from __future__ import annotations

from ai_research_agent.agents.experiment import ExperimentAgent
from ai_research_agent.agents.ideation import IdeationAgent
from ai_research_agent.agents.planning import PlanningAgent
from ai_research_agent.agents.writing import WritingAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.project import Project


class Orchestrator:
    """Runs the four-stage research pipeline end-to-end."""

    def __init__(self, project: Project, use_docker: bool = False) -> None:
        self.project = project
        self.use_docker = use_docker

    def run(self, topic: str) -> None:
        """Execute the full pipeline: ideation → planning → experiment → writing."""
        self.project.update_status(ProjectStatus.PENDING, step="pipeline starting")

        # Stage 1: Ideation
        ideation_agent = IdeationAgent(self.project, topic=topic)
        ideation_output = ideation_agent.execute()

        selected_idea = ideation_output.selected_idea or (
            ideation_output.ideas[0] if ideation_output.ideas else None
        )
        if selected_idea is None:
            raise RuntimeError("Ideation produced no ideas.")

        # Stage 2: Planning
        planning_agent = PlanningAgent(self.project, idea=selected_idea)
        planning_output = planning_agent.execute()

        # Stage 3: Experiment
        experiment_agent = ExperimentAgent(
            self.project,
            planning_output=planning_output,
            use_docker=self.use_docker,
        )
        experiment_output = experiment_agent.execute()

        # Stage 4: Writing
        writing_agent = WritingAgent(
            self.project,
            ideation_output=ideation_output,
            experiment_output=experiment_output,
        )
        writing_agent.execute()

        self.project.update_status(ProjectStatus.COMPLETED, step="pipeline finished")
