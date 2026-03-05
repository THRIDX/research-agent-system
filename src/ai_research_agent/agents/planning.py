"""Planning Agent – creates a detailed experimental plan from a research idea."""

from __future__ import annotations

from typing import Any

from ai_research_agent.agents.base import BaseAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.plan import (
    EvaluationMetric,
    ExperimentPlan,
    ExperimentStep,
    PlanningOutput,
)
from ai_research_agent.models.proposal import ResearchIdea
from ai_research_agent.tools.filesystem import write_json


class PlanningAgent(BaseAgent):
    """Creates a structured experiment plan from a selected research idea."""

    agent_status = ProjectStatus.PLANNING

    def __init__(self, project: Any, idea: ResearchIdea) -> None:
        super().__init__(project)
        self.idea = idea

    def run(self) -> PlanningOutput:
        self.project.update_status(
            status=ProjectStatus.PLANNING,
            current_agent=self.name,
            step="designing experiment plan",
        )

        steps = [
            ExperimentStep(
                step_id="step_01_setup",
                description="Set up environment and install dependencies",
                code_template="# Install dependencies\nimport subprocess\nsubprocess.run(['pip', 'install', '-r', 'requirements.txt'])",
                expected_output="Environment ready",
            ),
            ExperimentStep(
                step_id="step_02_baseline",
                description="Implement and evaluate baseline method",
                code_template="# Baseline evaluation\n# TODO: implement baseline",
                expected_output="Baseline metric values",
                dependencies=["step_01_setup"],
            ),
            ExperimentStep(
                step_id="step_03_proposed",
                description="Implement proposed method",
                code_template="# Proposed method\n# TODO: implement proposed method",
                expected_output="Proposed method implementation",
                dependencies=["step_02_baseline"],
            ),
            ExperimentStep(
                step_id="step_04_evaluate",
                description="Evaluate and compare methods",
                code_template="# Evaluation\n# TODO: compute metrics",
                expected_output="Comparison table",
                dependencies=["step_03_proposed"],
            ),
        ]

        metrics = [
            EvaluationMetric(
                name="accuracy",
                description="Classification or prediction accuracy",
                higher_is_better=True,
            ),
            EvaluationMetric(
                name="f1_score",
                description="F1 score (harmonic mean of precision and recall)",
                higher_is_better=True,
            ),
        ]

        plan = ExperimentPlan(
            title=f"Experiment Plan: {self.idea.title}",
            objective=self.idea.hypothesis,
            methodology=f"Experimental validation of: {self.idea.novelty}",
            datasets=["TBD – identify appropriate public datasets"],
            baseline_methods=["Random baseline", "State-of-the-art method from literature"],
            proposed_method=self.idea.title,
            steps=steps,
            metrics=metrics,
            compute_requirements="Standard GPU instance (e.g., NVIDIA T4)",
            estimated_duration="2–4 hours of compute",
        )

        output = PlanningOutput(
            idea_title=self.idea.title,
            plan=plan,
            risks=["Compute budget exceeded", "Dataset not publicly available"],
            mitigations=["Use smaller model variant", "Use synthetic dataset"],
        )

        out_path = self.project.plan_dir / "planning_output.json"
        self._log("write_json", {"path": str(out_path)})
        write_json(out_path, output.model_dump())
        self._log("write_json", {"path": str(out_path)}, outputs="written")

        return output
