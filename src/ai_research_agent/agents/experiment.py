"""Experiment Agent – executes experiment steps and collects results."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from ai_research_agent.agents.base import BaseAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.experiment import (
    ExperimentOutput,
    ExperimentRun,
    MetricResult,
    RunStatus,
)
from ai_research_agent.models.plan import PlanningOutput
from ai_research_agent.tools.execution import run_local
from ai_research_agent.tools.filesystem import write_json


class ExperimentAgent(BaseAgent):
    """Executes experiment steps defined in the plan."""

    agent_status = ProjectStatus.EXPERIMENT

    def __init__(
        self,
        project: Any,
        planning_output: PlanningOutput,
        use_docker: bool = False,
    ) -> None:
        super().__init__(project)
        self.planning_output = planning_output
        self.use_docker = use_docker

    def run(self) -> ExperimentOutput:
        plan = self.planning_output.plan
        runs: list[ExperimentRun] = []
        metric_results: list[MetricResult] = []

        for step in plan.steps:
            self.project.update_status(
                status=ProjectStatus.EXPERIMENT,
                current_agent=self.name,
                step=f"executing {step.step_id}",
            )

            run_id = str(uuid.uuid4())[:8]
            run = ExperimentRun(
                run_id=run_id,
                step_id=step.step_id,
                status=RunStatus.RUNNING,
                started_at=datetime.now(timezone.utc),
            )

            code = step.code_template or f"print('Step {step.step_id} placeholder')"
            self._log("run_local", {"step_id": step.step_id, "code_length": len(code)})

            try:
                result = run_local(
                    code,
                    working_dir=self.project.exp_dir / step.step_id,
                    timeout=60.0,
                )
                run.stdout = result.stdout
                run.stderr = result.stderr
                run.return_code = result.return_code
                run.duration_seconds = result.duration_seconds
                run.timed_out = result.timed_out
                run.status = RunStatus.SUCCESS if result.return_code == 0 else RunStatus.FAILED
                if result.timed_out:
                    run.status = RunStatus.TIMEOUT
                self._log(
                    "run_local",
                    {"step_id": step.step_id},
                    outputs={"return_code": result.return_code, "timed_out": result.timed_out},
                )
            except Exception as exc:
                run.status = RunStatus.FAILED
                run.stderr = str(exc)
                self._log("run_local", {"step_id": step.step_id}, error=str(exc))

            run.finished_at = datetime.now(timezone.utc)
            runs.append(run)

        # Collect dummy metric results (in production these come from stdout parsing)
        for metric in plan.metrics:
            metric_results.append(
                MetricResult(
                    metric_name=metric.name,
                    value=0.0,
                    unit="",
                    run_id=runs[-1].run_id if runs else "",
                    notes="Placeholder – implement metric parsing",
                )
            )

        output = ExperimentOutput(
            plan_title=plan.title,
            runs=runs,
            metric_results=metric_results,
            summary="Experiment pipeline completed.",
            conclusions=["Results require analysis – placeholder agent."],
        )

        out_path = self.project.exp_dir / "experiment_output.json"
        self._log("write_json", {"path": str(out_path)})
        write_json(out_path, output.model_dump())
        self._log("write_json", {"path": str(out_path)}, outputs="written")

        return output
