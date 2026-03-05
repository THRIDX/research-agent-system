"""Tests for the Experiment Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_research_agent.models.experiment import (
    ExperimentOutput,
    ExperimentResult,
    StepResult,
)
from ai_research_agent.models.plan import (
    EvaluationMetric,
    ExperimentPlan,
    ExperimentStep,
    PlanningOutput,
)


@pytest.fixture
def mock_project() -> MagicMock:
    """Create a mock project."""
    project = MagicMock()
    project.exp_dir = Path("/tmp/test_exp")
    project.exp_dir.mkdir(parents=True, exist_ok=True)
    project.log_audit = MagicMock()
    project.update_status = MagicMock()
    return project


@pytest.fixture
def sample_planning_output() -> PlanningOutput:
    """Create a sample planning output."""
    steps = [
        ExperimentStep(
            step_id="step_01_setup",
            description="Setup environment",
            code_template="print('Setup complete')",
            expected_output="Environment ready",
            dependencies=[],
        ),
        ExperimentStep(
            step_id="step_02_baseline",
            description="Run baseline",
            code_template="print('Baseline: 0.85')",
            expected_output="Baseline metrics",
            dependencies=["step_01_setup"],
        ),
        ExperimentStep(
            step_id="step_03_proposed",
            description="Run proposed method",
            code_template="print('Proposed: 0.90')",
            expected_output="Proposed metrics",
            dependencies=["step_02_baseline"],
        ),
    ]

    metrics = [
        EvaluationMetric(
            name="accuracy",
            description="Classification accuracy",
            higher_is_better=True,
            baseline_value=0.85,
        ),
    ]

    plan = ExperimentPlan(
        title="Test Plan",
        objective="Test hypothesis",
        methodology="Test methodology",
        datasets=["Test Dataset"],
        baseline_methods=["Baseline Method"],
        proposed_method="Proposed Method",
        steps=steps,
        metrics=metrics,
        compute_requirements="1 GPU",
        estimated_duration="1 hour",
    )

    return PlanningOutput(
        idea_title="Test Idea",
        plan=plan,
        risks=["Risk 1"],
        mitigations=["Mitigation 1"],
        random_seeds=[42, 123, 456],
    )


def test_experiment_agent_generates_code(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that experiment agent generates code files."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    code_dir = mock_project.exp_dir / "experiment_code"
    assert code_dir.exists()

    # Check that step files are created
    step_files = list(code_dir.glob("*.py"))
    assert len(step_files) > 0


def test_experiment_agent_writes_report(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that experiment agent writes report files."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    # Check report.json
    report_json_path = mock_project.exp_dir / "report.json"
    assert report_json_path.exists()

    with open(report_json_path) as f:
        loaded = json.load(f)

    assert loaded["plan_title"] == sample_planning_output.plan.title


def test_experiment_agent_writes_markdown_report(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that experiment agent writes markdown report."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    report_md_path = mock_project.exp_dir / "report.md"
    assert report_md_path.exists()

    content = report_md_path.read_text()
    assert "Experiment Report" in content
    assert "Main Results" in content


def test_step_results_tracked(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that step results are tracked."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    assert len(output.step_results) > 0
    for result in output.step_results:
        assert isinstance(result, StepResult)
        assert result.step_id


def test_results_aggregated(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that results are aggregated across seeds."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    assert len(output.results) > 0

    for result in output.results:
        assert isinstance(result, ExperimentResult)
        assert result.mean is not None
        assert result.std is not None
        assert len(result.values) > 0


def test_ablation_results_generated(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that ablation study results are generated."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    assert len(output.ablation_results) > 0


def test_completed_flag(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that completed flag is set."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    assert output.completed is True


def test_conclusions_generated(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that conclusions are generated from results."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    assert len(output.conclusions) > 0


def test_report_contains_statistics(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that report contains statistical analysis."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    report_md = mock_project.exp_dir / "report.md"
    content = report_md.read_text()

    assert "mean" in content.lower() or "Mean" in content
    assert "std" in content.lower() or "Std" in content


def test_report_contains_step_summary(
    mock_project: MagicMock, sample_planning_output: PlanningOutput
) -> None:
    """Test that report contains step execution summary."""
    from ai_research_agent.agents.experiment import ExperimentAgent

    agent = ExperimentAgent(mock_project, sample_planning_output)
    output = agent.run()

    report_md = mock_project.exp_dir / "report.md"
    content = report_md.read_text()

    assert "Step" in content or "step" in content
