"""Tests for the Planning Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_research_agent.models.plan import PlanningOutput
from ai_research_agent.models.proposal import (
    CandidateIdea,
    FailureMode,
    SuccessCriteria,
)


@pytest.fixture
def mock_project() -> MagicMock:
    """Create a mock project."""
    project = MagicMock()
    project.plan_dir = Path("/tmp/test_plan")
    project.plan_dir.mkdir(parents=True, exist_ok=True)
    project.log_audit = MagicMock()
    project.update_status = MagicMock()
    return project


@pytest.fixture
def sample_idea() -> CandidateIdea:
    """Create a sample candidate idea."""
    return CandidateIdea(
        title="Test Research Idea",
        hypothesis="Adding attention to CNNs improves image classification",
        hypothesis_binary="YES - attention mechanisms will improve CNN performance",
        motivation="Attention mechanisms have shown success in NLP",
        novelty_justification="Novel application to computer vision",
        methodology_sketch="Add self-attention to ResNet architecture",
        failure_modes=[
            FailureMode(
                scenario="Training instability",
                likelihood="MEDIUM",
                mitigation="Use gradient clipping",
            )
        ],
        success_criteria=SuccessCriteria(
            primary_metric="accuracy",
            target_value=0.95,
            statistical_threshold="p < 0.05",
        ),
        feasibility_score=0.8,
        impact_score=0.7,
        novelty_score=0.6,
        combined_score=0.8 * 0.7 * 0.6,
    )


def test_planning_agent_creates_plan(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that planning agent creates a complete plan."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    assert isinstance(output, PlanningOutput)
    assert output.idea_title == sample_idea.title
    assert not output.rejected
    assert len(output.plan.steps) > 0
    assert len(output.plan.metrics) > 0
    assert len(output.ablation_studies) >= 2
    assert len(output.hyperparameters) > 0
    assert output.random_seeds == [42, 123, 456]


def test_planning_agent_writes_plan_json(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that planning agent writes plan.json."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    plan_path = mock_project.plan_dir / "plan.json"
    assert plan_path.exists()

    with open(plan_path) as f:
        loaded = json.load(f)

    assert loaded["idea_title"] == sample_idea.title
    assert loaded["rejected"] is False


def test_planning_agent_writes_plan_md(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that planning agent writes plan.md."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    md_path = mock_project.plan_dir / "plan.md"
    assert md_path.exists()

    content = md_path.read_text()
    assert "Experiment Plan" in content
    assert "Research Objectives" in content
    assert "Ablation Studies" in content


def test_planning_agent_rejects_infeasible_idea(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that planning agent rejects infeasible ideas."""
    sample_idea.feasibility_score = 0.2

    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    assert output.rejected is True
    assert output.rejection_reason is not None
    assert "Feasibility" in output.rejection_reason


def test_ablation_studies_created(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that ablation studies are properly created."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    assert len(output.ablation_studies) >= 2

    # Check ablation structure
    for ablation in output.ablation_studies:
        assert ablation.name
        assert ablation.description
        assert ablation.config_changes


def test_hyperparameters_defined(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that hyperparameters are properly defined."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    # Check required hyperparameters
    assert "learning_rate" in output.hyperparameters
    assert "batch_size" in output.hyperparameters
    assert "epochs" in output.hyperparameters


def test_risk_assessment(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that risk assessment is generated."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    assert len(output.risks) >= 3
    assert len(output.mitigations) >= 3
    assert len(output.risks) == len(output.mitigations)


def test_baseline_methods(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that proper baselines are included."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    assert len(output.plan.baseline_methods) >= 2
    # Should be real methods, not toy baselines
    baselines_str = " ".join(output.plan.baseline_methods)
    assert "Random" not in baselines_str or "Forest" in baselines_str


def test_steps_have_valid_dependencies(
    mock_project: MagicMock, sample_idea: CandidateIdea
) -> None:
    """Test that step dependencies form a valid DAG."""
    from ai_research_agent.agents.planning import PlanningAgent

    agent = PlanningAgent(mock_project, sample_idea)
    output = agent.run()

    step_ids = {step.step_id for step in output.plan.steps}

    for step in output.plan.steps:
        for dep in step.dependencies:
            assert dep in step_ids, f"Dependency {dep} not found for step {step.step_id}"
