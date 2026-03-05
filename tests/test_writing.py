"""Tests for the Writing Agent."""

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
from ai_research_agent.models.paper import WritingOutput
from ai_research_agent.models.plan import (
    EvaluationMetric,
    ExperimentPlan,
    ExperimentStep,
    PlanningOutput,
)
from ai_research_agent.models.proposal import (
    CandidateIdea,
    FailureMode,
    RelatedWork,
    SuccessCriteria,
)


@pytest.fixture
def mock_project() -> MagicMock:
    """Create a mock project."""
    project = MagicMock()
    project.writing_dir = Path("/tmp/test_writing")
    project.writing_dir.mkdir(parents=True, exist_ok=True)
    project.log_audit = MagicMock()
    project.update_status = MagicMock()
    return project


@pytest.fixture
def sample_ideation_output() -> Any:
    """Create a sample ideation output."""
    from ai_research_agent.models.proposal import IdeationOutput

    return IdeationOutput(
        topic="Deep Learning for Image Classification",
        candidate_ideas=[
            CandidateIdea(
                title="Attention in CNNs",
                hypothesis="Adding attention to CNNs improves performance",
                hypothesis_binary="YES - attention improves CNNs",
                motivation="Attention has shown success in NLP",
                novelty_justification="Novel application to vision",
                methodology_sketch="Add self-attention to ResNet",
                failure_modes=[],
                success_criteria=SuccessCriteria(
                    primary_metric="accuracy",
                    target_value=0.95,
                ),
                feasibility_score=0.8,
                impact_score=0.7,
                novelty_score=0.6,
                combined_score=0.8 * 0.7 * 0.6,
                related_work=[
                    RelatedWork(
                        title="Attention is All You Need",
                        authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
                        year=2017,
                        arxiv_id="1706.03762",
                        summary="Transformer architecture with self-attention",
                    )
                ],
            )
        ],
        selected_idea=CandidateIdea(
            title="Attention in CNNs",
            hypothesis="Adding attention to CNNs improves performance",
            hypothesis_binary="YES - attention improves CNNs",
            motivation="Attention has shown success in NLP",
            novelty_justification="Novel application to vision",
            methodology_sketch="Add self-attention to ResNet",
            failure_modes=[],
            success_criteria=SuccessCriteria(
                primary_metric="accuracy",
                target_value=0.95,
            ),
            feasibility_score=0.8,
            impact_score=0.7,
            novelty_score=0.6,
            combined_score=0.8 * 0.7 * 0.6,
            related_work=[
                RelatedWork(
                    title="Attention is All You Need",
                    authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
                    year=2017,
                    arxiv_id="1706.03762",
                    summary="Transformer architecture with self-attention",
                )
            ],
        ),
    )


@pytest.fixture
def sample_planning_output() -> PlanningOutput:
    """Create a sample planning output."""
    steps = [
        ExperimentStep(
            step_id="step_01",
            description="Setup",
            code_template="print('setup')",
            expected_output="Ready",
        )
    ]

    plan = ExperimentPlan(
        title="Test Plan",
        objective="Test hypothesis",
        methodology="Test methodology",
        datasets=["CIFAR-10"],
        baseline_methods=["ResNet-18"],
        proposed_method="Proposed Method",
        steps=steps,
        metrics=[
            EvaluationMetric(
                name="accuracy",
                description="Accuracy",
                higher_is_better=True,
                baseline_value=0.85,
            )
        ],
        compute_requirements="1 GPU",
        estimated_duration="1 hour",
    )

    return PlanningOutput(
        idea_title="Attention in CNNs",
        plan=plan,
        random_seeds=[42, 123, 456],
        hyperparameters={"learning_rate": 0.001, "batch_size": 128},
    )


@pytest.fixture
def sample_experiment_output() -> ExperimentOutput:
    """Create a sample experiment output."""
    return ExperimentOutput(
        plan_title="Test Plan",
        step_results=[
            StepResult(
                step_id="step_01",
                success=True,
                stdout="done",
                stderr="",
                return_code=0,
                duration_seconds=10.0,
            )
        ],
        results=[
            ExperimentResult(
                metric_name="accuracy",
                mean=0.92,
                std=0.01,
                values=[0.91, 0.93, 0.92],
                baseline_mean=0.85,
                p_value=0.001,
                significant=True,
            )
        ],
        ablation_results=[
            ExperimentResult(
                metric_name="without_dropout_accuracy",
                mean=0.88,
                std=0.02,
                values=[0.87, 0.89, 0.88],
                baseline_mean=0.85,
                significant=False,
            )
        ],
        completed=True,
        conclusions=["Method achieves 92% accuracy"],
    )


def test_writing_agent_creates_paper(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that writing agent creates a paper."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    assert isinstance(output, WritingOutput)
    assert output.draft.title


def test_writing_agent_writes_latex(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that writing agent writes LaTeX file."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    latex_path = mock_project.writing_dir / "paper.tex"
    assert latex_path.exists()

    content = latex_path.read_text()
    assert "\\documentclass" in content
    assert "abstract" in content.lower()


def test_writing_agent_writes_markdown(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that writing agent writes markdown file."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    md_path = mock_project.writing_dir / "paper.md"
    assert md_path.exists()

    content = md_path.read_text()
    assert "# " in content  # Markdown header


def test_paper_has_required_sections(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that paper has all required sections."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    section_names = {s.name for s in output.draft.sections}

    required = {"Introduction", "Related Work", "Methodology", "Experiments", "Results", "Conclusion"}
    assert required.issubset(section_names)


def test_paper_has_citations(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that paper has citations."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    assert len(output.draft.citations) > 0


def test_quality_checks_run(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that quality checks are run."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    assert len(output.quality_checks_passed) > 0


def test_abstract_generated(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that abstract is generated."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    assert output.draft.abstract
    assert len(output.draft.abstract) > 50


def test_results_in_paper(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that experiment results appear in paper."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    latex = output.draft.latex_source

    # Should contain accuracy results
    assert "0.92" in latex or "92" in latex


def test_writing_output_json(
    mock_project: MagicMock,
    sample_ideation_output: Any,
    sample_planning_output: PlanningOutput,
    sample_experiment_output: ExperimentOutput,
) -> None:
    """Test that writing output JSON is written."""
    from ai_research_agent.agents.writing import WritingAgent

    agent = WritingAgent(
        mock_project,
        sample_ideation_output,
        sample_planning_output,
        sample_experiment_output,
    )
    output = agent.run()

    out_path = mock_project.writing_dir / "writing_output.json"
    assert out_path.exists()

    with open(out_path) as f:
        loaded = json.load(f)

    assert loaded["idea_title"] == "Attention in CNNs"
