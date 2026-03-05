"""Tests for the IdeationAgent."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_research_agent.agents.ideation import IdeationAgent, MIN_FEASIBILITY, MIN_IMPACT, MIN_NOVELTY_SCORE
from ai_research_agent.models.proposal import (
    CandidateIdea,
    FailureMode,
    IdeationOutput,
    NoveltyCheckResult,
    SuccessCriteria,
)
from ai_research_agent.project import Project


class MockArxivResult:
    """Mock arxiv result for testing."""

    def __init__(self, title: str, abstract: str, authors: list[str], arxiv_id: str = "1234.5678"):
        self.title = title
        self.abstract = abstract
        self.authors = [type("Author", (), {"name": a}) for a in authors]
        self.arxiv_id = arxiv_id
        self.published = "2024-01-01"
        self.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"
        self.categories = ["cs.AI"]


class MockSemanticResult:
    """Mock semantic scholar result for testing."""

    def __init__(self, title: str, abstract: str, authors: list[str], year: int = 2024):
        self.title = title
        self.abstract = abstract
        self.authors = [{"name": a} for a in authors]
        self.year = year
        self.citation_count = 10
        self.url = "http://semantic scholar"
        self.paper_id = "mock-id"


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project(temp_project_dir):
    """Create a test project."""
    proj = Project(temp_project_dir)
    proj.initialize()
    return proj


@pytest.fixture
def mock_arxiv_results():
    """Mock arxiv search results."""
    return [
        MockArxivResult(
            title="LoRA: Low-Rank Adaptation of Large Language Models",
            abstract="We propose a parameter-efficient fine-tuning method...",
            authors=["Edward Hu", "Yelong Shen"],
        ),
        MockArxivResult(
            title="Parameter-Efficient Fine-Tuning for Vision Transformers",
            abstract="Applying LoRA to vision transformers for efficient adaptation...",
            authors=["Jinghao Zhou", "Chen Wei"],
        ),
    ]


@pytest.fixture
def mock_semantic_results():
    """Mock semantic scholar results."""
    return [
        MockSemanticResult(
            title="Contrastive Learning for Image Classification",
            abstract="A novel contrastive learning approach...",
            authors=["Ting Chen", "Simon Kornblith"],
            year=2023,
        ),
    ]


class TestIdeationAgent:
    """Test cases for IdeationAgent."""

    def test_import_ok(self):
        """Test that agent can be imported."""
        from ai_research_agent.agents import IdeationAgent
        assert IdeationAgent is not None

    def test_agent_initialization(self, project):
        """Test agent initializes correctly."""
        agent = IdeationAgent(project, "parameter efficient fine-tuning")
        assert agent.topic == "parameter efficient fine-tuning"
        assert agent.num_ideas == 5
        assert agent.min_feasibility == MIN_FEASIBILITY
        assert agent.min_impact == MIN_IMPACT
        assert agent.min_novelty == MIN_NOVELTY_SCORE

    def test_agent_with_custom_thresholds(self, project):
        """Test agent with custom thresholds."""
        agent = IdeationAgent(
            project,
            "test topic",
            num_ideas=3,
            min_feasibility=0.5,
            min_impact=0.4,
            min_novelty=0.6,
        )
        assert agent.num_ideas == 3
        assert agent.min_feasibility == 0.5

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_run_produces_valid_output(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that run produces valid IdeationOutput."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "parameter efficient fine-tuning", num_ideas=3)
        output = agent.run()

        assert isinstance(output, IdeationOutput)
        assert output.topic == "parameter efficient fine-tuning"
        assert len(output.candidate_ideas) == 3

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_hypothesis_generation(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that agent generates at least 1 hypothesis."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=3)
        output = agent.run()

        for idea in output.candidate_ideas:
            assert idea.hypothesis
            assert idea.hypothesis_binary
            assert "?" not in idea.hypothesis_binary or "YES/NO" in idea.hypothesis_binary

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_failure_modes_count(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that each idea has at least 3 failure modes."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=3)
        output = agent.run()

        for idea in output.candidate_ideas:
            assert len(idea.failure_modes) >= 3
            for fm in idea.failure_modes:
                assert fm.scenario
                assert fm.likelihood in ["HIGH", "MEDIUM", "LOW"]

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents_ideation.search_semantic_scholar")
    def test_selection_logic(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that selection logic works - selects best or rejects."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=3)
        output = agent.run()

        # Either selected or rejected
        assert output.selected_idea is not None or output.rejected

        if output.selected_idea:
            # Check scores meet thresholds
            assert output.selected_idea.feasibility_score >= agent.min_feasibility
            assert output.selected_idea.impact_score >= agent.min_impact
            assert output.selected_idea.novelty_score >= agent.min_novelty

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_output_files_written(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that proposal.md and ideation_output.json are written."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=2)
        output = agent.run()

        json_path = project.idea_dir / "ideation_output.json"
        proposal_path = project.idea_dir / "proposal.md"

        assert json_path.exists()
        assert proposal_path.exists()

        # Verify JSON is valid
        with open(json_path) as f:
            data = json.load(f)
        assert "topic" in data
        assert "candidate_ideas" in data

        # Verify proposal has required sections
        proposal_content = proposal_path.read_text()
        assert "Research Proposal" in proposal_content
        assert "Research Hypothesis" in proposal_content
        assert "Success Criteria" in proposal_content
        assert "Failure Modes" in proposal_content

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_audit_logs_created(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that audit logs are created."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=2)
        output = agent.run()

        audit_path = project.audit_path
        assert audit_path.exists()

        # Read and check audit entries
        entries = []
        with open(audit_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        tool_names = [e["tool_name"] for e in entries]
        assert "search_arxiv" in tool_names
        assert "novelty_kill_search" in tool_names
        assert "write_json" in tool_names
        assert "write_file" in tool_names

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_binary_falsifiable_hypothesis(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that hypotheses are binary falsifiable."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=3)
        output = agent.run()

        for idea in output.candidate_ideas:
            # Check for numeric threshold in hypothesis
            binary = idea.hypothesis_binary
            assert any(char.isdigit() for char in binary), f"No numeric threshold in: {binary}"
            # Check for YES/NO or > or < in hypothesis
            assert any(kw in binary for kw in ["YES", "NO", ">", "<", "%"]), f"Not falsifiable: {binary}"

    @patch("ai_research_agent.agents.ideation.search_arxiv")
    @patch("ai_research_agent.agents.ideation.search_semantic_scholar")
    def test_success_criteria_locked(self, mock_semantic, mock_arxiv, project, mock_arxiv_results, mock_semantic_results):
        """Test that success criteria are quantitative and locked."""
        mock_arxiv.return_value = mock_arxiv_results
        mock_semantic.return_value = mock_semantic_results

        agent = IdeationAgent(project, "test topic", num_ideas=3)
        output = agent.run()

        for idea in output.candidate_ideas:
            sc = idea.success_criteria
            assert sc.primary_metric
            assert sc.target_value > 0
            assert sc.statistical_threshold
            assert sc.num_random_seeds >= 1


class TestIdeationOutput:
    """Test cases for IdeationOutput model."""

    def test_rejected_output(self):
        """Test rejected output creation."""
        output = IdeationOutput(
            topic="test",
            rejected=True,
            rejection_reason="All candidates failed thresholds",
        )
        assert output.rejected
        assert output.selected_idea is None
        assert "failed" in output.rejection_reason.lower()

    def test_selected_output(self):
        """Test selected output creation."""
        candidate = CandidateIdea(
            title="Test Idea",
            hypothesis="Test hypothesis",
            hypothesis_binary="Test hypothesis (YES/NO)",
            motivation="Test motivation",
            novelty_justification="Test novelty",
            methodology_sketch="Test methodology",
            success_criteria=SuccessCriteria(
                primary_metric="accuracy",
                target_value=85.0,
            ),
            feasibility_score=0.8,
            impact_score=0.7,
            novelty_score=0.9,
            combined_score=0.5,
        )
        output = IdeationOutput(
            topic="test",
            candidate_ideas=[candidate],
            selected_idea=candidate,
            selection_rationale="Best score",
        )
        assert not output.rejected
        assert output.selected_idea == candidate

    def test_legacy_ideas_compatibility(self):
        """Test that ideas property works for legacy compatibility."""
        candidate = CandidateIdea(
            title="Test Idea",
            hypothesis="Test hypothesis",
            hypothesis_binary="Test hypothesis (YES/NO)",
            motivation="Test motivation",
            novelty_justification="Test novelty",
            methodology_sketch="Test methodology",
            success_criteria=SuccessCriteria(
                primary_metric="accuracy",
                target_value=85.0,
            ),
            feasibility_score=0.8,
            impact_score=0.7,
            novelty_score=0.9,
            combined_score=0.5,
        )
        output = IdeationOutput(
            topic="test",
            candidate_ideas=[candidate],
        )
        # Legacy ideas should convert candidates to ResearchIdea
        legacy_ideas = output.ideas
        assert len(legacy_ideas) == 1
        assert legacy_ideas[0].title == "Test Idea"


class TestNoveltyCheckResult:
    """Test cases for NoveltyCheckResult."""

    def test_is_novel_when_no_papers(self):
        """Test that novelty is high when no papers found."""
        result = NoveltyCheckResult(
            hypothesis="Test hypothesis",
            search_queries=["test query"],
            closest_papers=[],
            novelty_score=1.0,
            is_novel=True,
        )
        assert result.is_novel
        assert result.novelty_score == 1.0

    def test_is_not_novel_when_papers_found(self):
        """Test that novelty is low when papers found."""
        from ai_research_agent.models.proposal import ClosestPaper

        result = NoveltyCheckResult(
            hypothesis="Test hypothesis",
            search_queries=["test query"],
            closest_papers=[
                ClosestPaper(
                    title="Similar Paper",
                    authors=["Author"],
                    similarity="High",
                    differentiation="None",
                )
            ],
            novelty_score=0.5,
            is_novel=False,
        )
        assert not result.is_novel
        assert result.novelty_score < 1.0
