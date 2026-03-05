"""Pytest fixtures for integration tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.project import Project


# =============================================================================
# Mock Data Classes
# =============================================================================


class MockArxivResult:
    """Mock arxiv result for testing."""

    def __init__(
        self,
        title: str,
        abstract: str,
        authors: list[str],
        arxiv_id: str = "1234.5678",
        published: str = "2024-01-15",
    ):
        self.title = title
        self.abstract = abstract
        self.authors = [type("Author", (), {"name": a})() for a in authors]
        self.arxiv_id = arxiv_id
        self.published = published
        self.pdf_url = f"http://arxiv.org/pdf/{arxiv_id}"
        self.categories = ["cs.AI", "cs.LG"]


class MockSemanticResult:
    """Mock semantic scholar result for testing."""

    def __init__(
        self,
        title: str,
        abstract: str,
        authors: list[str],
        year: int = 2024,
        paper_id: str = "mock-id-123",
    ):
        self.title = title
        self.abstract = abstract
        self.authors = [{"name": a} for a in authors]
        self.year = year
        self.citation_count = 100
        self.url = f"https://www.semanticscholar.org/paper/{paper_id}"
        self.paper_id = paper_id


class MockExecutionResult:
    """Mock experiment execution result."""

    def __init__(
        self,
        return_code: int = 0,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
    ):
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr
        self.error = error


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_project_dir():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def project(temp_project_dir):
    """Create an initialized Project in temp directory."""
    proj = Project(temp_project_dir)
    proj.initialize()
    return proj


@pytest.fixture
def mock_search_results():
    """Mock arxiv and semantic scholar search results with realistic data."""
    return {
        "arxiv": [
            MockArxivResult(
                title="LoRA: Low-Rank Adaptation of Large Language Models",
                abstract="We propose a parameter-efficient fine-tuning method that reduces the number of trainable parameters by downcasting the weight matrices to a lower rank. This approach enables fine-tuning large language models with minimal computational overhead.",
                authors=["Edward Hu", "Yelong Shen", "Pengfei Liu", "Zeyu Wang", "Weizhu Chen"],
                arxiv_id="2106.09685",
                published="2021-06-17",
            ),
            MockArxivResult(
                title="Parameter-Efficient Fine-Tuning for Vision Transformers",
                abstract="We explore parameter-efficient fine-tuning methods for vision transformers. Our approach adapts pre-trained models with minimal parameter changes while maintaining competitive performance on downstream tasks.",
                authors=["Jinghao Zhou", "Chen Wei", "Hao Luo", "Zheng Zhu"],
                arxiv_id="2202.00708",
                published="2022-02-01",
            ),
            MockArxivResult(
                title="Scaling Laws for Fine-Tuning Foundation Models",
                abstract="We study the scaling laws governing the transfer of knowledge from foundation models to downstream tasks. Our empirical findings reveal optimal strategies for compute-efficient fine-tuning.",
                authors=["Jordan Hoffmann", "Sebastian Ruder", "Matei Zaharescu"],
                arxiv_id="2303.18290",
                published="2023-03-31",
            ),
            MockArxivResult(
                title="QLoRA: Efficient Finetuning of Quantized LLMs",
                abstract="We present QLoRA, an efficient fine-tuning approach that combines quantization with low-rank adapters. This method enables fine-tuning of 65B parameter models on a single GPU.",
                authors=["Tim Dettmers", "Artidoro Pagnoni", "Ari Holtzman", "Luke Zettlemoyer"],
                arxiv_id="2305.14314",
                published="2023-05-24",
            ),
            MockArxivResult(
                title="AdapterFusion: Non-invasive Transfer Learning for Intermediate Tasks",
                abstract="We propose AdapterFusion, a novel architecture that combines knowledge from multiple tasks without modifying the original model parameters. This approach enables modular transfer learning.",
                authors=["Jonas Gehring", "Yannic Kilcher", "Lucas Beyer"],
                arxiv_id="2201.05667",
                published="2022-01-17",
            ),
            MockArxivResult(
                title="Prefix-Tuning: Optimizing Continuous Prompts for Generation",
                abstract="We introduce prefix-tuning, a lightweight alternative to fine-tuning where we prepend trainable continuous vectors to the model input. This preserves language model parameters while achieving strong performance.",
                authors=["Pengfei Liu", "Weizhu Chen", "Barlas Oğuz"],
                arxiv_id="2101.00190",
                published="2021-01-01",
            ),
            MockArxivResult(
                title="Prompt Tuning Rapidly Learns Task-Specific Prompt Strategies",
                abstract="We analyze the learning dynamics of prompt tuning and show that it rapidly acquires effective task-specific prompt strategies across various downstream tasks.",
                authors=["Katherine H. Gao", "Micheal Y. Lee", "Andrew W. Brown"],
                arxiv_id="2306.14045",
                published="2023-06-23",
            ),
            MockArxivResult(
                title="The Power of Scale for Parameter-Efficient Prompt Tuning",
                abstract="We demonstrate that prompt tuning becomes more effective as model scale increases. Our analysis shows that large models can learn effective prompts from very few examples.",
                authors=["Quoc V. Le", "Barret Zoph", "Angus G. Taylor"],
                arxiv_id="2112.07626",
                published="2021-12-14",
            ),
        ],
        "semantic_scholar": [
            MockSemanticResult(
                title="Efficient Transformers: A Survey on Efficient Attention Mechanisms",
                abstract="We survey efficient attention mechanisms in transformer models, covering linear attention, sparse attention, and routing-based approaches for scalable language modeling.",
                authors=["Yi Tay", "Mostafa A. H. Abdel-rahman", "Dara Bahri"],
                year=2023,
                paper_id="efficient-transformers-2023",
            ),
            MockSemanticResult(
                title="Contrastive Learning for Fine-Tuning Language Models",
                abstract="We propose a novel contrastive fine-tuning approach that improves downstream task performance by learning better representations from limited labeled data.",
                authors=["Ting Chen", "Simon Kornblith", "Mohammad Norouzi"],
                year=2023,
                paper_id="contrastive-ft-2023",
            ),
            MockSemanticResult(
                title="Meta-Learning for Few-Shot Fine-Tuning",
                abstract="We develop a meta-learning framework for few-shot fine-tuning that enables rapid adaptation to new tasks with minimal training data.",
                authors=["Chelsea Finn", "Pieter Abbeel", "Sergey Levine"],
                year=2022,
                paper_id="meta-learning-ft-2022",
            ),
            MockSemanticResult(
                title="Multi-Modal Fine-Tuning for Vision-Language Models",
                abstract="We present techniques for efficiently fine-tuning vision-language models on downstream tasks while preserving pre-trained knowledge.",
                authors=["Jiahui Xu", "Jiachen Lu", "Zhongzhi Yu"],
                year=2024,
                paper_id="multimodal-ft-2024",
            ),
            MockSemanticResult(
                title="Adapters for Efficient Knowledge Distillation",
                abstract="We introduce adapter-based knowledge distillation, enabling efficient transfer of knowledge from large teacher models to smaller student models.",
                authors=["Geoffrey Hinton", "Oriol Vinyals", "Jeff Dean"],
                year=2023,
                paper_id="adapter-kd-2023",
            ),
        ],
    }


@pytest.fixture
def mock_search_results_low_quality():
    """Mock search results that will produce low-quality ideas (all fail thresholds)."""
    return {
        "arxiv": [
            MockArxivResult(
                title="A Survey on Neural Network Training",
                abstract="This survey covers various training techniques. Limited progress has been made in recent years due to computational constraints and limited data.",
                authors=["John Smith", "Jane Doe"],
                arxiv_id="1901.00001",
                published="2019-01-01",
            ),
        ],
        "semantic_scholar": [
            MockSemanticResult(
                title="Basic Machine Learning Techniques",
                abstract="An introduction to basic machine learning techniques used in various applications.",
                authors=["Alice Brown", "Bob Wilson"],
                year=2018,
                paper_id="basic-ml-2018",
            ),
        ],
    }


@pytest.fixture
def mock_experiment_results():
    """Mock successful experiment execution results."""
    return {
        "accuracy": {
            "mean": 0.92,
            "std": 0.01,
            "values": [0.91, 0.93, 0.92],
            "baseline_mean": 0.85,
            "p_value": 0.003,
            "significant": True,
        },
        "f1_score": {
            "mean": 0.91,
            "std": 0.015,
            "values": [0.90, 0.92, 0.91],
            "baseline_mean": 0.84,
            "p_value": 0.008,
            "significant": True,
        },
        "ablation": {
            "without_dropout": {"mean": 0.87, "std": 0.02},
            "smaller_hidden": {"mean": 0.88, "std": 0.015},
            "no_data_augmentation": {"mean": 0.86, "std": 0.02},
        },
    }


@pytest.fixture
def mock_experiment_failure():
    """Mock experiment failure results."""
    return {
        "step_id": "step_03_proposed",
        "success": False,
        "error": "CUDA out of memory - allocation failed",
        "stderr": "RuntimeError: CUDA out of memory",
        "return_code": -1,
    }


# =============================================================================
# Mock Helper Functions
# =============================================================================


def mock_search_tools(mocker, results: dict):
    """Patch all search functions with mock results.

    Args:
        mocker: pytest-mock fixture
        results: Dictionary with 'arxiv' and 'semantic_scholar' keys
    """
    mocker.patch(
        "ai_research_agent.agents.ideation.search_arxiv",
        return_value=results.get("arxiv", []),
    )
    mocker.patch(
        "ai_research_agent.agents.ideation.search_semantic_scholar",
        return_value=results.get("semantic_scholar", []),
    )
    # Also patch novelty kill search calls
    mocker.patch(
        "ai_research_agent.agents.ideation.search_arxiv",
        return_value=results.get("arxiv", []),
    )


def mock_execution_tools(mocker, success: bool = True, failure_step: str | None = None):
    """Patch execution functions for experiment agent.

    Args:
        mocker: pytest-mock fixture
        success: Whether execution should succeed
        failure_step: If set, cause failure at this step
    """
    if success:
        # Mock successful execution
        mock_result = {
            "return_code": 0,
            "stdout": "Execution completed successfully",
            "stderr": "",
        }
    else:
        # Mock failed execution
        mock_result = {
            "return_code": -1,
            "stdout": "",
            "stderr": "Execution failed",
            "error": "Simulated failure",
        }

    mocker.patch(
        "ai_research_agent.agents.experiment.ExperimentAgent._run_code_locally",
        return_value=mock_result,
    )


def mock_write_tools(mocker):
    """Mock filesystem write tools (they should already work, but ensure they're not real)."""
    # The filesystem tools write to disk, which is fine for tests
    # We don't need to mock them - they operate on the temp directory
    pass


# =============================================================================
# Validation Helpers
# =============================================================================


def validate_ideation_output(project_dir: Path) -> dict[str, bool]:
    """Validate ideation output files exist and have required content."""
    results = {}

    # Check files exist
    json_path = project_dir / "idea" / "ideation_output.json"
    md_path = project_dir / "idea" / "proposal.md"

    results["json_exists"] = json_path.exists()
    results["md_exists"] = md_path.exists()

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        # Check required fields
        results["has_topic"] = "topic" in data
        results["has_candidates"] = "candidate_ideas" in data
        results["has_selected_or_rejected"] = (
            "selected_idea" in data or data.get("rejected", False)
        )

    if md_path.exists():
        content = md_path.read_text()
        results["md_has_hypothesis"] = "Hypothesis" in content
        results["md_has_methodology"] = "Methodology" in content or "methodology" in content

    return results


def validate_planning_output(project_dir: Path) -> dict[str, bool]:
    """Validate planning output files."""
    results = {}

    json_path = project_dir / "plan" / "plan.json"
    md_path = project_dir / "plan" / "plan.md"

    results["json_exists"] = json_path.exists()
    results["md_exists"] = md_path.exists()

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        results["has_plan"] = "plan" in data or "rejected" in data

    if md_path.exists():
        content = md_path.read_text()
        results["md_has_objective"] = "objective" in content.lower() or "Objective" in content
        results["md_has_steps"] = "Steps" in content or "Implementation" in content

    return results


def validate_experiment_output(project_dir: Path) -> dict[str, bool]:
    """Validate experiment output files."""
    results = {}

    json_path = project_dir / "experiments" / "report.json"
    md_path = project_dir / "experiments" / "report.md"

    results["json_exists"] = json_path.exists()
    results["md_exists"] = md_path.exists()

    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)

        results["has_results"] = "results" in data or "step_results" in data

    if md_path.exists():
        content = md_path.read_text()
        results["md_has_results"] = "Results" in content or "results" in content.lower()

    return results


def validate_writing_output(project_dir: Path) -> dict[str, bool]:
    """Validate writing output files."""
    results = {}

    tex_path = project_dir / "writing" / "paper.tex"
    md_path = project_dir / "writing" / "paper.md"
    json_path = project_dir / "writing" / "writing_output.json"

    results["tex_exists"] = tex_path.exists()
    results["md_exists"] = md_path.exists()
    results["json_exists"] = json_path.exists()

    if tex_path.exists():
        content = tex_path.read_text()
        results["tex_has_abstract"] = "abstract" in content.lower()
        results["tex_has_sections"] = all(
            s in content for s in ["Introduction", "Methodology", "Experiments"]
        )

    if md_path.exists():
        content = md_path.read_text()
        results["md_has_title"] = "# " in content

    return results


def validate_audit_log(project_dir: Path) -> dict[str, bool]:
    """Validate audit log has all expected entries."""
    results = {}

    audit_path = project_dir / "audit.jsonl"
    results["audit_exists"] = audit_path.exists()

    if audit_path.exists():
        entries = []
        with open(audit_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        results["has_entries"] = len(entries) > 0

        tool_names = {e["tool_name"] for e in entries}
        results["has_search_arxiv"] = "search_arxiv" in tool_names
        results["has_search_semantic"] = "search_semantic_scholar" in tool_names
        results["has_write_json"] = "write_json" in tool_names or any(
            "write" in t for t in tool_names
        )
        results["total_entries"] = len(entries)
        results["tool_names"] = tool_names

    return results


def validate_status_file(project_dir: Path) -> dict[str, Any]:
    """Validate status.json has proper structure."""
    results = {}

    status_path = project_dir / "status.json"

    results["status_exists"] = status_path.exists()

    if status_path.exists():
        with open(status_path) as f:
            data = json.load(f)

        results["has_status"] = "status" in data
        results["status_value"] = data.get("status")

    return results
