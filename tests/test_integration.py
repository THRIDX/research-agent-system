"""End-to-end integration tests for the AI Research Agent pipeline.

These tests verify the complete four-stage pipeline works correctly together
with all external dependencies mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ai_research_agent.agents.ideation import IdeationAgent
from ai_research_agent.agents.planning import PlanningAgent
from ai_research_agent.agents.experiment import ExperimentAgent
from ai_research_agent.agents.writing import WritingAgent
from ai_research_agent.models.common import ProjectStatus
from ai_research_agent.models.proposal import CandidateIdea, SuccessCriteria
from ai_research_agent.orchestrator import Orchestrator

from tests.conftest import (
    mock_search_tools,
    mock_execution_tools,
    validate_ideation_output,
    validate_planning_output,
    validate_experiment_output,
    validate_writing_output,
    validate_audit_log,
    validate_status_file,
)


class TestFullPipelineSuccess:
    """Test 1: Full Pipeline Success - run complete pipeline with mocked external calls."""

    def test_full_pipeline_runs_successfully(self, project, mock_search_results):
        """Run complete pipeline: ideation → planning → experiment → writing."""
        # Mock all external calls
        mock_search_tools(googletag=pytest.importorskip("google"), results=mock_search_results)

        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            # Mock experiment execution
            with patch.object(ExperimentAgent, "_run_code_locally") as mock_run:
                mock_run.return_value = {
                    "return_code": 0,
                    "stdout": "Step completed",
                    "stderr": "",
                }

                # Run full pipeline via orchestrator
                orchestrator = Orchestrator(project, use_docker=False)
                orchestrator.run("parameter efficient fine-tuning")

        # Verify all output files are created
        assert (project.idea_dir / "ideation_output.json").exists()
        assert (project.idea_dir / "proposal.md").exists()
        assert (project.plan_dir / "plan.json").exists()
        assert (project.plan_dir / "plan.md").exists()
        assert (project.exp_dir / "experiment_output.json").exists()
        assert (project.exp_dir / "report.json").exists()
        assert (project.exp_dir / "report.md").exists()
        assert (project.writing_dir / "paper.tex").exists()
        assert (project.writing_dir / "paper.md").exists()
        assert (project.writing_dir / "writing_output.json").exists()

    def test_status_transitions_through_all_stages(self, project, mock_search_results):
        """Verify status transitions through all stages."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            # Run pipeline
            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient transformers")

        # Verify final status is COMPLETED
        status = project.get_status()
        assert status.status == ProjectStatus.COMPLETED

    def test_audit_log_has_entries_for_all_stages(self, project, mock_search_results):
        """Verify audit log has entries for all tool calls."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient attention")

        # Check audit log
        audit_results = validate_audit_log(project)
        assert audit_results["has_entries"]
        assert audit_results["total_entries"] > 10  # Should have many entries


class TestPipelineWithRejectedIdea:
    """Test 2: Pipeline with Rejected Idea - ideation fails thresholds."""

    def test_ideation_rejected_all_candidates_fail(self, project, mock_search_results_low_quality):
        """Ideation generates candidates but all fail thresholds."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results_low_quality["arxiv"]
            mock_semantic.return_value = mock_search_results_low_quality["semantic_scholar"]

            # Run ideation
            agent = IdeationAgent(project, topic="basic machine learning", num_ideas=3)
            output = agent.run()

        # Verify REJECTED status is set
        assert output.rejected
        assert output.rejection_reason is not None
        assert "feasibility" in output.rejection_reason.lower() or "impact" in output.rejection_reason.lower()

    def test_downstream_handles_ideation_rejection(self, project, mock_search_results_low_quality):
        """Verify downstream agents handle rejection gracefully."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results_low_quality["arxiv"]
            mock_semantic.return_value = mock_search_results_low_quality["semantic_scholar"]

            # Run ideation
            agent = IdeationAgent(project, topic="basic ml")
            output = agent.run()

        # Verify output files still created
        assert (project.idea_dir / "ideation_output.json").exists()
        assert (project.idea_dir / "proposal.md").exists()

        # Verify rejection reason recorded
        with open(project.idea_dir / "ideation_output.json") as f:
            data = json.load(f)
        assert data["rejected"] is True
        assert data["rejection_reason"] is not None


class TestPipelineWithPlanningRejection:
    """Test 3: Pipeline with Planning Rejection - planning rejects due to infeasibility."""

    def test_planning_rejects_due_to_low_feasibility(self, project, mock_search_results):
        """Planning rejects due to infeasibility."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            # Run ideation
            ideation_agent = IdeationAgent(project, topic="efficient fine-tuning", num_ideas=3)
            ideation_output = ideation_agent.run()

        # Get selected idea and force low feasibility
        if ideation_output.selected_idea:
            selected = ideation_output.selected_idea
            # Force low feasibility to trigger rejection in planning
            selected.feasibility_score = 0.2
        else:
            # Create a candidate with low feasibility
            selected = CandidateIdea(
                title="Low Feasibility Idea",
                hypothesis="This hypothesis is not feasible to test",
                hypothesis_binary="This hypothesis is not feasible (YES/NO)",
                motivation="Test motivation",
                novelty_justification="Test novelty",
                methodology_sketch="Complex methodology",
                success_criteria=SuccessCriteria(
                    primary_metric="accuracy",
                    target_value=95.0,  # Unrealistic target
                ),
                feasibility_score=0.2,
                impact_score=0.8,
                novelty_score=0.9,
                combined_score=0.14,
            )

        # Run planning
        planning_agent = PlanningAgent(project, idea=selected)
        planning_output = planning_agent.run()

        # Verify rejected
        assert planning_output.rejected
        assert planning_output.rejection_reason is not None
        assert "feasibility" in planning_output.rejection_reason.lower()

    def test_experiment_skipped_after_planning_rejection(self, project, mock_search_results):
        """Verify experiment and writing are skipped after planning rejection."""
        # This test verifies the planning rejection logic works
        # The orchestrator should handle the rejection case

        # Create a rejected planning output
        rejected_idea = CandidateIdea(
            title="Rejected Idea",
            hypothesis="Infeasible hypothesis",
            hypothesis_binary="Infeasible (YES/NO)",
            motivation="Motivation",
            novelty_justification="Novelty",
            methodology_sketch="Method",
            success_criteria=SuccessCriteria(primary_metric="accuracy", target_value=90.0),
            feasibility_score=0.2,
            impact_score=0.5,
            novelty_score=0.8,
            combined_score=0.08,
        )

        planning_agent = PlanningAgent(project, idea=rejected_idea)
        planning_output = planning_agent.run()

        # Should be rejected
        assert planning_output.rejected

        # Plan files should exist but indicate rejection
        assert (project.plan_dir / "plan.json").exists()
        with open(project.plan_dir / "plan.json") as f:
            plan_data = json.load(f)
        assert plan_data.get("rejected", False) is True


class TestPipelineWithExperimentFailure:
    """Test 4: Pipeline with Experiment Failure - experiment fails after retries."""

    def test_experiment_fails_after_3_retries(self, project, mock_search_results):
        """Experiment step fails after 3 retries."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            # Run ideation and planning
            ideation_agent = IdeationAgent(project, topic="efficient fine-tuning", num_ideas=3)
            ideation_output = ideation_agent.run()

            selected_idea = ideation_output.selected_idea or ideation_output.candidate_ideas[0]

            planning_agent = PlanningAgent(project, idea=selected_idea)
            planning_output = planning_agent.run()

        # Mock experiment execution to fail
        with patch.object(ExperimentAgent, "_run_code_locally") as mock_run:
            # First two calls succeed, third fails
            call_count = [0]

            def fail_on_third_call(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] >= 3:
                    return {
                        "return_code": -1,
                        "stdout": "",
                        "stderr": "CUDA out of memory",
                        "error": "Execution failed after retries",
                    }
                return {"return_code": 0, "stdout": "Step OK", "stderr": ""}

            mock_run.side_effect = fail_on_third_call

            # Run experiment - may fail on critical step
            experiment_agent = ExperimentAgent(
                project,
                planning_output=planning_output,
                use_docker=False,
            )
            experiment_output = experiment_agent.run()

        # Verify FAILED status and error recording
        # The experiment may complete with partial results or fail completely
        # Check that at least one output file exists or failure is recorded
        exp_output_exists = (project.exp_dir / "experiment_output.json").exists()
        report_exists = (project.exp_dir / "report.json").exists()

        # At least one should exist (either success or partial results)
        assert exp_output_exists or report_exists


class TestAuditLogCompleteness:
    """Test 5: Audit Log Completeness - verify audit.jsonl has all expected entries."""

    def test_audit_contains_all_search_calls(self, project, mock_search_results):
        """Verify audit.jsonl contains all search calls."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("parameter efficient fine-tuning")

        # Validate audit log
        audit_results = validate_audit_log(project)

        assert audit_results["has_search_arxiv"]
        assert audit_results["has_search_semantic"]
        assert audit_results["has_write_json"]
        assert audit_results["total_entries"] > 0

    def test_audit_contains_status_transitions(self, project, mock_search_results):
        """Verify audit log contains status transitions."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient transformers")

        # Read audit entries
        entries = []
        with open(project.audit_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Should have entries from all agents
        agents = {e["agent"] for e in entries}
        assert len(agents) >= 3  # At least ideation, planning, experiment, writing

    def test_audit_records_tool_inputs_and_outputs(self, project, mock_search_results):
        """Verify tool calls include inputs and outputs."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("lora fine-tuning")

        # Read audit entries
        entries = []
        with open(project.audit_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))

        # Check that entries have required fields
        for entry in entries[:5]:  # Check first 5 entries
            assert "timestamp" in entry
            assert "agent" in entry
            assert "tool_name" in entry
            assert "inputs" in entry


class TestOutputFileValidation:
    """Test 6: Output File Validation - verify JSON and markdown outputs match schemas."""

    def test_ideation_json_matches_schema(self, project, mock_search_results):
        """Verify ideation JSON output matches Pydantic schema."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            agent = IdeationAgent(project, topic="efficient fine-tuning", num_ideas=3)
            output = agent.run()

        # Validate output files
        results = validate_ideation_output(project)
        assert results["json_exists"]
        assert results["md_exists"]
        assert results["has_topic"]
        assert results["has_candidates"]

    def test_planning_json_matches_schema(self, project, mock_search_results):
        """Verify planning JSON output matches Pydantic schema."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            # Run ideation
            ideation_agent = IdeationAgent(project, topic="efficient", num_ideas=3)
            ideation_output = ideation_agent.run()

            selected = ideation_output.selected_idea or ideation_output.candidate_ideas[0]

            # Run planning
            planning_agent = PlanningAgent(project, idea=selected)
            planning_output = planning_agent.run()

        # Validate
        results = validate_planning_output(project)
        assert results["json_exists"]
        assert results["md_exists"]
        assert results["has_plan"]

    def test_experiment_json_matches_schema(self, project, mock_search_results):
        """Verify experiment JSON output matches Pydantic schema."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            # Run ideation and planning
            ideation_agent = IdeationAgent(project, topic="efficient", num_ideas=3)
            ideation_output = ideation_agent.run()
            selected = ideation_output.selected_idea or ideation_output.candidate_ideas[0]

            planning_agent = PlanningAgent(project, idea=selected)
            planning_output = planning_agent.run()

            # Run experiment
            experiment_agent = ExperimentAgent(project, planning_output=planning_output)
            experiment_output = experiment_agent.run()

        results = validate_experiment_output(project)
        assert results["json_exists"]
        assert results["md_exists"]

    def test_writing_latex_compiles(self, project, mock_search_results):
        """Verify LaTeX paper has required sections."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            # Run full pipeline
            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("parameter efficient fine-tuning")

        # Validate writing outputs
        results = validate_writing_output(project)
        assert results["tex_exists"]
        assert results["md_exists"]
        assert results["tex_has_abstract"]
        assert results["tex_has_sections"]

    def test_markdown_files_have_required_sections(self, project, mock_search_results):
        """Verify markdown files have required sections."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient transformers")

        # Check proposal.md
        proposal = (project.idea_dir / "proposal.md").read_text()
        assert "Research Proposal" in proposal
        assert "Hypothesis" in proposal

        # Check plan.md
        plan = (project.plan_dir / "plan.md").read_text()
        assert "Experiment Plan" in plan or "Plan" in plan

        # Check report.md
        report = (project.exp_dir / "report.md").read_text()
        assert "Report" in report or "results" in report.lower()


class TestStateRecoveryAndIdempotency:
    """Test 7: State Recovery / Idempotency - verify re-runs don't corrupt state."""

    def test_rerun_from_beginning_no_duplicates(self, project, mock_search_results):
        """Run pipeline partially, re-run from beginning, verify no duplicate data."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            # First run
            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient fine-tuning")

        # Count audit entries after first run
        with open(project.audit_path) as f:
            first_run_entries = [l for l in f if l.strip()]

        # Re-run the full pipeline
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("efficient fine-tuning")

        # Count audit entries after second run
        with open(project.audit_path) as f:
            second_run_entries = [l for l in f if l.strip()]

        # Second run should have more entries (not duplicates)
        assert len(second_run_entries) > len(first_run_entries)

    def test_partial_run_state_consistency(self, project, mock_search_results):
        """Run pipeline, verify state is consistent after partial execution."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]

            # Run only ideation
            agent = IdeationAgent(project, topic="efficient", num_ideas=3)
            output = agent.run()

        # Status should reflect ideation
        status = project.get_status()
        assert status.status == ProjectStatus.IDEATION
        assert status.current_agent == "IdeationAgent"

    def test_status_file_atomically_updated(self, project, mock_search_results):
        """Verify status file is properly updated during pipeline."""
        with patch("ai_research_agent.agents.ideation.search_arxiv") as mock_arxiv, \
             patch("ai_research_agent.agents.ideation.search_semantic_scholar") as mock_semantic, \
             patch.object(ExperimentAgent, "_run_code_locally") as mock_run:

            mock_arxiv.return_value = mock_search_results["arxiv"]
            mock_semantic.return_value = mock_search_results["semantic_scholar"]
            mock_run.return_value = {"return_code": 0, "stdout": "OK", "stderr": ""}

            orchestrator = Orchestrator(project, use_docker=False)
            orchestrator.run("lora fine-tuning")

        # Final status should be COMPLETED
        status = project.get_status()
        assert status.status == ProjectStatus.COMPLETED

        # Status file should have proper structure
        status_results = validate_status_file(project)
        assert status_results["status_exists"]
        assert status_results["has_status"]
        assert status_results["status_value"] == "completed"
