"""Tests for Pydantic models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_research_agent.models.common import AuditLogEntry, AtomicWrite, ProjectStatus, StatusRecord
from ai_research_agent.models.experiment import ExperimentOutput, ExperimentRun, RunStatus
from ai_research_agent.models.paper import PaperDraft, PaperSection
from ai_research_agent.models.plan import ExperimentPlan, PlanningOutput
from ai_research_agent.models.proposal import IdeationOutput, RelatedWork, ResearchIdea


class TestProjectStatus:
    def test_all_statuses_defined(self) -> None:
        statuses = {s.value for s in ProjectStatus}
        assert statuses == {"pending", "ideation", "planning", "experiment", "writing", "completed", "failed"}


class TestStatusRecord:
    def test_defaults(self) -> None:
        record = StatusRecord(status=ProjectStatus.PENDING)
        assert record.current_agent is None
        assert record.step is None
        assert isinstance(record.updated_at, datetime)

    def test_round_trip(self) -> None:
        record = StatusRecord(status=ProjectStatus.IDEATION, current_agent="IdeationAgent", step="searching")
        data = record.model_dump()
        restored = StatusRecord.model_validate(data)
        assert restored.status == record.status
        assert restored.current_agent == record.current_agent


class TestAuditLogEntry:
    def test_required_fields(self) -> None:
        entry = AuditLogEntry(agent="TestAgent", tool_name="test_tool", inputs={"key": "val"})
        assert entry.outputs is None
        assert entry.error is None
        assert isinstance(entry.timestamp, datetime)

    def test_with_error(self) -> None:
        entry = AuditLogEntry(agent="A", tool_name="t", inputs={}, error="oops")
        assert entry.error == "oops"


class TestAtomicWrite:
    def test_write_and_read(self, tmp_path: "Path") -> None:
        from pathlib import Path

        path = tmp_path / "test.txt"
        with AtomicWrite(path) as f:
            f.write("hello")
        assert path.read_text() == "hello"

    def test_rollback_on_exception(self, tmp_path: "Path") -> None:
        from pathlib import Path

        path = tmp_path / "test.txt"
        with pytest.raises(ValueError):
            with AtomicWrite(path) as f:
                f.write("partial")
                raise ValueError("abort")
        assert not path.exists()


class TestResearchIdea:
    def test_defaults(self) -> None:
        idea = ResearchIdea(
            title="Test Idea",
            hypothesis="H",
            motivation="M",
            novelty="N",
        )
        assert idea.feasibility_score == 0.5
        assert idea.impact_score == 0.5
        assert idea.related_work == []

    def test_feasibility_bounds(self) -> None:
        with pytest.raises(Exception):
            ResearchIdea(title="x", hypothesis="h", motivation="m", novelty="n", feasibility_score=1.5)


class TestIdeationOutput:
    def test_empty(self) -> None:
        out = IdeationOutput(topic="test topic")
        assert out.ideas == []
        assert out.selected_idea is None


class TestExperimentRun:
    def test_default_status(self) -> None:
        run = ExperimentRun(run_id="r1", step_id="s1")
        assert run.status == RunStatus.PENDING


class TestPaperDraft:
    def test_sections(self) -> None:
        draft = PaperDraft(title="My Paper")
        assert draft.sections == []
        sec = PaperSection(name="Introduction", content="Hello")
        draft.sections.append(sec)
        assert len(draft.sections) == 1
