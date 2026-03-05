"""Project directory management for a single research project."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_research_agent.models.common import AuditLogEntry, ProjectStatus, StatusRecord
from ai_research_agent.tools.filesystem import append_jsonl, read_json, write_json


class Project:
    """Manages a research project directory with status tracking and audit logging."""

    def __init__(self, project_dir: Path) -> None:
        self.project_dir = Path(project_dir)

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------

    @property
    def status_path(self) -> Path:
        return self.project_dir / "status.json"

    @property
    def audit_path(self) -> Path:
        return self.project_dir / "audit.jsonl"

    @property
    def idea_dir(self) -> Path:
        return self.project_dir / "idea"

    @property
    def plan_dir(self) -> Path:
        return self.project_dir / "plan"

    @property
    def exp_dir(self) -> Path:
        return self.project_dir / "experiments"

    @property
    def writing_dir(self) -> Path:
        return self.project_dir / "writing"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Create directory structure, status.json, and audit.jsonl."""
        for d in (
            self.project_dir,
            self.idea_dir,
            self.plan_dir,
            self.exp_dir,
            self.writing_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        initial_status = StatusRecord(
            status=ProjectStatus.PENDING,
            current_agent=None,
            step="initialized",
            updated_at=datetime.now(timezone.utc),
        )
        write_json(self.status_path, initial_status.model_dump())

        # Ensure audit file exists
        if not self.audit_path.exists():
            self.audit_path.touch()

    # ------------------------------------------------------------------
    # Status management
    # ------------------------------------------------------------------

    def get_status(self) -> StatusRecord:
        """Read and return the current project status."""
        data = read_json(self.status_path)
        return StatusRecord.model_validate(data)

    def update_status(
        self,
        status: ProjectStatus,
        current_agent: Optional[str] = None,
        step: Optional[str] = None,
    ) -> None:
        """Atomically update the project status file."""
        record = StatusRecord(
            status=status,
            current_agent=current_agent,
            step=step,
            updated_at=datetime.now(timezone.utc),
        )
        write_json(self.status_path, record.model_dump())

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def log_audit(
        self,
        agent: str,
        tool_name: str,
        inputs: dict[str, Any],
        outputs: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Append an audit log entry to audit.jsonl."""
        entry = AuditLogEntry(
            agent=agent,
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            error=error,
        )
        append_jsonl(self.audit_path, entry.model_dump())
