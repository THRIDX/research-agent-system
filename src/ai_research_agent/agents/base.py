"""Base agent class with state machine and audit logging."""

from __future__ import annotations

import abc
from typing import Any, Optional

from ai_research_agent.models.common import ProjectStatus


class BaseAgent(abc.ABC):
    """Base class for all research pipeline agents.

    Subclasses must implement :meth:`run` which performs the agent's work
    and returns a result dict.  The base class handles status transitions
    and audit logging via the :class:`~ai_research_agent.project.Project`.
    """

    #: The :class:`ProjectStatus` that activates this agent.
    agent_status: ProjectStatus

    def __init__(self, project: Any) -> None:  # project: Project (avoid circular import)
        self.project = project
        self.name: str = self.__class__.__name__

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def execute(self) -> Any:
        """Transition status, run agent logic, and return result."""
        self.project.update_status(
            status=self.agent_status,
            current_agent=self.name,
            step="starting",
        )
        try:
            result = self.run()
            self.project.update_status(
                status=self.agent_status,
                current_agent=self.name,
                step="completed",
            )
            return result
        except Exception as exc:
            self._log(tool_name="execute", inputs={}, error=str(exc))
            self.project.update_status(
                status=ProjectStatus.FAILED,
                current_agent=self.name,
                step=f"failed: {exc}",
            )
            raise

    @abc.abstractmethod
    def run(self) -> Any:
        """Perform the agent's primary work.  Must be implemented by subclasses."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        outputs: Optional[Any] = None,
        error: Optional[str] = None,
    ) -> None:
        """Write an audit log entry via the project."""
        self.project.log_audit(
            agent=self.name,
            tool_name=tool_name,
            inputs=inputs,
            outputs=outputs,
            error=error,
        )
