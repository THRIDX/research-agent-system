"""Shared types used across the entire pipeline."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    PENDING = "pending"
    IDEATION = "ideation"
    PLANNING = "planning"
    EXPERIMENT = "experiment"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


class StatusRecord(BaseModel):
    status: ProjectStatus
    current_agent: Optional[str] = None
    step: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AuditLogEntry(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    tool_name: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: Optional[Any] = None
    error: Optional[str] = None


class AtomicWrite:
    """Context manager for atomic file writes (write to temp, then rename)."""

    def __init__(self, path: Path, mode: str = "w", encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.mode = mode
        self.encoding = encoding
        self._tmp_path: Optional[Path] = None
        self._file: Any = None

    def __enter__(self) -> Any:
        dir_ = self.path.parent
        dir_.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=dir_, prefix=".tmp_", suffix=self.path.suffix)
        self._tmp_path = Path(tmp)
        if "b" in self.mode:
            self._file = os.fdopen(fd, self.mode)
        else:
            self._file = os.fdopen(fd, self.mode, encoding=self.encoding)
        return self._file

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        self._file.close()
        if exc_type is None and self._tmp_path is not None:
            os.replace(self._tmp_path, self.path)
        else:
            if self._tmp_path is not None and self._tmp_path.exists():
                self._tmp_path.unlink()
        return False
