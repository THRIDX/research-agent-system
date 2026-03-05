"""Atomic file operations for the research agent system."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml  # type: ignore[import-untyped]


def atomic_write(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """Write content to path atomically using a temp file + rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path | str, content: bytes) -> None:
    """Write bytes to path atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: Path | str) -> Any:
    """Read and parse a JSON file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path | str, data: Any, indent: int = 2) -> None:
    """Write data as JSON atomically."""
    content = json.dumps(data, indent=indent, default=str)
    atomic_write(path, content)


def append_jsonl(path: Path | str, record: Any) -> None:
    """Append a single JSON record to a .jsonl file (not atomic, but append is safe)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_markdown(path: Path | str) -> tuple[Optional[dict[str, Any]], str]:
    """Read a markdown file, returning (frontmatter_dict_or_None, body_text).

    Parses optional YAML frontmatter delimited by ``---`` fences.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")

    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                frontmatter: Optional[dict[str, Any]] = yaml.safe_load(parts[1])
            except yaml.YAMLError:
                frontmatter = None
            body = parts[2].lstrip("\n")
            return frontmatter, body

    return None, text
