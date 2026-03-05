"""Tests for filesystem and execution tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_research_agent.tools.filesystem import (
    append_jsonl,
    atomic_write,
    atomic_write_bytes,
    read_json,
    read_markdown,
    write_json,
)
from ai_research_agent.tools.execution import run_local


class TestAtomicWrite:
    def test_basic(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        atomic_write(p, "hello world")
        assert p.read_text() == "hello world"

    def test_overwrites(self, tmp_path: Path) -> None:
        p = tmp_path / "out.txt"
        atomic_write(p, "first")
        atomic_write(p, "second")
        assert p.read_text() == "second"

    def test_creates_parents(self, tmp_path: Path) -> None:
        p = tmp_path / "a" / "b" / "c.txt"
        atomic_write(p, "deep")
        assert p.read_text() == "deep"

    def test_bytes(self, tmp_path: Path) -> None:
        p = tmp_path / "data.bin"
        atomic_write_bytes(p, b"\x00\x01\x02")
        assert p.read_bytes() == b"\x00\x01\x02"


class TestJsonHelpers:
    def test_write_read_roundtrip(self, tmp_path: Path) -> None:
        p = tmp_path / "data.json"
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        write_json(p, data)
        loaded = read_json(p)
        assert loaded == data

    def test_nested(self, tmp_path: Path) -> None:
        p = tmp_path / "nested.json"
        data = {"a": {"b": {"c": True}}}
        write_json(p, data)
        assert read_json(p) == data


class TestAppendJsonl:
    def test_appends_multiple(self, tmp_path: Path) -> None:
        p = tmp_path / "log.jsonl"
        append_jsonl(p, {"msg": "first"})
        append_jsonl(p, {"msg": "second"})
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"msg": "first"}
        assert json.loads(lines[1]) == {"msg": "second"}


class TestReadMarkdown:
    def test_no_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("# Hello\n\nWorld")
        fm, body = read_markdown(p)
        assert fm is None
        assert "Hello" in body

    def test_with_frontmatter(self, tmp_path: Path) -> None:
        p = tmp_path / "doc.md"
        p.write_text("---\ntitle: Test\nauthor: AI\n---\n\n# Body")
        fm, body = read_markdown(p)
        assert fm is not None
        assert fm["title"] == "Test"
        assert "Body" in body


class TestRunLocal:
    def test_success(self, tmp_path: Path) -> None:
        result = run_local("print('hello')", working_dir=tmp_path)
        assert result.return_code == 0
        assert "hello" in result.stdout

    def test_failure(self, tmp_path: Path) -> None:
        result = run_local("raise ValueError('boom')", working_dir=tmp_path)
        assert result.return_code != 0
        assert "ValueError" in result.stderr

    def test_timeout(self, tmp_path: Path) -> None:
        result = run_local("import time; time.sleep(10)", working_dir=tmp_path, timeout=0.5)
        assert result.timed_out is True
