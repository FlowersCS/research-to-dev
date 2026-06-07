"""Tests for trace reading — read_trace loads PipelineTrace from JSON,
extract_hypotheses returns the Hypothesis list."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research_to_dev.cli.orchestrator import PipelineTrace
from research_to_dev.experiment.trace import extract_hypotheses, read_trace


def _build_trace_json(
    hypotheses: list[dict] | None = None,
    query: str = "test query",
    codebase_path: str = "/tmp/test",
) -> str:
    """Build a minimal PipelineTrace JSON string for testing."""
    trace = {
        "query": query,
        "codebase_path": codebase_path,
        "timestamp": "2024-01-01T00:00:00+00:00",
        "steps": {},
        "hypotheses": hypotheses or [],
        "warnings": [],
    }
    return json.dumps(trace, indent=2)


class TestReadTrace:
    """read_trace loads valid PipelineTrace JSON."""

    def test_read_trace_loads_valid_json(self) -> None:
        """read_trace loads a valid trace file and returns PipelineTrace."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json(query="my search"))
            trace_path = f.name

        try:
            trace = read_trace(trace_path)
            assert isinstance(trace, PipelineTrace)
            assert trace.query == "my search"
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_read_trace_missing_file_raises(self) -> None:
        """read_trace raises FileNotFoundError for missing trace file."""
        with pytest.raises(FileNotFoundError, match="Trace not found"):
            read_trace("/nonexistent/trace.json")

    def test_read_trace_empty_hypotheses(self) -> None:
        """read_trace with empty hypotheses list works fine."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json(hypotheses=[]))
            trace_path = f.name

        try:
            trace = read_trace(trace_path)
            assert trace.hypotheses == []
        finally:
            Path(trace_path).unlink(missing_ok=True)


class TestExtractHypotheses:
    """extract_hypotheses returns Hypothesis dataclasses from trace."""

    def test_extract_hypotheses_from_valid_trace(self) -> None:
        """Valid hypotheses are extracted as Hypothesis objects."""
        hypos_raw = [
            {
                "id": "abc123",
                "title": "Improve caching",
                "description": "Add LRU cache to pipeline",
                "approach": "Use functools.lru_cache",
                "supporting_papers": ["Paper A"],
                "target_metric": "accuracy",
                "expected_improvement": "20% faster",
                "code_changes": "src/main.py",
                "composite": 8.5,
                "success_criteria": "p < 0.05",
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json(hypotheses=hypos_raw))
            trace_path = f.name

        try:
            trace = read_trace(trace_path)
            hypotheses = extract_hypotheses(trace)

            assert len(hypotheses) == 1
            h = hypotheses[0]
            assert h.id == "abc123"
            assert h.title == "Improve caching"
            assert h.target_metric == "accuracy"
            assert h.composite == 8.5
            assert h.success_criteria == "p < 0.05"
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_extract_hypotheses_empty(self) -> None:
        """Empty hypotheses list returns empty list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json(hypotheses=[]))
            trace_path = f.name

        try:
            trace = read_trace(trace_path)
            hypotheses = extract_hypotheses(trace)
            assert hypotheses == []
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_extract_hypotheses_multiple(self) -> None:
        """Multiple hypotheses are all extracted."""
        hypos_raw = [
            {"id": "h1", "title": "T1", "description": "D1"},
            {"id": "h2", "title": "T2", "description": "D2"},
            {"id": "h3", "title": "T3", "description": "D3"},
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_json(hypotheses=hypos_raw))
            trace_path = f.name

        try:
            trace = read_trace(trace_path)
            hypotheses = extract_hypotheses(trace)
            assert len(hypotheses) == 3
            assert hypotheses[0].id == "h1"
            assert hypotheses[2].id == "h3"
        finally:
            Path(trace_path).unlink(missing_ok=True)
