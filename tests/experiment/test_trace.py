"""Tests for trace reading — read_trace loads PipelineTrace from JSON,
extract_hypotheses returns the Hypothesis list, and
read_trace_with_correlations extracts hypothesis + correlation data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from research_to_dev.cli.orchestrator import PipelineTrace
from research_to_dev.experiment.trace import (
    extract_hypotheses,
    read_trace,
    read_trace_with_correlations,
)


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


# ===================================================================
# TR-07: read_trace_with_correlations tests
# ===================================================================


def _build_trace_with_correlations_json(
    hypotheses: list[dict] | None = None,
    correlations: dict | None = None,
    include_correlation_key: bool = True,
) -> str:
    """Build a trace.json string with hypothesis and correlation data."""
    data: dict = {"query": "test", "codebase_path": "/tmp/test",
                   "timestamp": "2024-01-01T00:00:00+00:00",
                   "hypotheses": hypotheses or [],
                   "steps": {}, "warnings": []}
    if include_correlation_key:
        data["correlation"] = correlations
    return json.dumps(data, indent=2)


class TestReadTraceWithCorrelations:
    """read_trace_with_correlations extracts hypothesis and correlation data."""

    def test_valid_trace_json(self):
        """GIVEN a valid trace.json with hypotheses and correlations
        WHEN read_trace_with_correlations() is called
        THEN it returns (hypothesis_dicts, correlation_dict)."""
        hypos = [{"id": "h1", "title": "Test"}]
        corrs = {"c1": {"claim": {"text": "Claim"}, "target_name": "Target"}}

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_with_correlations_json(
                hypotheses=hypos, correlations=corrs
            ))
            trace_path = f.name

        try:
            hypo_dicts, corr_dict = read_trace_with_correlations(trace_path)
            assert isinstance(hypo_dicts, list)
            assert len(hypo_dicts) == 1
            assert hypo_dicts[0]["id"] == "h1"
            assert isinstance(corr_dict, dict)
            assert "c1" in corr_dict
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_missing_file(self):
        """GIVEN a non-existent trace path
        WHEN read_trace_with_correlations() is called
        THEN it returns ([], None) with no exception."""
        hypo_dicts, corr_dict = read_trace_with_correlations("/nonexistent/trace.json")
        assert hypo_dicts == []
        assert corr_dict is None

    def test_malformed_json(self):
        """GIVEN a file with invalid JSON content
        WHEN read_trace_with_correlations() is called
        THEN it returns ([], None) with no exception."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("{not valid json")
            trace_path = f.name

        try:
            hypo_dicts, corr_dict = read_trace_with_correlations(trace_path)
            assert hypo_dicts == []
            assert corr_dict is None
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_missing_correlation_key(self):
        """GIVEN a valid trace.json without 'correlation' key
        WHEN read_trace_with_correlations() is called
        THEN it returns hypotheses with correlation=None."""
        hypos = [{"id": "h1", "title": "No correlations"}]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write(_build_trace_with_correlations_json(
                hypotheses=hypos, correlations=None
            ))
            trace_path = f.name

        try:
            hypo_dicts, corr_dict = read_trace_with_correlations(trace_path)
            assert len(hypo_dicts) == 1
            assert corr_dict is None
        finally:
            Path(trace_path).unlink(missing_ok=True)

    def test_missing_hypothesis_key(self):
        """GIVEN a trace.json without hypothesis data in expected key
        WHEN read_trace_with_correlations() is called
        THEN it returns empty hypothesis list."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            # No 'hypotheses' key at all
            data = {"query": "test", "correlation": {"x": {}}}
            f.write(json.dumps(data))
            trace_path = f.name

        try:
            hypo_dicts, corr_dict = read_trace_with_correlations(trace_path)
            assert hypo_dicts == []
        finally:
            Path(trace_path).unlink(missing_ok=True)
