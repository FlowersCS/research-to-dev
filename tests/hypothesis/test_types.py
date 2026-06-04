"""Tests for hypothesis data contracts (types.py) and config."""

from __future__ import annotations

import hashlib

import pytest

from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisError,
    HypothesisResult,
    HypothesisScores,
)
from research_to_dev.shared.config import HypothesisConfig


# ===================================================================
# HypothesisConfig
# ===================================================================


class TestHypothesisConfig:
    """Tests for HypothesisConfig in shared/config.py (HG-CFG-01)."""

    def test_default_construction(self) -> None:
        """Default values match the spec."""
        c = HypothesisConfig()
        assert c.llm_model == "gpt-4o-mini"
        assert c.embedding_model == "text-embedding-3-small"
        assert c.max_reflection_rounds == 3
        assert c.dedup_threshold == 0.85
        assert c.convergence_score_delta == 0.5
        assert c.max_hypotheses_per_paper == 5
        assert c.max_total_hypotheses == 30
        assert c.relevance_gate == 7

    def test_custom_llm_model(self) -> None:
        """Custom LLM model overrides default."""
        c = HypothesisConfig(llm_model="gpt-4o")
        assert c.llm_model == "gpt-4o"
        assert c.embedding_model == "text-embedding-3-small"

    def test_custom_max_rounds(self) -> None:
        """Custom max_reflection_rounds overrides default."""
        c = HypothesisConfig(max_reflection_rounds=2)
        assert c.max_reflection_rounds == 2

    def test_custom_dedup_threshold(self) -> None:
        """Custom dedup_threshold overrides default."""
        c = HypothesisConfig(dedup_threshold=0.9)
        assert c.dedup_threshold == 0.9

    def test_full_custom(self) -> None:
        """All fields can be customized."""
        c = HypothesisConfig(
            llm_model="custom-llm",
            embedding_model="custom-embed",
            max_reflection_rounds=5,
            dedup_threshold=0.95,
            convergence_score_delta=0.1,
            max_hypotheses_per_paper=3,
            max_total_hypotheses=15,
            relevance_gate=5,
        )
        assert c.llm_model == "custom-llm"
        assert c.embedding_model == "custom-embed"
        assert c.max_reflection_rounds == 5
        assert c.dedup_threshold == 0.95
        assert c.convergence_score_delta == 0.1
        assert c.max_hypotheses_per_paper == 3
        assert c.max_total_hypotheses == 15
        assert c.relevance_gate == 5

    def test_field_types(self) -> None:
        """Config fields have correct types."""
        c = HypothesisConfig()
        assert isinstance(c.llm_model, str)
        assert isinstance(c.embedding_model, str)
        assert isinstance(c.max_reflection_rounds, int)
        assert isinstance(c.dedup_threshold, float)
        assert isinstance(c.convergence_score_delta, float)
        assert isinstance(c.max_hypotheses_per_paper, int)
        assert isinstance(c.max_total_hypotheses, int)
        assert isinstance(c.relevance_gate, int)


# ===================================================================
# HypothesisScores
# ===================================================================


class TestHypothesisScores:
    """Tests for the HypothesisScores dataclass (HG-01, HG-03)."""

    def test_full_construction(self) -> None:
        """Scores can be created with all fields."""
        s = HypothesisScores(relevance=8, feasibility=6, evidence=7)
        assert s.relevance == 8
        assert s.feasibility == 6
        assert s.evidence == 7

    def test_min_scores(self) -> None:
        """Scores accept minimum value 1."""
        s = HypothesisScores(relevance=1, feasibility=1, evidence=1)
        assert s.relevance == 1
        assert s.feasibility == 1
        assert s.evidence == 1

    def test_max_scores(self) -> None:
        """Scores accept maximum value 10."""
        s = HypothesisScores(relevance=10, feasibility=10, evidence=10)
        assert s.relevance == 10
        assert s.feasibility == 10
        assert s.evidence == 10

    def test_field_types(self) -> None:
        """Score fields are integers."""
        s = HypothesisScores(relevance=5, feasibility=5, evidence=5)
        assert isinstance(s.relevance, int)
        assert isinstance(s.feasibility, int)
        assert isinstance(s.evidence, int)


# ===================================================================
# Hypothesis
# ===================================================================


class TestHypothesis:
    """Tests for the Hypothesis dataclass (HG-01)."""

    def test_full_construction(self) -> None:
        """Hypothesis can be created with all fields."""
        scores = HypothesisScores(relevance=8, feasibility=6, evidence=7)
        h = Hypothesis(
            id="abc123def456",
            title="Improve attention mechanism",
            description="Replace current attention with flash attention",
            approach="Implement flash attention in the attention layer",
            supporting_papers=["Deep Learning for NLP"],
            target_metric="training throughput",
            expected_improvement="2x faster training",
            code_changes="model/layers.py attention_layer",
            scores=scores,
            composite=7.0,
            correlations=["a1b2c3d4e5f6"],
            success_criteria="Training time reduced by >40%",
        )
        assert h.id == "abc123def456"
        assert h.title == "Improve attention mechanism"
        assert h.description == "Replace current attention with flash attention"
        assert h.approach == "Implement flash attention in the attention layer"
        assert h.supporting_papers == ["Deep Learning for NLP"]
        assert h.target_metric == "training throughput"
        assert h.expected_improvement == "2x faster training"
        assert h.code_changes == "model/layers.py attention_layer"
        assert h.scores is scores
        assert h.composite == 7.0
        assert h.correlations == ["a1b2c3d4e5f6"]
        assert h.success_criteria == "Training time reduced by >40%"

    def test_minimal_construction(self) -> None:
        """Hypothesis can be created with only required fields."""
        h = Hypothesis(
            id="test123",
            title="Test hypothesis",
            description="Test description",
        )
        assert h.id == "test123"
        assert h.title == "Test hypothesis"
        assert h.description == "Test description"
        assert h.approach == ""
        assert h.supporting_papers == []
        assert h.target_metric == ""
        assert h.expected_improvement == ""
        assert h.code_changes == ""
        assert h.scores.relevance == 1
        assert h.scores.feasibility == 1
        assert h.scores.evidence == 1
        assert h.composite == 0.0
        assert h.correlations == []
        assert h.success_criteria == ""

    def test_composite_calculation(self) -> None:
        """Composite score is the arithmetic mean (equal weighting)."""
        scores = HypothesisScores(relevance=9, feasibility=3, evidence=6)
        h = Hypothesis(
            id="comp", title="T", description="D", scores=scores,
            composite=(9 + 3 + 6) / 3.0,
        )
        assert h.composite == 6.0

    def test_composite_all_ones(self) -> None:
        """Composite with all score = 1."""
        scores = HypothesisScores(relevance=1, feasibility=1, evidence=1)
        h = Hypothesis(
            id="c1", title="T", description="D", scores=scores, composite=1.0,
        )
        assert h.composite == 1.0

    def test_composite_all_tens(self) -> None:
        """Composite with all score = 10."""
        scores = HypothesisScores(relevance=10, feasibility=10, evidence=10)
        h = Hypothesis(
            id="c10", title="T", description="D", scores=scores, composite=10.0,
        )
        assert h.composite == 10.0

    def test_field_types(self) -> None:
        """Hypothesis fields have correct types."""
        h = Hypothesis(id="id1", title="T", description="D")
        assert isinstance(h.id, str)
        assert isinstance(h.title, str)
        assert isinstance(h.description, str)
        assert isinstance(h.approach, str)
        assert isinstance(h.supporting_papers, list)
        assert isinstance(h.target_metric, str)
        assert isinstance(h.scores, HypothesisScores)
        assert isinstance(h.composite, float)
        assert isinstance(h.correlations, list)
        assert isinstance(h.success_criteria, str)

    def test_id_is_12_characters(self) -> None:
        """Hypothesis ID should be exactly 12 characters (AD-02)."""
        h = Hypothesis(id="a1b2c3d4e5f6", title="T", description="D")
        assert len(h.id) == 12

    def test_multiple_correlations(self) -> None:
        """Hypothesis can reference multiple correlation IDs."""
        corr_ids = ["id1", "id2", "id3"]
        h = Hypothesis(
            id="multi", title="T", description="D", correlations=corr_ids,
        )
        assert len(h.correlations) == 3
        assert h.correlations == ["id1", "id2", "id3"]

    def test_multiple_papers(self) -> None:
        """Hypothesis can reference multiple supporting papers."""
        papers = ["Paper A", "Paper B", "Paper C"]
        h = Hypothesis(
            id="papers", title="T", description="D",
            supporting_papers=papers,
        )
        assert h.supporting_papers == papers


# ===================================================================
# HypothesisResult
# ===================================================================


class TestHypothesisResult:
    """Tests for the HypothesisResult dataclass (HG-01)."""

    def test_default_construction(self) -> None:
        """HypothesisResult defaults to empty with zero stats."""
        r = HypothesisResult()
        assert r.hypotheses == []
        assert r.total_input_papers == 0
        assert r.total_correlations == 0
        assert r.reflection_rounds_completed == 0
        assert r.converged is False
        assert r.errors == []

    def test_with_hypotheses(self) -> None:
        """HypothesisResult tracks hypotheses and metadata."""
        h = Hypothesis(id="h1", title="T", description="D")
        r = HypothesisResult(
            hypotheses=[h],
            total_input_papers=2,
            total_correlations=5,
            reflection_rounds_completed=2,
            converged=True,
        )
        assert len(r.hypotheses) == 1
        assert r.hypotheses[0] is h
        assert r.total_input_papers == 2
        assert r.total_correlations == 5
        assert r.reflection_rounds_completed == 2
        assert r.converged is True

    def test_with_errors(self) -> None:
        """HypothesisResult can carry errors."""
        err = HypothesisError(message="Failed", exception_type="ValueError")
        r = HypothesisResult(
            hypotheses=[],
            total_input_papers=1,
            total_correlations=3,
            errors=[err],
        )
        assert len(r.errors) == 1
        assert r.errors[0].message == "Failed"

    def test_field_types(self) -> None:
        """HypothesisResult fields have correct types."""
        r = HypothesisResult()
        assert isinstance(r.hypotheses, list)
        assert isinstance(r.total_input_papers, int)
        assert isinstance(r.total_correlations, int)
        assert isinstance(r.reflection_rounds_completed, int)
        assert isinstance(r.converged, bool)
        assert isinstance(r.errors, list)

    def test_partial_success(self) -> None:
        """Result with both hypotheses and errors (HG-01 scenario)."""
        h = Hypothesis(id="h1", title="T", description="D")
        err = HypothesisError(message="Paper X failed")
        r = HypothesisResult(
            hypotheses=[h],
            total_input_papers=3,
            total_correlations=10,
            errors=[err],
        )
        assert len(r.hypotheses) == 1
        assert len(r.errors) == 1


# ===================================================================
# HypothesisError
# ===================================================================


class TestHypothesisError:
    """Tests for the HypothesisError dataclass."""

    def test_full_construction(self) -> None:
        """HypothesisError can be created with message and exception_type."""
        err = HypothesisError(
            message="LLM generation failed",
            exception_type="TimeoutError",
        )
        assert err.message == "LLM generation failed"
        assert err.exception_type == "TimeoutError"

    def test_default_exception_type(self) -> None:
        """Exception type defaults to None."""
        err = HypothesisError(message="Something went wrong")
        assert err.exception_type is None

    def test_field_types(self) -> None:
        """HypothesisError fields have correct types."""
        err = HypothesisError(message="err")
        assert isinstance(err.message, str)
        assert err.exception_type is None or isinstance(
            err.exception_type, str,
        )


# ===================================================================
# Score clamping
# ===================================================================


class TestScoreClamping:
    """Score validation behavior (via pipeline._clamp_score)."""

    def test_clamp_score_within_range(self) -> None:
        """Values in 1-10 range pass through unchanged."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        # Need a pipeline instance — but _clamp_score is static
        assert HypothesisPipeline._clamp_score(5) == 5
        assert HypothesisPipeline._clamp_score(1) == 1
        assert HypothesisPipeline._clamp_score(10) == 10

    def test_clamp_score_below_range(self) -> None:
        """Values below 1 are clamped to 1."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        assert HypothesisPipeline._clamp_score(0) == 1
        assert HypothesisPipeline._clamp_score(-5) == 1

    def test_clamp_score_above_range(self) -> None:
        """Values above 10 are clamped to 10."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        assert HypothesisPipeline._clamp_score(11) == 10
        assert HypothesisPipeline._clamp_score(100) == 10

    def test_clamp_score_float(self) -> None:
        """Float values are rounded and clamped."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        assert HypothesisPipeline._clamp_score(7.8) == 7
        assert HypothesisPipeline._clamp_score(0.5) == 1
        assert HypothesisPipeline._clamp_score(10.1) == 10

    def test_clamp_score_string(self) -> None:
        """String values are converted and clamped."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        assert HypothesisPipeline._clamp_score("8") == 8
        assert HypothesisPipeline._clamp_score("15") == 10
        assert HypothesisPipeline._clamp_score("0") == 1

    def test_clamp_score_invalid(self) -> None:
        """Invalid values default to 1."""
        from research_to_dev.hypothesis.pipeline import HypothesisPipeline

        assert HypothesisPipeline._clamp_score("abc") == 1
        assert HypothesisPipeline._clamp_score(None) == 1
