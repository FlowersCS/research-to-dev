"""Unit tests for ranking data types and configuration."""

from __future__ import annotations

from datetime import datetime

import pytest

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.types import RankedPaper, RankingError, RankingResult
from research_to_dev.shared.config import RankingConfig


# -------------------------------------------------------------------
# RankingConfig (SC-01)
# -------------------------------------------------------------------


class TestRankingConfig:
    """Verify RankingConfig dataclass creation and defaults (SC-01)."""

    def test_defaults_use_current_year_minus_five(self) -> None:
        """SC-01: min_year defaults to current_year - 5."""
        cfg = RankingConfig()
        expected = datetime.now().year - 5
        assert cfg.min_year == expected

    def test_default_values_are_correct(self) -> None:
        """SC-01: semantic_top_k=20, llm_min_papers=5, llm_max_papers=10."""
        cfg = RankingConfig()
        assert cfg.semantic_top_k == 20
        assert cfg.llm_min_papers == 5
        assert cfg.llm_max_papers == 10

    def test_model_defaults_are_set(self) -> None:
        """Design: embedding_model and llm_model have reasonable defaults."""
        cfg = RankingConfig()
        assert cfg.embedding_model == "text-embedding-3-small"
        assert cfg.llm_model == "gpt-4o-mini"

    def test_custom_values_preserved_exactly(self) -> None:
        """SC-01: Custom values preserve exactly."""
        cfg = RankingConfig(
            min_year=2015,
            semantic_top_k=10,
            llm_min_papers=3,
            llm_max_papers=7,
            embedding_model="custom-embed",
            llm_model="custom-llm",
        )
        assert cfg.min_year == 2015
        assert cfg.semantic_top_k == 10
        assert cfg.llm_min_papers == 3
        assert cfg.llm_max_papers == 7
        assert cfg.embedding_model == "custom-embed"
        assert cfg.llm_model == "custom-llm"


# -------------------------------------------------------------------
# RankedPaper
# -------------------------------------------------------------------


class TestRankedPaper:
    """Verify RankedPaper dataclass creation."""

    def test_full_fields(self) -> None:
        paper = AcademicContent(source="arxiv", title="Test Paper")
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.85,
            semantic_score=0.72,
            llm_reasoning="Relevant because it addresses the query directly.",
            passed_metadata_filter=True,
        )
        assert ranked.paper is paper
        assert ranked.relevance_score == 0.85
        assert ranked.semantic_score == 0.72
        assert ranked.llm_reasoning == "Relevant because it addresses the query directly."
        assert ranked.passed_metadata_filter is True

    def test_zero_scores_represent_failures(self) -> None:
        """RF-05: Zero scores represent stage failures."""
        paper = AcademicContent(source="arxiv", title="Failed Paper")
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.0,
            semantic_score=0.0,
            llm_reasoning="",
            passed_metadata_filter=True,
        )
        assert ranked.relevance_score == 0.0
        assert ranked.semantic_score == 0.0
        assert ranked.llm_reasoning == ""


# -------------------------------------------------------------------
# RankingResult
# -------------------------------------------------------------------


class TestRankingResult:
    """Verify RankingResult aggregation dataclass."""

    def test_populated_result(self) -> None:
        paper = AcademicContent(source="arxiv", title="Paper A")
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.9,
            semantic_score=0.8,
            llm_reasoning="Good match.",
            passed_metadata_filter=True,
        )
        result = RankingResult(
            papers=[ranked],
            total_input=10,
            filtered_by_metadata=2,
            semantically_ranked=8,
        )
        assert len(result.papers) == 1
        assert result.total_input == 10
        assert result.filtered_by_metadata == 2
        assert result.semantically_ranked == 8

    def test_empty_result(self) -> None:
        """RF-04: Empty input produces empty result."""
        result = RankingResult(
            papers=[],
            total_input=0,
            filtered_by_metadata=0,
            semantically_ranked=0,
        )
        assert result.papers == []
        assert result.total_input == 0


# -------------------------------------------------------------------
# RankingError
# -------------------------------------------------------------------


class TestRankingError:
    """Verify RankingError dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        error = RankingError(
            source="semantic_rank",
            url="https://arxiv.org/abs/1234",
            message="Embedding API timeout",
            exception_type="TimeoutError",
        )
        assert error.source == "semantic_rank"
        assert error.url == "https://arxiv.org/abs/1234"
        assert error.message == "Embedding API timeout"
        assert error.exception_type == "TimeoutError"

    def test_minimal_fields(self) -> None:
        error = RankingError(source="metadata_filter")
        assert error.source == "metadata_filter"
        assert error.url is None
        assert error.message == ""
        assert error.exception_type is None
