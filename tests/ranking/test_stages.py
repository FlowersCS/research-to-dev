"""Unit tests for ranking stage functions (RF-01, RF-02, RF-03, RF-05)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.stages import llm_judge, metadata_filter, semantic_rank


# ==================================================================
# Stage 1: Metadata Filter (RF-01)
# ==================================================================


class TestMetadataFilter:
    """Verify metadata_filter() excludes papers below min_year."""

    def test_filters_papers_below_min_year(self) -> None:
        """RF-01: Papers with published_date year < min_year are excluded."""
        papers = [
            AcademicContent(source="arxiv", title="P2024", published_date="2024-01-01"),
            AcademicContent(source="arxiv", title="P2022", published_date="2022-06-15"),
            AcademicContent(source="arxiv", title="P2018", published_date="2018-03-20"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2020)

        assert len(filtered) == 2
        assert excluded == 1
        assert filtered[0].title == "P2024"
        assert filtered[1].title == "P2022"

    def test_missing_date_passes_through(self) -> None:
        """RF-01: Papers with published_date=None pass the filter; papers below min_year are excluded."""
        papers = [
            AcademicContent(source="arxiv", title="Undated", published_date=None),
            AcademicContent(source="arxiv", title="Old", published_date="2019-01-01"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2020)

        # Undated passes, Old (2019 < 2020) is excluded
        assert len(filtered) == 1
        assert excluded == 1
        assert filtered[0].title == "Undated"

    def test_unparseable_date_passes_through(self) -> None:
        """RF-01: Unparseable date strings pass the filter."""
        papers = [
            AcademicContent(source="arxiv", title="Bad Date", published_date="not-a-date"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2020)
        assert len(filtered) == 1
        assert excluded == 0

    def test_exact_boundary_included(self) -> None:
        """Papers at the min_year boundary are included (inclusive)."""
        papers = [
            AcademicContent(source="arxiv", title="Boundary", published_date="2020-01-01"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2020)
        assert len(filtered) == 1
        assert excluded == 0

    def test_empty_input(self) -> None:
        """Empty input list produces empty output with 0 excluded."""
        filtered, excluded = metadata_filter([], min_year=2020)
        assert filtered == []
        assert excluded == 0

    def test_only_year_prefix_needed(self) -> None:
        """Only the first 4 characters of published_date are used."""
        papers = [
            AcademicContent(source="arxiv", title="Short Date", published_date="2022"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2022)
        assert len(filtered) == 1
        assert excluded == 0

    def test_excludes_all_when_none_qualify(self) -> None:
        """When no papers qualify, only the ones with missing/unparseable dates pass."""
        papers = [
            AcademicContent(source="arxiv", title="P2019", published_date="2019-01-01"),
            AcademicContent(source="arxiv", title="P2018", published_date="2018-01-01"),
        ]
        filtered, excluded = metadata_filter(papers, min_year=2020)
        assert len(filtered) == 0
        assert excluded == 2


# ==================================================================
# Stage 2: Semantic Ranking (RF-02, RF-05)
# ==================================================================


class TestSemanticRank:
    """Verify semantic_rank() computes cosine similarity and selects top-k."""

    @pytest.mark.asyncio
    async def test_top_k_selection(self, mock_embedder: AsyncMock) -> None:
        """RF-02: Returns top_k papers sorted by descending semantic_score."""
        papers = [
            AcademicContent(source="arxiv", title="A", abstract="deep learning advances"),
            AcademicContent(source="arxiv", title="B", abstract="quantum computing theory"),
            AcademicContent(source="arxiv", title="C", abstract="reinforcement learning methods"),
        ]
        # All vectors identical — same cosine sim (1.0)
        mock_embedder.embed.return_value = [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]

        result = await semantic_rank(
            papers, "deep learning", embedder=mock_embedder, top_k=2
        )

        assert len(result) == 2
        # All scores should be ~1.0 since vectors are aligned
        for _, score in result:
            assert abs(score - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_missing_abstract_scores_zero(self, mock_embedder: AsyncMock) -> None:
        """RF-02: Papers with abstract=None get semantic_score=0.0."""
        papers = [
            AcademicContent(source="arxiv", title="With Abstract", abstract="deep learning"),
            AcademicContent(source="arxiv", title="No Abstract", abstract=None),
        ]
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

        result = await semantic_rank(
            papers, "deep learning", embedder=mock_embedder, top_k=2
        )

        assert len(result) == 2
        scores = {paper.title: score for paper, score in result}
        assert scores["No Abstract"] == 0.0
        assert scores["With Abstract"] > 0.0

    @pytest.mark.asyncio
    async def test_cosine_similarity_correctness(self, mock_embedder: AsyncMock) -> None:
        """RF-02: Cosine similarity matches expected values for known vectors."""
        papers = [
            AcademicContent(source="arxiv", title="Aligned", abstract="same direction"),
            AcademicContent(source="arxiv", title="Orthogonal", abstract="ninety degrees"),
            AcademicContent(source="arxiv", title="Opposite", abstract="opposite direction"),
        ]
        # query = [1, 0, 0], papers = [[1, 0, 0], [0, 1, 0], [-1, 0, 0]]
        # Expected: 1.0, 0.0, -1.0
        mock_embedder.embed.return_value = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]

        result = await semantic_rank(
            papers, "directional test", embedder=mock_embedder, top_k=3
        )

        scores = {paper.title: score for paper, score in result}
        assert abs(scores["Aligned"] - 1.0) < 0.001
        assert abs(scores["Orthogonal"] - 0.0) < 0.001
        assert abs(scores["Opposite"] - (-1.0)) < 0.001

    @pytest.mark.asyncio
    async def test_embedder_failure_returns_zeros(self) -> None:
        """RF-05: Embedder failure → all papers get semantic_score=0.0, pipeline continues."""
        papers = [
            AcademicContent(source="arxiv", title="Paper 1", abstract="content 1"),
            AcademicContent(source="arxiv", title="Paper 2", abstract="content 2"),
        ]
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = RuntimeError("API down")
        failing_embedder.embed.side_effect = RuntimeError("API down")

        result = await semantic_rank(
            papers, "query", embedder=failing_embedder, top_k=2
        )

        assert len(result) == 2
        for _, score in result:
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_empty_abstracts_all_zero(self, mock_embedder: AsyncMock) -> None:
        """All papers with empty abstracts get semantic_score=0.0."""
        papers = [
            AcademicContent(source="arxiv", title="Empty 1", abstract=""),
            AcademicContent(source="arxiv", title="Empty 2", abstract=None),
        ]

        result = await semantic_rank(
            papers, "query", embedder=mock_embedder, top_k=2
        )

        for _, score in result:
            assert score == 0.0

    @pytest.mark.asyncio
    async def test_top_k_caps_output(self, mock_embedder: AsyncMock) -> None:
        """Output length never exceeds top_k."""
        papers = [
            AcademicContent(source="arxiv", title=f"Paper {i}", abstract=f"content {i}")
            for i in range(10)
        ]
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0] for _ in range(10)]

        result = await semantic_rank(
            papers, "query", embedder=mock_embedder, top_k=3
        )

        assert len(result) == 3


# ==================================================================
# Stage 3: LLM Judgment (RF-03, RF-05)
# ==================================================================


class TestLLMJudge:
    """Verify llm_judge() returns scored and reasoned results."""

    @pytest.mark.asyncio
    async def test_normal_judgment_returns_papers(self, mock_llm_judge: AsyncMock) -> None:
        """RF-03: LLM returns papers with relevance_score and reasoning."""
        papers = [
            AcademicContent(source="arxiv", title="NLP Survey", abstract="deep learning review"),
            AcademicContent(source="arxiv", title="Vision Paper", abstract="computer vision"),
        ]
        ranked = [(papers[0], 0.8), (papers[1], 0.5)]

        result = await llm_judge(
            ranked, "NLP research", judge=mock_llm_judge,
            min_papers=2, max_papers=5,
        )

        assert len(result) == 2
        for r in result:
            assert "paper_index" in r
            assert "relevance_score" in r
            assert "reasoning" in r
            assert isinstance(r["relevance_score"], float)

    @pytest.mark.asyncio
    async def test_llm_failure_returns_degraded(self) -> None:
        """RF-05: LLM failure → all papers get relevance_score=0.0, empty reasoning."""
        papers = [
            AcademicContent(source="arxiv", title="Paper 1", abstract="content"),
            AcademicContent(source="arxiv", title="Paper 2", abstract="content"),
        ]
        ranked = [(papers[0], 0.8), (papers[1], 0.5)]

        failing_judge = AsyncMock()
        failing_judge.judge.side_effect = RuntimeError("LLM API error")

        result = await llm_judge(
            ranked, "query", judge=failing_judge,
            min_papers=2, max_papers=5,
        )

        assert len(result) == 2
        for r in result:
            assert r["relevance_score"] == 0.0
            assert r["reasoning"] == ""

    @pytest.mark.asyncio
    async def test_max_papers_caps_output(self, mock_llm_judge: AsyncMock) -> None:
        """Output is capped at max_papers."""
        # Return 5 judgments from the mock
        mock_llm_judge.judge.return_value = [
            {"paper_index": i, "relevance_score": 1.0 - i * 0.1, "reasoning": f"Reason {i}"}
            for i in range(5)
        ]

        papers = [
            AcademicContent(source="arxiv", title=f"Paper {i}", abstract=f"content {i}")
            for i in range(5)
        ]
        ranked = [(p, 0.5) for p in papers]

        result = await llm_judge(
            ranked, "query", judge=mock_llm_judge,
            min_papers=2, max_papers=3,
        )

        assert len(result) == 3  # capped at max_papers

    @pytest.mark.asyncio
    async def test_invalid_scores_filtered_out(self, mock_llm_judge: AsyncMock) -> None:
        """Results with out-of-range or missing relevance_score are filtered."""
        mock_llm_judge.judge.return_value = [
            {"paper_index": 0, "relevance_score": 0.8, "reasoning": "good"},
            {"paper_index": 1, "relevance_score": 1.5, "reasoning": "out of range"},  # invalid
            {"paper_index": 2, "reasoning": "missing score"},  # missing
            {"paper_index": 3, "relevance_score": -0.5, "reasoning": "negative"},  # invalid
        ]

        papers = [
            AcademicContent(source="arxiv", title=f"P{i}", abstract=f"a{i}")
            for i in range(4)
        ]
        ranked = [(p, 0.5) for p in papers]

        result = await llm_judge(
            ranked, "query", judge=mock_llm_judge,
            min_papers=1, max_papers=10,
        )

        # Only paper_index=0 should survive
        assert len(result) == 1
        assert result[0]["paper_index"] == 0
        assert result[0]["relevance_score"] == 0.8

    @pytest.mark.asyncio
    async def test_passes_paper_metadata_to_judge(self, mock_llm_judge: AsyncMock) -> None:
        """Verify the paper dicts sent to the judge contain expected fields."""
        paper = AcademicContent(
            source="arxiv",
            title="Test Paper",
            abstract="Test abstract",
            url="https://arxiv.org/abs/test",
        )
        ranked = [(paper, 0.75)]

        await llm_judge(
            ranked, "research query", judge=mock_llm_judge,
            min_papers=1, max_papers=5,
        )

        # Verify judge was called with correct structure
        call_args = mock_llm_judge.judge.call_args
        assert call_args is not None
        query_arg, papers_arg = call_args[0]

        assert query_arg == "research query"
        assert len(papers_arg) == 1
        assert papers_arg[0]["paper_index"] == 0
        assert papers_arg[0]["title"] == "Test Paper"
        assert papers_arg[0]["abstract"] == "Test abstract"
        assert papers_arg[0]["semantic_score"] == 0.75
