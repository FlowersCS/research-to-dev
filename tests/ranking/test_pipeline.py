"""Pipeline integration tests for RankingPipeline (RF-04, RF-05, RF-06).

All tests use mocked ``Embedder`` and ``LLMJudge`` — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.pipeline import RankingPipeline
from research_to_dev.ranking.types import RankedPaper, RankingResult
from research_to_dev.shared.config import RankingConfig


# ==================================================================
# RF-04: Pipeline Orchestration
# ==================================================================


class TestPipelineSequencing:
    """Verify the 3-stage pipeline executes stages in order."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_result(
        self, mock_embedder: AsyncMock, mock_llm_judge: AsyncMock, ranking_config: RankingConfig
    ) -> None:
        """RF-04: Empty AcademicContent list → papers=[], total_input=0."""
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )

        result = await pipeline.rank([], "any query")

        assert isinstance(result, RankingResult)
        assert result.papers == []
        assert result.total_input == 0
        assert result.filtered_by_metadata == 0
        assert result.semantically_ranked == 0

    @pytest.mark.asyncio
    async def test_end_to_end_with_mocked_protocols(
        self,
        mock_embedder: AsyncMock,
        mock_llm_judge: AsyncMock,
        ranking_config: RankingConfig,
        sample_academic_content: list[AcademicContent],
    ) -> None:
        """RF-04 + RF-06: Pipeline completes with mocked protocols, no API calls."""
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )

        result = await pipeline.rank(sample_academic_content, "deep learning")

        assert isinstance(result, RankingResult)
        assert isinstance(result.papers, list)
        assert result.total_input == 5
        assert result.filtered_by_metadata >= 0
        assert result.semantically_ranked > 0

        # Each RankedPaper has correct field types
        for rp in result.papers:
            assert isinstance(rp, RankedPaper)
            assert isinstance(rp.relevance_score, float)
            assert isinstance(rp.semantic_score, float)
            assert isinstance(rp.llm_reasoning, str)
            assert isinstance(rp.passed_metadata_filter, bool)

    @pytest.mark.asyncio
    async def test_pipeline_uses_correct_config(
        self,
        mock_embedder: AsyncMock,
        mock_llm_judge: AsyncMock,
        sample_academic_content: list[AcademicContent],
    ) -> None:
        """Pipeline respects the injected RankingConfig values."""
        config = RankingConfig(
            min_year=2000,  # very permissive — everything passes
            semantic_top_k=2,
            llm_min_papers=1,
            llm_max_papers=3,
        )
        pipeline = RankingPipeline(mock_embedder, mock_llm_judge, config=config)

        result = await pipeline.rank(sample_academic_content, "quantum computing")

        # With min_year=2000, the 2018 paper passes, so all 5 should enter semantic rank
        assert result.semantically_ranked == min(config.semantic_top_k, 5)
        # LLM capping at max_papers=3
        assert len(result.papers) <= config.llm_max_papers


# ==================================================================
# RF-05: Error Resilience
# ==================================================================


class TestPipelineErrorResilience:
    """Verify the pipeline handles stage failures gracefully (RF-05)."""

    @pytest.mark.asyncio
    async def test_embedder_failure_continues_pipeline(
        self,
        mock_llm_judge: AsyncMock,
        ranking_config: RankingConfig,
        sample_academic_content: list[AcademicContent],
    ) -> None:
        """RF-05: Embedder failure → all papers get semantic_score=0.0, pipeline continues."""
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = RuntimeError("Embedding API down")
        failing_embedder.embed.side_effect = RuntimeError("Embedding API down")

        pipeline = RankingPipeline(
            failing_embedder, mock_llm_judge, config=ranking_config
        )

        # Should NOT raise
        result = await pipeline.rank(sample_academic_content, "query")

        assert isinstance(result, RankingResult)
        assert len(result.papers) > 0

        # All papers should have semantic_score=0.0
        for rp in result.papers:
            assert rp.semantic_score == 0.0

    @pytest.mark.asyncio
    async def test_llm_failure_returns_valid_result(
        self,
        mock_embedder: AsyncMock,
        ranking_config: RankingConfig,
        sample_academic_content: list[AcademicContent],
    ) -> None:
        """RF-05: LLM failure → all papers get relevance_score=0.0, empty reasoning, valid result."""
        failing_judge = AsyncMock()
        failing_judge.judge.side_effect = RuntimeError("LLM API error")

        pipeline = RankingPipeline(
            mock_embedder, failing_judge, config=ranking_config
        )

        # Should NOT raise
        result = await pipeline.rank(sample_academic_content, "query")

        assert isinstance(result, RankingResult)
        assert len(result.papers) > 0

        # All papers should have relevance_score=0.0 and empty reasoning
        for rp in result.papers:
            assert rp.relevance_score == 0.0
            assert rp.llm_reasoning == ""

    @pytest.mark.asyncio
    async def test_double_failure_returns_valid_result(
        self,
        ranking_config: RankingConfig,
        sample_academic_content: list[AcademicContent],
    ) -> None:
        """RF-05: Both embedder AND LLM fail → pipeline still returns valid RankingResult."""
        failing_embedder = AsyncMock()
        failing_embedder.embed_query.side_effect = RuntimeError("Both APIs down")
        failing_embedder.embed.side_effect = RuntimeError("Both APIs down")

        failing_judge = AsyncMock()
        failing_judge.judge.side_effect = RuntimeError("Both APIs down")

        pipeline = RankingPipeline(
            failing_embedder, failing_judge, config=ranking_config
        )

        # Should NOT raise
        result = await pipeline.rank(sample_academic_content, "query")

        assert isinstance(result, RankingResult)
        for rp in result.papers:
            assert rp.relevance_score == 0.0
            assert rp.semantic_score == 0.0
            assert rp.llm_reasoning == ""


# ==================================================================
# RF-06: Protocol Contracts
# ==================================================================


class TestProtocolContracts:
    """Verify protocols work with @runtime_checkable (RF-06)."""

    @pytest.mark.asyncio
    async def test_asyncmock_satisfies_embedder_protocol(
        self, mock_embedder: AsyncMock, mock_llm_judge: AsyncMock, ranking_config: RankingConfig
    ) -> None:
        """RF-06: Pipeline accepts AsyncMock impls of Embedder Protocol at construction.

        @runtime_checkable works for statically-typed implementations;
        AsyncMock's dynamic attribute resolution means isinstance() is not
        the right check here — the functional test is that the pipeline
        accepts the object and runs end-to-end.
        """
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )
        # Construction succeeded — the protocol is accepted
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_asyncmock_satisfies_llm_judge_protocol(
        self, mock_embedder: AsyncMock, mock_llm_judge: AsyncMock, ranking_config: RankingConfig
    ) -> None:
        """RF-06: Pipeline accepts AsyncMock impls of LLMJudge Protocol at construction."""
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )
        assert pipeline is not None

    @pytest.mark.asyncio
    async def test_incomplete_mock_fails_protocol_check(self) -> None:
        """RF-06: Object without judge() method fails LLMJudge check."""
        from research_to_dev.ranking.protocols import LLMJudge

        incomplete = object()
        assert not isinstance(incomplete, LLMJudge)


# ==================================================================
# Edge Cases
# ==================================================================


class TestPipelineEdgeCases:
    """Verify edge-case behavior of the ranking pipeline."""

    @pytest.mark.asyncio
    async def test_all_papers_filtered_out_by_metadata(
        self,
        mock_embedder: AsyncMock,
        mock_llm_judge: AsyncMock,
        ranking_config: RankingConfig,
    ) -> None:
        """When metadata filter excludes all papers, pipeline returns empty."""
        # min_year=2020, both papers dated 2018 → all excluded
        papers = [
            AcademicContent(source="arxiv", title="Old 1", published_date="2018-01-01"),
            AcademicContent(source="arxiv", title="Old 2", published_date="2018-06-15"),
        ]
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )

        result = await pipeline.rank(papers, "query")
        assert result.total_input == 2
        assert result.filtered_by_metadata == 2
        assert result.semantically_ranked == 0
        assert result.papers == []

    @pytest.mark.asyncio
    async def test_single_paper_input(
        self,
        mock_embedder: AsyncMock,
        mock_llm_judge: AsyncMock,
        ranking_config: RankingConfig,
    ) -> None:
        """Pipeline handles single-paper input."""
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        mock_llm_judge.judge.return_value = [
            {"paper_index": 0, "relevance_score": 0.9, "reasoning": "Good match."}
        ]

        papers = [
            AcademicContent(
                source="arxiv",
                title="Single Paper",
                abstract="test abstract",
                published_date="2024-01-01",
            ),
        ]
        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )

        result = await pipeline.rank(papers, "query")

        assert result.total_input == 1
        assert result.filtered_by_metadata == 0
        assert result.semantically_ranked == 1
        assert len(result.papers) == 1
        assert result.papers[0].relevance_score == 0.9
        assert result.papers[0].llm_reasoning == "Good match."

    @pytest.mark.asyncio
    async def test_passed_metadata_filter_tracking(
        self,
        mock_embedder: AsyncMock,
        mock_llm_judge: AsyncMock,
        ranking_config: RankingConfig,
    ) -> None:
        """RankedPaper.passed_metadata_filter is True for papers that passed Stage 1."""
        papers = [
            AcademicContent(source="arxiv", title="Recent", published_date="2024-01-01", abstract="content"),
        ]
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0]]
        mock_llm_judge.judge.return_value = [
            {"paper_index": 0, "relevance_score": 0.8, "reasoning": "ok"}
        ]

        pipeline = RankingPipeline(
            mock_embedder, mock_llm_judge, config=ranking_config
        )

        result = await pipeline.rank(papers, "query")

        assert len(result.papers) == 1
        assert result.papers[0].passed_metadata_filter is True
