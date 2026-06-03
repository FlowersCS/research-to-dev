"""Pipeline integration tests for ProfilingPipeline.

All tests use a mocked ``SectionExtractor`` — no real API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.paper_profile.pipeline import ProfilingPipeline
from research_to_dev.paper_profile.types import PaperProfile, ProfilingResult
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig


class TestPipelineSequencing:
    """Verify the profiling pipeline runs correctly end-to-end."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_result(
        self, mock_extractor: AsyncMock, profile_config: ProfileConfig
    ) -> None:
        """Spec: Empty input → ProfilingResult with empty profiles and total_input=0."""
        pipeline = ProfilingPipeline(
            mock_extractor, config=profile_config
        )

        result = await pipeline.profile([])

        assert isinstance(result, ProfilingResult)
        assert result.profiles == []
        assert result.total_input == 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_end_to_end_with_mocked_extractor(
        self,
        mock_extractor: AsyncMock,
        profile_config: ProfileConfig,
        sample_ranked_papers: list[RankedPaper],
    ) -> None:
        """Pipeline completes with mocked extractor, returns correct shapes."""
        pipeline = ProfilingPipeline(
            mock_extractor, config=profile_config
        )

        result = await pipeline.profile(sample_ranked_papers)

        assert isinstance(result, ProfilingResult)
        assert isinstance(result.profiles, list)
        assert result.total_input == 3
        assert len(result.profiles) == 3

        # Each profile is a valid PaperProfile
        for i, profile in enumerate(result.profiles):
            assert isinstance(profile, PaperProfile)
            assert profile.ranked_paper is sample_ranked_papers[i]
            assert isinstance(profile.sections, list)
            assert isinstance(profile.claims, list)

    @pytest.mark.asyncio
    async def test_pipeline_uses_correct_config(
        self,
        mock_extractor: AsyncMock,
        sample_ranked_papers: list[RankedPaper],
    ) -> None:
        """Pipeline respects the injected ProfileConfig."""
        config = ProfileConfig(model="custom-model", max_claims_per_paper=3)
        pipeline = ProfilingPipeline(mock_extractor, config=config)

        result = await pipeline.profile(sample_ranked_papers)

        # All profiles should have claims capped at 3
        for profile in result.profiles:
            assert len(profile.claims) <= 3


class TestPipelineErrorResilience:
    """Verify the pipeline handles extractor failures gracefully (RF-05)."""

    @pytest.mark.asyncio
    async def test_extractor_failure_produces_degraded_profiles(
        self,
        profile_config: ProfileConfig,
        sample_ranked_papers: list[RankedPaper],
    ) -> None:
        """RF-05: Extractor failure → degraded profiles with errors, pipeline does NOT raise."""
        failing_extractor = AsyncMock()
        failing_extractor.extract.side_effect = RuntimeError(
            "LLM API connection error"
        )

        pipeline = ProfilingPipeline(
            failing_extractor, config=profile_config
        )

        # Should NOT raise
        result = await pipeline.profile(sample_ranked_papers)

        assert isinstance(result, ProfilingResult)
        assert result.total_input == 3
        assert len(result.profiles) == 3
        assert len(result.errors) == 3

        # All profiles degraded
        for profile in result.profiles:
            assert isinstance(profile, PaperProfile)
            assert profile.sections == []
            assert profile.claims == []

        # All errors recorded
        for error in result.errors:
            assert "LLM API connection error" in error.message
            assert error.exception_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_partial_failure_mixed_results(
        self,
        profile_config: ProfileConfig,
        sample_ranked_papers: list[RankedPaper],
    ) -> None:
        """Some papers succeed, some fail → mix of good profiles and errors."""
        call_count = 0

        async def flaky_extract(title, abstract):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Second paper fails
                raise RuntimeError("Transient error")
            return {
                "sections": [
                    {
                        "name": "abstract",
                        "content": "test",
                        "claims": [{"text": "Claim from paper"}],
                    }
                ],
            }

        flaky = AsyncMock()
        flaky.extract.side_effect = flaky_extract

        pipeline = ProfilingPipeline(flaky, config=profile_config)

        result = await pipeline.profile(sample_ranked_papers)

        assert result.total_input == 3
        assert len(result.profiles) == 3

        # Paper 0: success (has claims)
        assert len(result.profiles[0].claims) > 0
        # Paper 1: failed (empty)
        assert result.profiles[1].claims == []
        # Paper 2: success (has claims)
        assert len(result.profiles[2].claims) > 0

        # One error recorded (for paper 1)
        assert len(result.errors) == 1
        assert "Transient error" in result.errors[0].message

    @pytest.mark.asyncio
    async def test_ultimate_safety_net(
        self,
        profile_config: ProfileConfig,
        sample_ranked_paper: RankedPaper,
    ) -> None:
        """Even if the extractor does something weird (returns non-dict), pipeline survives."""
        weird_extractor = AsyncMock()
        weird_extractor.extract.return_value = 42  # not a dict

        papers = [
            RankedPaper(
                paper=sample_ranked_paper.paper,
                relevance_score=0.5,
                semantic_score=0.5,
                llm_reasoning="",
                passed_metadata_filter=True,
            ),
        ]

        pipeline = ProfilingPipeline(weird_extractor, config=profile_config)

        # Should NOT raise
        result = await pipeline.profile(papers)

        assert len(result.profiles) == 1
        # profile_paper catches it → degraded + error
        assert result.profiles[0].claims == []
        assert len(result.errors) == 1


class TestPipelineConfigInjection:
    """Verify config injection and defaults."""

    @pytest.mark.asyncio
    async def test_default_config_used_when_none_provided(
        self, mock_extractor: AsyncMock
    ) -> None:
        """Pipeline uses ProfileConfig() defaults when no config is injected."""
        pipeline = ProfilingPipeline(mock_extractor)

        assert pipeline._config.model == "gpt-4o-mini"
        assert pipeline._config.max_claims_per_paper == 20

    @pytest.mark.asyncio
    async def test_custom_config_injected(
        self, mock_extractor: AsyncMock
    ) -> None:
        """Custom ProfileConfig is preserved after injection."""
        config = ProfileConfig(model="gpt-4o", max_claims_per_paper=50)
        pipeline = ProfilingPipeline(mock_extractor, config=config)

        assert pipeline._config is config
        assert pipeline._config.model == "gpt-4o"
        assert pipeline._config.max_claims_per_paper == 50
