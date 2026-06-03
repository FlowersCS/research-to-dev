"""Unit tests for paper_profile stage functions."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.paper_profile.stages import build_paper_id, profile_paper
from research_to_dev.paper_profile.types import PaperProfile, ProfilingError
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig


# ==================================================================
# build_paper_id (AD-04)
# ==================================================================


class TestBuildPaperId:
    """Verify build_paper_id() constructs deterministic identifiers."""

    def test_uses_url_when_available(self) -> None:
        """AD-04: URL is preferred when present."""
        pid = build_paper_id(
            url="https://arxiv.org/abs/2501.00001",
            source="arxiv",
            title="Some Paper",
        )
        assert pid == "https://arxiv.org/abs/2501.00001"

    def test_falls_back_to_hash_when_no_url(self) -> None:
        """When URL is None, produces source:hash format."""
        pid = build_paper_id(
            url=None,
            source="arxiv",
            title="Deep Learning Advances",
        )
        assert pid.startswith("arxiv:")
        # Hash should be 12 hex chars after the prefix
        hash_part = pid.split(":", 1)[1]
        assert len(hash_part) == 12

    def test_deterministic_for_same_input(self) -> None:
        """Same input always produces the same ID."""
        a = build_paper_id(None, "arxiv", "Test Paper")
        b = build_paper_id(None, "arxiv", "Test Paper")
        assert a == b

    def test_different_titles_produce_different_ids(self) -> None:
        """Different titles produce different hashes."""
        a = build_paper_id(None, "arxiv", "Paper A")
        b = build_paper_id(None, "arxiv", "Paper B")
        assert a != b

    def test_different_sources_produce_different_ids(self) -> None:
        """Different sources produce different hashes for the same title."""
        a = build_paper_id(None, "arxiv", "Same Title")
        b = build_paper_id(None, "semantic_scholar", "Same Title")
        assert a != b

    def test_url_takes_precedence_over_hash(self) -> None:
        """URL is used even when source and title are also available."""
        pid = build_paper_id(
            url="https://semanticscholar.org/paper/abc",
            source="arxiv",
            title="Irrelevant",
        )
        assert pid == "https://semanticscholar.org/paper/abc"


# ==================================================================
# profile_paper (AD-03, RF-05)
# ==================================================================


class TestProfilePaper:
    """Verify profile_paper() extracts sections+claims and handles failures."""

    @pytest.mark.asyncio
    async def test_success_path_constructs_paper_profile(
        self,
        sample_ranked_paper: RankedPaper,
        mock_extractor: AsyncMock,
        profile_config: ProfileConfig,
    ) -> None:
        """AD-03: Extractor called once, PaperProfile has sections and claims."""
        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=mock_extractor,
            config=profile_config,
        )

        assert isinstance(profile, PaperProfile)
        assert error is None
        assert profile.ranked_paper is sample_ranked_paper
        assert len(profile.sections) == 2
        assert profile.sections[0].name == "abstract"
        assert profile.sections[1].name == "methods"
        assert len(profile.claims) == 3

        # Verify extractor was called once with correct args
        mock_extractor.extract.assert_called_once_with(
            "Deep Learning for NLP in 2025",
            "A comprehensive survey of deep learning techniques for natural language processing published in 2025.",
        )

    @pytest.mark.asyncio
    async def test_extractor_failure_returns_degraded_profile(
        self,
        sample_ranked_paper: RankedPaper,
        profile_config: ProfileConfig,
    ) -> None:
        """RF-05: Extractor raises → degraded PaperProfile with ProfilingError."""
        failing_extractor = AsyncMock()
        failing_extractor.extract.side_effect = RuntimeError(
            "API connection error"
        )

        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=failing_extractor,
            config=profile_config,
        )

        assert isinstance(profile, PaperProfile)
        assert profile.ranked_paper is sample_ranked_paper
        assert profile.sections == []
        assert profile.claims == []

        assert isinstance(error, ProfilingError)
        assert error.paper_title == "Deep Learning for NLP in 2025"
        assert "API connection error" in error.message
        assert error.exception_type == "RuntimeError"

    @pytest.mark.asyncio
    async def test_missing_abstract_calls_extractor_with_empty_string(
        self,
        mock_extractor: AsyncMock,
    ) -> None:
        """When abstract is None, extractor receives empty string (Spec: Abstract available)."""
        paper = sample_ranked_paper_ac(
            source="arxiv",
            title="No Abstract Paper",
            abstract=None,
        )
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.5,
            semantic_score=0.0,
            llm_reasoning="",
            passed_metadata_filter=True,
        )

        await profile_paper(ranked, extractor=mock_extractor)

        mock_extractor.extract.assert_called_once_with(
            "No Abstract Paper", ""
        )

    @pytest.mark.asyncio
    async def test_empty_sections_in_response(
        self,
        sample_ranked_paper: RankedPaper,
        profile_config: ProfileConfig,
    ) -> None:
        """When LLM returns empty sections, profile has empty lists."""
        empty_extractor = AsyncMock()
        empty_extractor.extract.return_value = {"sections": []}

        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=empty_extractor,
            config=profile_config,
        )

        assert profile.sections == []
        assert profile.claims == []
        assert error is None

    @pytest.mark.asyncio
    async def test_invalid_response_structure_returns_degraded(
        self,
        sample_ranked_paper: RankedPaper,
        profile_config: ProfileConfig,
    ) -> None:
        """RF-05: Missing 'sections' key or non-list → degraded profile."""
        bad_extractor = AsyncMock()
        bad_extractor.extract.return_value = {"wrong_key": "value"}

        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=bad_extractor,
            config=profile_config,
        )

        assert profile.sections == []
        assert profile.claims == []
        assert isinstance(error, ProfilingError)
        assert "Invalid response structure" in error.message

    @pytest.mark.asyncio
    async def test_invalid_json_from_extractor(
        self,
        sample_ranked_paper: RankedPaper,
        profile_config: ProfileConfig,
    ) -> None:
        """RF-05: Extractor returns non-dict → degraded profile."""
        bad_extractor = AsyncMock()
        bad_extractor.extract.return_value = "not a dict"

        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=bad_extractor,
            config=profile_config,
        )

        assert profile.sections == []
        assert error is not None

    @pytest.mark.asyncio
    async def test_claims_capped_at_max_claims_per_paper(
        self,
        sample_ranked_paper: RankedPaper,
    ) -> None:
        """Claims are capped at config.max_claims_per_paper."""
        many_claims_extractor = AsyncMock()
        many_claims_extractor.extract.return_value = {
            "sections": [
                {
                    "name": "results",
                    "content": "Many claims.",
                    "claims": [
                        {"text": f"Claim number {i}"} for i in range(15)
                    ],
                },
            ],
        }
        tight_config = ProfileConfig(max_claims_per_paper=5)

        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=many_claims_extractor,
            config=tight_config,
        )

        assert error is None
        assert len(profile.claims) == 5

    @pytest.mark.asyncio
    async def test_duplicate_claims_filtered(
        self,
        sample_ranked_paper: RankedPaper,
    ) -> None:
        """Duplicate claim texts are filtered with a warning."""
        dup_extractor = AsyncMock()
        dup_extractor.extract.return_value = {
            "sections": [
                {
                    "name": "abstract",
                    "content": "test",
                    "claims": [
                        {"text": "Unique claim."},
                        {"text": "Duplicate claim."},
                        {"text": "Duplicate claim."},
                    ],
                },
            ],
        }

        profile, error = await profile_paper(
            sample_ranked_paper, extractor=dup_extractor
        )

        assert error is None
        assert len(profile.claims) == 2  # Duplicate filtered
        assert profile.claims[0].text == "Unique claim."
        assert profile.claims[1].text == "Duplicate claim."

    @pytest.mark.asyncio
    async def test_claim_provenance_includes_paper_id_and_section(
        self,
        sample_ranked_paper: RankedPaper,
        mock_extractor: AsyncMock,
        profile_config: ProfileConfig,
    ) -> None:
        """AD-06: Each claim carries paper_id and section_name for traceability."""
        profile, error = await profile_paper(
            sample_ranked_paper,
            extractor=mock_extractor,
            config=profile_config,
        )

        assert error is None
        assert len(profile.claims) > 0

        paper_id = profile.claims[0].paper_id
        assert paper_id == "https://arxiv.org/abs/2501.00001"

        for claim in profile.claims:
            assert claim.paper_id == paper_id
            assert claim.section_name in {"abstract", "methods"}

    @pytest.mark.asyncio
    async def test_skips_malformed_section_entries(
        self,
        sample_ranked_paper: RankedPaper,
    ) -> None:
        """Malformed section entries (non-dict, missing name/content) are skipped."""
        messy_extractor = AsyncMock()
        messy_extractor.extract.return_value = {
            "sections": [
                "not a dict",  # skipped
                {"name": "abstract", "content": "valid"},  # kept
                {"name": 123, "content": "bad name type"},  # skipped
                {"no_name": "missing"},  # skipped
                {"name": "results", "content": "valid results"},  # kept
            ],
        }

        profile, error = await profile_paper(
            sample_ranked_paper, extractor=messy_extractor
        )

        assert error is None
        assert len(profile.sections) == 2
        assert profile.sections[0].name == "abstract"
        assert profile.sections[1].name == "results"


# -------------------------------------------------------------------
# Helper
# -------------------------------------------------------------------


def sample_ranked_paper_ac(**kwargs) -> AcademicContent:
    """Build an AcademicContent with given overrides."""
    from research_to_dev.extraction.types import AcademicContent

    defaults = {
        "source": "arxiv",
        "title": "Test Paper",
    }
    defaults.update(kwargs)
    return AcademicContent(**defaults)
