"""Unit tests for paper_profile data types and configuration."""

from __future__ import annotations

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.paper_profile.types import (
    Claim,
    PaperProfile,
    ProfilingError,
    ProfilingResult,
    Section,
)
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig


# -------------------------------------------------------------------
# ProfileConfig
# -------------------------------------------------------------------


class TestProfileConfig:
    """Verify ProfileConfig dataclass creation and defaults."""

    def test_default_values_are_correct(self) -> None:
        """model defaults to 'gpt-4o-mini', max_claims_per_paper defaults to 20."""
        cfg = ProfileConfig()
        assert cfg.model == "gpt-4o-mini"
        assert cfg.max_claims_per_paper == 20

    def test_custom_values_preserved_exactly(self) -> None:
        """Custom values preserve exactly."""
        cfg = ProfileConfig(model="gpt-4o", max_claims_per_paper=5)
        assert cfg.model == "gpt-4o"
        assert cfg.max_claims_per_paper == 5


# -------------------------------------------------------------------
# Section
# -------------------------------------------------------------------


class TestSection:
    """Verify Section dataclass creation."""

    def test_full_fields(self) -> None:
        section = Section(name="methods", content="We used a ResNet-50 backbone.")
        assert section.name == "methods"
        assert section.content == "We used a ResNet-50 backbone."

    def test_empty_content(self) -> None:
        section = Section(name="abstract", content="")
        assert section.name == "abstract"
        assert section.content == ""


# -------------------------------------------------------------------
# Claim
# -------------------------------------------------------------------


class TestClaim:
    """Verify Claim dataclass creation and provenance."""

    def test_full_fields(self) -> None:
        claim = Claim(
            text="Our method achieves 94.2% accuracy.",
            paper_id="arxiv:abc123def456",
            section_name="results",
        )
        assert claim.text == "Our method achieves 94.2% accuracy."
        assert claim.paper_id == "arxiv:abc123def456"
        assert claim.section_name == "results"


# -------------------------------------------------------------------
# PaperProfile — wrapper traceability (AD-02, Spec: Wrap RankedPaper)
# -------------------------------------------------------------------


class TestPaperProfile:
    """Verify PaperProfile wraps RankedPaper preserving traceability."""

    def test_wraps_ranked_paper_with_full_traceability(self) -> None:
        """AD-02: ranked_paper is preserved unmodified."""
        paper = AcademicContent(
            source="arxiv",
            title="Test Paper",
            url="https://arxiv.org/abs/1234",
            abstract="Test abstract.",
        )
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.95,
            semantic_score=0.88,
            llm_reasoning="Good match.",
            passed_metadata_filter=True,
        )
        sections = [Section(name="abstract", content="Test abstract.")]
        claims = [
            Claim(
                text="Achieves 95% accuracy.",
                paper_id="https://arxiv.org/abs/1234",
                section_name="abstract",
            ),
        ]

        profile = PaperProfile(
            ranked_paper=ranked,
            sections=sections,
            claims=claims,
        )

        # Wrapper traceability
        assert profile.ranked_paper is ranked
        assert profile.ranked_paper.paper is paper
        assert profile.ranked_paper.relevance_score == 0.95
        assert profile.ranked_paper.semantic_score == 0.88
        assert profile.ranked_paper.llm_reasoning == "Good match."
        assert profile.ranked_paper.passed_metadata_filter is True

        # Enriched fields
        assert len(profile.sections) == 1
        assert profile.sections[0].name == "abstract"
        assert len(profile.claims) == 1
        assert profile.claims[0].text == "Achieves 95% accuracy."

    def test_default_empty_lists(self) -> None:
        """sections and claims default to empty lists."""
        paper = AcademicContent(source="arxiv", title="Empty")
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.0,
            semantic_score=0.0,
            llm_reasoning="",
            passed_metadata_filter=False,
        )
        profile = PaperProfile(ranked_paper=ranked)
        assert profile.sections == []
        assert profile.claims == []


# -------------------------------------------------------------------
# ProfilingError
# -------------------------------------------------------------------


class TestProfilingError:
    """Verify ProfilingError dataclass creation and defaults."""

    def test_full_fields(self) -> None:
        error = ProfilingError(
            paper_title="Test Paper",
            message="LLM call failed",
            exception_type="RuntimeError",
        )
        assert error.paper_title == "Test Paper"
        assert error.message == "LLM call failed"
        assert error.exception_type == "RuntimeError"

    def test_minimal_fields(self) -> None:
        error = ProfilingError(
            paper_title="Minimal Paper",
            message="Unknown error",
        )
        assert error.paper_title == "Minimal Paper"
        assert error.message == "Unknown error"
        assert error.exception_type is None


# -------------------------------------------------------------------
# ProfilingResult
# -------------------------------------------------------------------


class TestProfilingResult:
    """Verify ProfilingResult aggregation dataclass."""

    def test_populated_result(self) -> None:
        paper = AcademicContent(source="arxiv", title="Paper A")
        ranked = RankedPaper(
            paper=paper,
            relevance_score=0.9,
            semantic_score=0.8,
            llm_reasoning="Good match.",
            passed_metadata_filter=True,
        )
        profile = PaperProfile(ranked_paper=ranked)
        error = ProfilingError(
            paper_title="Paper B",
            message="Failed to profile",
        )

        result = ProfilingResult(
            profiles=[profile],
            total_input=2,
            errors=[error],
        )
        assert len(result.profiles) == 1
        assert result.total_input == 2
        assert len(result.errors) == 1
        assert result.errors[0].paper_title == "Paper B"

    def test_empty_result(self) -> None:
        """Empty input produces empty result."""
        result = ProfilingResult(profiles=[], total_input=0)
        assert result.profiles == []
        assert result.total_input == 0
        assert result.errors == []
