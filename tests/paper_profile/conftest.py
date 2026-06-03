"""Pytest configuration and shared fixtures for paper_profile tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import ProfileConfig


# -------------------------------------------------------------------
# ProfileConfig fixture
# -------------------------------------------------------------------


@pytest.fixture
def profile_config() -> ProfileConfig:
    """Default ProfileConfig with test-friendly limits."""
    return ProfileConfig(
        model="gpt-4o-mini",
        max_claims_per_paper=10,
    )


# -------------------------------------------------------------------
# AcademicContent fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_academic_content() -> AcademicContent:
    """A single AcademicContent item for stage function tests."""
    return AcademicContent(
        source="arxiv",
        title="Deep Learning for NLP in 2025",
        url="https://arxiv.org/abs/2501.00001",
        abstract="A comprehensive survey of deep learning techniques for natural language processing published in 2025.",
        authors=["Alice Smith", "Bob Jones"],
        published_date="2025-01-15",
        metadata={"citations": 42},
    )


@pytest.fixture
def sample_ranked_paper(sample_academic_content: AcademicContent) -> RankedPaper:
    """A RankedPaper wrapping the sample AcademicContent."""
    return RankedPaper(
        paper=sample_academic_content,
        relevance_score=0.85,
        semantic_score=0.72,
        llm_reasoning="Highly relevant to the research topic.",
        passed_metadata_filter=True,
    )


@pytest.fixture
def sample_ranked_papers() -> list[RankedPaper]:
    """Multiple RankedPaper items for pipeline tests."""
    papers = [
        AcademicContent(
            source="arxiv",
            title="Deep Learning Advances 2025",
            url="https://arxiv.org/abs/2501.00001",
            abstract="Survey of deep learning breakthroughs including attention mechanisms, transformers, and large language models.",
            authors=["Alice Smith"],
            published_date="2025-01-15",
        ),
        AcademicContent(
            source="semantic_scholar",
            title="Quantum Error Correction",
            url="https://semanticscholar.org/paper/abc123",
            abstract="Recent breakthroughs in quantum error correction achieve 99.9% fidelity using surface codes.",
            authors=["Carol Chen"],
            published_date="2024-06-01",
        ),
        AcademicContent(
            source="arxiv",
            title="Paper Without Abstract",
            url="https://arxiv.org/abs/2301.00001",
            abstract=None,
            authors=["Eve Wilson"],
            published_date="2023-11-10",
        ),
    ]
    return [
        RankedPaper(
            paper=p,
            relevance_score=0.9 - i * 0.1,
            semantic_score=0.8 - i * 0.2,
            llm_reasoning=f"Reason {i}",
            passed_metadata_filter=True,
        )
        for i, p in enumerate(papers)
    ]


# -------------------------------------------------------------------
# Protocol mock fixture
# -------------------------------------------------------------------


@pytest.fixture
def mock_extractor() -> AsyncMock:
    """AsyncMock that satisfies the SectionExtractor Protocol.

    Returns a predictable JSON dict with sections and nested claims.
    """
    extractor = AsyncMock()
    extractor.extract.return_value = {
        "sections": [
            {
                "name": "abstract",
                "content": "A comprehensive survey of deep learning techniques.",
                "claims": [
                    {"text": "Deep learning achieves state-of-the-art in NLP tasks."},
                    {"text": "Transformer architecture dominates NLP research in 2025."},
                ],
            },
            {
                "name": "methods",
                "content": "We surveyed 200 papers from 2020-2025.",
                "claims": [
                    {"text": "Survey covers 200 papers from major NLP venues."},
                ],
            },
        ],
    }
    return extractor


# -------------------------------------------------------------------
# Raw LLM response fixture
# -------------------------------------------------------------------


@pytest.fixture
def sample_llm_response() -> dict:
    """A realistic LLM JSON response for testing parsing and validation."""
    return {
        "sections": [
            {
                "name": "abstract",
                "content": "We present a novel approach to machine learning.",
                "claims": [
                    {"text": "Our method achieves 94.2% accuracy on ImageNet."},
                    {"text": "We outperform the baseline by 3.1 percentage points."},
                ],
            },
            {
                "name": "methods",
                "content": "We used a ResNet-50 backbone with custom augmentations.",
                "claims": [
                    {"text": "Training uses Adam optimizer with learning rate 1e-4."},
                ],
            },
            {
                "name": "results",
                "content": "Results show consistent improvement across benchmarks.",
                "claims": [
                    {"text": "Ablation study confirms data augmentation is critical."},
                ],
            },
        ],
    }
