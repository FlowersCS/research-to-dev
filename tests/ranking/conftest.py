"""Pytest configuration and shared fixtures for ranking tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from research_to_dev.extraction.types import AcademicContent
from research_to_dev.shared.config import RankingConfig


# -------------------------------------------------------------------
# AcademicContent fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_academic_content() -> list[AcademicContent]:
    """A diverse set of 5 AcademicContent items for testing the pipeline."""
    return [
        AcademicContent(
            source="arxiv",
            title="Deep Learning for NLP in 2025",
            url="https://arxiv.org/abs/2501.00001",
            abstract="A comprehensive survey of deep learning techniques for natural language processing published in 2025.",
            authors=["Alice Smith"],
            published_date="2025-01-15",
            metadata={"citations": 42},
        ),
        AcademicContent(
            source="arxiv",
            title="Classical NLP Techniques",
            url="https://arxiv.org/abs/1801.00001",
            abstract="A review of classical NLP methods from the 2010s.",
            authors=["Bob Jones"],
            published_date="2018-03-20",
            metadata={"citations": 10},
        ),
        AcademicContent(
            source="semantic_scholar",
            title="Quantum Computing Advances",
            url="https://semanticscholar.org/paper/abc123",
            abstract="Recent breakthroughs in quantum error correction and their implications for scalable quantum computing.",
            authors=["Carol Chen"],
            published_date="2024-06-01",
            metadata={"citation_count": 88},
        ),
        AcademicContent(
            source="arxiv",
            title="Undated Preprint",
            url="https://arxiv.org/abs/2401.00001",
            abstract="A preprint with no publication date listed.",
            authors=["Dave Miller"],
            published_date=None,
            metadata={},
        ),
        AcademicContent(
            source="arxiv",
            title="Paper Without Abstract",
            url="https://arxiv.org/abs/2301.00001",
            abstract=None,
            authors=["Eve Wilson"],
            published_date="2023-11-10",
            metadata={},
        ),
    ]


# -------------------------------------------------------------------
# Config fixture
# -------------------------------------------------------------------


@pytest.fixture
def ranking_config() -> RankingConfig:
    """Default RankingConfig with test-friendly year."""
    return RankingConfig(
        min_year=2020,
        semantic_top_k=3,
        llm_min_papers=2,
        llm_max_papers=5,
    )


# -------------------------------------------------------------------
# Protocol mock fixtures
# -------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """AsyncMock that satisfies the Embedder Protocol.

    Returns predictable 3-dimensional embeddings so tests can verify
    cosine similarity math without real API calls.
    """
    embedder = AsyncMock()

    # Default: 5 papers → 5 vectors of size 3
    embedder.embed.return_value = [
        [1.0, 0.0, 0.0],  # paper 0 — aligned with query
        [0.0, 1.0, 0.0],  # paper 1 — orthogonal
        [-1.0, 0.0, 0.0],  # paper 2 — opposite
        [0.5, 0.5, 0.0],  # paper 3 — partially aligned
        [0.0, 0.0, 1.0],  # paper 4 — different axis
    ]

    # Query vector aligned with x-axis
    embedder.embed_query.return_value = [1.0, 0.0, 0.0]

    return embedder


@pytest.fixture
def mock_llm_judge() -> AsyncMock:
    """AsyncMock that satisfies the LLMJudge Protocol.

    Returns a fixed judgment for paper_index=0.
    """
    judge = AsyncMock()
    judge.judge.return_value = [
        {
            "paper_index": 0,
            "relevance_score": 0.95,
            "reasoning": "Directly relevant to the query — covers the exact topic.",
        },
        {
            "paper_index": 1,
            "relevance_score": 0.60,
            "reasoning": "Somewhat related methodology.",
        },
    ]
    return judge
