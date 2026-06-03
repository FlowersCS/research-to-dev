"""Pytest configuration and shared fixtures for correlation tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from research_to_dev.codebase.types import CodebaseContext, Component, ModuleSummary
from research_to_dev.extraction.types import AcademicContent
from research_to_dev.paper_profile.types import Claim, PaperProfile, Section
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import CorrelationMapConfig


# -------------------------------------------------------------------
# CorrelationMapConfig fixture
# -------------------------------------------------------------------


@pytest.fixture
def correlation_config() -> CorrelationMapConfig:
    """Default CorrelationMapConfig with test-friendly settings."""
    return CorrelationMapConfig(
        similarity_threshold=0.7,
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
    )


# -------------------------------------------------------------------
# Mock embedder — returns fixed vectors (10-dimensional)
# -------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """AsyncMock that satisfies the Embedder Protocol.

    Returns fixed 10-dimensional vectors so cosine similarity values
    are deterministic and easy to reason about in tests.
    """
    embedder = AsyncMock()

    async def _embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for i, t in enumerate(texts):
            text = t.strip()
            if not text:
                vectors.append([0.0] * 10)
            else:
                # Deterministic vector based on position — not on content
                # so tests can control similarity by fixture design.
                seed = hash(text) % 100
                rng = np.random.default_rng(abs(seed) + 42)
                vec = rng.uniform(-1, 1, size=10).tolist()
                vectors.append(vec)
        return vectors

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


# -------------------------------------------------------------------
# Mock classifier — returns validated matches from a lookup dict
# -------------------------------------------------------------------


@pytest.fixture
def mock_classifier() -> AsyncMock:
    """AsyncMock that satisfies the CorrelationClassifier Protocol.

    Returns a fixed set of validated matches for any input, unless
    overridden in specific tests.
    """
    classifier = AsyncMock()

    async def _classify(
        title: str, paper_claims: str, candidates: list[dict]
    ) -> list[dict]:
        # Default: validate all candidates
        results = []
        for c in candidates:
            results.append({
                "claim_text": c["claim_text"],
                "target_name": c["target_name"],
                "correlation_type": "direct_solution",
                "reasoning": f"Mock reasoning for {c['target_name']}",
            })
        return results

    classifier.classify = AsyncMock(side_effect=_classify)
    return classifier


# -------------------------------------------------------------------
# Sample PaperProfile fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_academic_content() -> AcademicContent:
    """A single AcademicContent item for building PaperProfiles."""
    return AcademicContent(
        source="arxiv",
        title="Deep Learning for NLP",
        url="https://arxiv.org/abs/2501.00001",
        abstract="A survey of deep learning techniques for NLP.",
        authors=["Alice Smith"],
        published_date="2025-01-15",
    )


@pytest.fixture
def sample_ranked_paper(sample_academic_content: AcademicContent) -> RankedPaper:
    """A RankedPaper wrapping sample AcademicContent."""
    return RankedPaper(
        paper=sample_academic_content,
        relevance_score=0.85,
        semantic_score=0.72,
        llm_reasoning="Relevant",
        passed_metadata_filter=True,
    )


@pytest.fixture
def sample_claims() -> list[Claim]:
    """A set of sample claims for testing."""
    return [
        Claim(
            text="Attention mechanisms improve NLP accuracy",
            paper_id="paper-1",
            section_name="abstract",
        ),
        Claim(
            text="Transformer architecture enables parallelization",
            paper_id="paper-1",
            section_name="methods",
        ),
        Claim(
            text="Our method uses gradient descent for optimization",
            paper_id="paper-1",
            section_name="methods",
        ),
    ]


@pytest.fixture
def sample_paper_profile(
    sample_ranked_paper: RankedPaper, sample_claims: list[Claim],
) -> PaperProfile:
    """A single PaperProfile with sample claims."""
    return PaperProfile(
        ranked_paper=sample_ranked_paper,
        sections=[
            Section(name="abstract", content="Abstract text."),
            Section(name="methods", content="Methods text."),
        ],
        claims=sample_claims,
    )


@pytest.fixture
def sample_profiles(sample_paper_profile: PaperProfile) -> list[PaperProfile]:
    """A list containing a single PaperProfile."""
    return [sample_paper_profile]


# -------------------------------------------------------------------
# Sample CodebaseContext fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_components() -> list[Component]:
    """A set of sample components for testing."""
    return [
        Component(
            name="attention_layer",
            kind="class",
            file_path="model/layers.py",
            line_start=10,
            line_end=50,
            signature="class AttentionLayer(nn.Module)",
            description="Implements multi-head attention mechanism for transformer models",
            module_name="model.layers",
        ),
        Component(
            name="train_model",
            kind="function",
            file_path="train.py",
            line_start=15,
            line_end=45,
            signature="def train_model(model, data, epochs)",
            description="Training loop using gradient descent optimization with Adam",
            module_name="train",
        ),
        Component(
            name="data_loader",
            kind="class",
            file_path="data/loader.py",
            line_start=5,
            line_end=30,
            signature="class DataLoader",
            description="Loads and preprocesses CSV data files for model training",
            module_name="data.loader",
        ),
    ]


@pytest.fixture
def sample_modules() -> list[ModuleSummary]:
    """A set of sample module summaries for testing."""
    return [
        ModuleSummary(
            module_name="model.layers",
            file_path="model/layers.py",
            summary="Implements neural network layer components including attention mechanisms and normalization layers",
            key_responsibilities=["attention computation", "layer normalization"],
        ),
        ModuleSummary(
            module_name="train",
            file_path="train.py",
            summary="Training pipeline for deep learning models including gradient-based optimization",
            key_responsibilities=["model training", "gradient computation"],
        ),
    ]


@pytest.fixture
def sample_codebase(
    sample_components: list[Component],
    sample_modules: list[ModuleSummary],
) -> CodebaseContext:
    """A CodebaseContext with sample components and modules."""
    return CodebaseContext(
        project_name="test-project",
        project_path="/tmp/test-project",
        language="python",
        components=sample_components,
        modules=sample_modules,
        total_files_scanned=3,
        total_components_found=3,
    )
