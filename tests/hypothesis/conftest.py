"""Pytest configuration and shared fixtures for hypothesis tests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pytest

from research_to_dev.codebase.types import Component, ModuleSummary
from research_to_dev.correlation.types import Correlation, CorrelationMap, ModuleCorrelation
from research_to_dev.extraction.types import AcademicContent
from research_to_dev.paper_profile.types import Claim
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import HypothesisConfig


# -------------------------------------------------------------------
# HypothesisConfig fixture
# -------------------------------------------------------------------


@pytest.fixture
def hypothesis_config() -> HypothesisConfig:
    """Default HypothesisConfig with test-friendly values."""
    return HypothesisConfig(
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        max_reflection_rounds=3,
        dedup_threshold=0.85,
        convergence_score_delta=0.5,
        max_hypotheses_per_paper=5,
        max_total_hypotheses=30,
        relevance_gate=7,
    )


@pytest.fixture
def lenient_config() -> HypothesisConfig:
    """HypothesisConfig with lower thresholds for easier testing."""
    return HypothesisConfig(
        relevance_gate=1,
        dedup_threshold=0.99,
        max_hypotheses_per_paper=10,
        max_total_hypotheses=100,
        convergence_score_delta=0.01,
    )


# -------------------------------------------------------------------
# Mock generator — returns predefined hypotheses
# -------------------------------------------------------------------


@pytest.fixture
def mock_generator() -> AsyncMock:
    """AsyncMock that satisfies the HypothesisGenerator Protocol.

    Returns predefined hypotheses for generate() and empty
    new_hypotheses for critique_and_expand() (no new additions).
    """
    gen = AsyncMock()

    async def _generate(
        title: str,
        paper_claims: str,
        correlations: list[dict],
        query: str,
        round_num: int = 0,
        general_content: str | None = None,
    ) -> list[dict]:
        return [
            {
                "title": f"Hypothesis about {title[:20]}",
                "description": f"A test hypothesis generated from correlations for {title}",
                "approach": "Implement the proposed technique",
                "supporting_papers": [title],
                "target_metric": "accuracy",
                "expected_improvement": "10-15% improvement",
                "code_changes": "modify core modules",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [
                    c["id"] for c in correlations[:2]
                ],
                "success_criteria": "metric improves by >5%",
            },
            {
                "title": f"Second hypothesis from {title[:20]}",
                "description": "Another perspective on the same correlations",
                "approach": "Try alternative implementation",
                "supporting_papers": [title],
                "target_metric": "latency",
                "expected_improvement": "20% reduction",
                "code_changes": "refactor pipeline",
                "scores": {"relevance": 7, "feasibility": 9, "evidence": 5},
                "correlations": [
                    c["id"] for c in correlations
                ],
                "success_criteria": "latency p99 < 100ms",
            },
        ]

    gen.generate = AsyncMock(side_effect=_generate)

    async def _critique(
        title: str,
        paper_claims: str,
        correlations: list[dict],
        existing_hypotheses: list[dict],
        query: str,
        round_num: int,
        general_content: str | None = None,
    ) -> dict:
        return {"critiques": [], "new_hypotheses": []}

    gen.critique_and_expand = AsyncMock(side_effect=_critique)
    return gen


# -------------------------------------------------------------------
# Mock embedder — deterministic content-based vectors
# -------------------------------------------------------------------


@pytest.fixture
def mock_embedder() -> AsyncMock:
    """AsyncMock that satisfies the Embedder Protocol.

    Returns deterministic vectors based on text content hash so same
    text → same vector (controls dedup behavior in tests).
    """
    embedder = AsyncMock()

    async def _embed(texts: list[str]) -> list[list[float]]:
        vectors = []
        for t in texts:
            text = t.strip()
            if not text:
                vectors.append([0.0] * 10)
            else:
                seed = hash(text) % 1000
                rng = np.random.default_rng(abs(seed) + 42)
                vec = rng.uniform(-1, 1, size=10).tolist()
                vectors.append(vec)
        return vectors

    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


# -------------------------------------------------------------------
# Sample Claim fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_claims() -> list[Claim]:
    """Claims from two different papers."""
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
            text="Gradient descent convergence depends on learning rate",
            paper_id="paper-2",
            section_name="methods",
        ),
    ]


# -------------------------------------------------------------------
# Sample components and modules
# -------------------------------------------------------------------


@pytest.fixture
def sample_components() -> list[Component]:
    """Sample components for building correlations."""
    return [
        Component(
            name="attention_layer",
            kind="class",
            file_path="model/layers.py",
            line_start=10,
            line_end=50,
            signature="class AttentionLayer",
            description="Implements multi-head attention mechanism",
            module_name="model.layers",
        ),
        Component(
            name="train_model",
            kind="function",
            file_path="train.py",
            line_start=15,
            line_end=45,
            signature="def train_model(data)",
            description="Training loop using gradient descent",
            module_name="train",
        ),
    ]


@pytest.fixture
def sample_modules() -> list[ModuleSummary]:
    """Sample modules for building correlations."""
    return [
        ModuleSummary(
            module_name="model.layers",
            file_path="model/layers.py",
            summary="Attention-based neural network layers",
            key_responsibilities=["attention", "normalization"],
        ),
        ModuleSummary(
            module_name="train",
            file_path="train.py",
            summary="Training pipeline with gradient optimization",
            key_responsibilities=["training", "optimization"],
        ),
    ]


# -------------------------------------------------------------------
# Sample correlation fixtures
# -------------------------------------------------------------------


@pytest.fixture
def sample_correlations(sample_claims: list[Claim], sample_components: list[Component]) -> list[Correlation]:
    """Component-level correlations from two papers."""
    return [
        Correlation(
            claim=sample_claims[0],
            paper_title="Deep Learning for NLP",
            component=sample_components[0],
            module_name="model.layers",
            similarity_score=0.85,
            correlation_type="direct_solution",
            reasoning="Directly implements attention mechanism",
        ),
        Correlation(
            claim=sample_claims[1],
            paper_title="Deep Learning for NLP",
            component=sample_components[0],
            module_name="model.layers",
            similarity_score=0.72,
            correlation_type="related_technique",
            reasoning="Transformer uses attention layers",
        ),
        Correlation(
            claim=sample_claims[2],
            paper_title="Optimization Techniques",
            component=sample_components[1],
            module_name="train",
            similarity_score=0.88,
            correlation_type="direct_solution",
            reasoning="Implements gradient descent",
        ),
    ]


@pytest.fixture
def sample_module_correlations(sample_claims: list[Claim], sample_modules: list[ModuleSummary]) -> list[ModuleCorrelation]:
    """Module-level correlations."""
    return [
        ModuleCorrelation(
            claim=sample_claims[0],
            paper_title="Deep Learning for NLP",
            module=sample_modules[0],
            similarity_score=0.80,
            correlation_type="direct_solution",
            reasoning="Attention layer module",
        ),
        ModuleCorrelation(
            claim=sample_claims[2],
            paper_title="Optimization Techniques",
            module=sample_modules[1],
            similarity_score=0.82,
            correlation_type="direct_solution",
            reasoning="Training module",
        ),
    ]


@pytest.fixture
def sample_correlation_map(
    sample_correlations: list[Correlation],
    sample_module_correlations: list[ModuleCorrelation],
) -> CorrelationMap:
    """Full CorrelationMap with both correlation types."""
    return CorrelationMap(
        correlations=sample_correlations,
        module_correlations=sample_module_correlations,
        papers_analyzed=2,
        total_claims=3,
        total_components=2,
        total_modules=2,
        matches_found=5,
        similarity_threshold=0.7,
        embedding_model="text-embedding-3-small",
    )


# -------------------------------------------------------------------
# Single-paper CorrelationMap (for edge case tests)
# -------------------------------------------------------------------


@pytest.fixture
def single_paper_correlation_map(sample_correlations: list[Correlation]) -> CorrelationMap:
    """CorrelationMap with correlations from a single paper."""
    # Only keep paper-1 correlations
    paper1_corrs = [
        c for c in sample_correlations if c.claim.paper_id == "paper-1"
    ]
    return CorrelationMap(
        correlations=paper1_corrs,
        module_correlations=[],
        papers_analyzed=1,
        total_claims=2,
        total_components=2,
        total_modules=0,
        matches_found=len(paper1_corrs),
    )
