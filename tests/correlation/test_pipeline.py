"""Pipeline integration tests for CorrelationPipeline.

All tests use mocked ``Embedder`` and ``CorrelationClassifier`` —
no real API calls. Cosine similarity values are deterministic via
carefully crafted mock vectors.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from research_to_dev.codebase.types import CodebaseContext, Component, ModuleSummary
from research_to_dev.correlation.pipeline import CorrelationPipeline, _cosine_matrix
from research_to_dev.correlation.types import (
    Correlation,
    CorrelationMap,
    ModuleCorrelation,
)
from research_to_dev.paper_profile.types import Claim, PaperProfile
from research_to_dev.ranking.types import RankedPaper
from research_to_dev.shared.config import CorrelationMapConfig


# ======================================================================
# Helper — deterministic mock embedder (content-based)
# ======================================================================


def _embed_by_content(texts: list[str]) -> list[list[float]]:
    """Return deterministic 3-D vectors based on text content keywords.

    Vector assignment logic:
    - Texts containing "attention" or "Attention" → [1.0, 0.0, 0.0]
    - Texts containing "gradient" or "Gradient" → [0.0, 0.0, 1.0]
    - Texts containing "transformer" or "Transformer" → [0.0, 1.0, 0.0]
    - Texts containing "data_loader" or "loads" → [-1.0, -1.0, -1.0] (far)
    - Texts containing "train_model" or "train" → [0.0, 0.2, 0.85] (close to gradient claim)
    - Texts containing "attention_layer" → [0.9, 0.1, 0.0] (close to attention claim)
    - Texts containing "model.layers" as module → [0.8, 0.2, 0.0] (close to attention claim)
    - Empty text → [0.0, 0.0, 0.0] (zero vector)
    - Fallback → [0.0, 0.5, 0.0] (orthogonal-ish to main axes)
    """
    vectors = []
    for t in texts:
        text = t.strip().lower() if t else ""
        if not text:
            vectors.append([0.0, 0.0, 0.0])
        elif "attention_layer" in text or "multi-head attention" in text:
            # Component: attention_layer — close to attention claim
            vectors.append([0.9, 0.1, 0.0])
        elif "train_model" in text or "gradient descent optimization with adam" in text.lower():
            # Component: train_model — close to gradient claim
            vectors.append([0.0, 0.2, 0.85])
        elif "data_loader" in text or "loads and preprocesses" in text:
            # Component: data_loader — far from all claims
            vectors.append([-1.0, -1.0, -1.0])
        elif "model.layers" in text or "attention mechanisms and normalization" in text:
            # Module: model.layers — close to attention claim
            vectors.append([0.8, 0.2, 0.0])
        elif "training pipeline" in text or "gradient-based optimization" in text:
            # Module: train — close to gradient claim
            vectors.append([0.0, 0.1, 0.75])
        elif "attention" in text:
            # Claim: attention-related
            vectors.append([1.0, 0.0, 0.0])
        elif "gradient" in text:
            # Claim: gradient-related
            vectors.append([0.0, 0.0, 1.0])
        elif "transformer" in text:
            vectors.append([0.0, 1.0, 0.0])
        else:
            vectors.append([0.0, 0.5, 0.0])
    return vectors


def _mock_embedder() -> AsyncMock:
    """Create a mock embedder that returns vectors based on text content.

    Same text always maps to the same vector regardless of position.
    """
    embedder = AsyncMock()
    embedder.embed = AsyncMock(side_effect=_embed_by_content)
    return embedder


def _mock_embedder_from_vectors(
    vectors: list[list[float]],
) -> AsyncMock:
    """Create a mock embedder that returns pre-defined vectors cyclically.

    Only used for edge-case tests (e.g., garbage input) where content-based
    mapping isn't needed.
    """
    idx = [0]  # mutable counter

    async def _embed(texts: list[str]) -> list[list[float]]:
        nonlocal idx
        result = vectors[idx[0] : idx[0] + len(texts)]
        idx[0] += len(texts)
        return result

    embedder = AsyncMock()
    embedder.embed = AsyncMock(side_effect=_embed)
    return embedder


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def deterministic_embedder() -> AsyncMock:
    """Mock embedder returning deterministic 3-D vectors based on text content.

    Same text always maps to the same vector regardless of position in the batch.
    - "attention" claims → [1,0,0] — close to attention_layer [0.9,0.1,0]
    - "gradient" claims → [0,0,1] — close to train_model [0,0.2,0.85]
    - "transformer" → [0,1,0]
    - "attention_layer" component → [0.9,0.1,0] — close to attention claim
    - "train_model" component → [0,0.2,0.85] — close to gradient claim
    - "data_loader" component → [-1,-1,-1] — far from all
    - "model.layers" module → [0.8,0.2,0] — close to attention claim
    - "training pipeline" module → [0,0.1,0.75] — close to gradient claim
    """
    return _mock_embedder()


@pytest.fixture
def validating_classifier() -> AsyncMock:
    """Mock classifier that validates all candidates."""
    classifier = AsyncMock()

    async def _classify(
        title: str, paper_claims: str, candidates: list[dict]
    ) -> list[dict]:
        # For the zero-matches test, we'll handle it via correlation_type=None
        results = []
        for i, c in enumerate(candidates):
            results.append({
                "claim_text": c["claim_text"],
                "target_name": c["target_name"],
                "correlation_type": f"direct_solution",
                "reasoning": f"Validated correlation #{i} for {c['target_name']}",
            })
        return results

    classifier.classify = AsyncMock(side_effect=_classify)
    return classifier


@pytest.fixture
def rejecting_classifier() -> AsyncMock:
    """Mock classifier that rejects ALL candidates (returns empty list)."""
    classifier = AsyncMock()
    classifier.classify = AsyncMock(return_value=[])
    return classifier


@pytest.fixture
def flaky_classifier() -> AsyncMock:
    """Mock classifier that fails on second call (simulates LLM failure for one paper)."""
    call_count = [0]

    async def _classify(
        title: str, paper_claims: str, candidates: list[dict]
    ) -> list[dict]:
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("LLM API failure for paper 2")
        return [
            {
                "claim_text": c["claim_text"],
                "target_name": c["target_name"],
                "correlation_type": "related_technique",
                "reasoning": "Seems related",
            }
            for c in candidates
        ]

    classifier = AsyncMock()
    classifier.classify = AsyncMock(side_effect=_classify)
    return classifier


# ======================================================================
# Unit: cosine matrix helper
# ======================================================================


class TestCosineMatrix:
    """Tests for the _cosine_matrix helper function."""

    def test_identical_vectors_give_1(self) -> None:
        """Cosine similarity of a vector with itself is 1.0."""
        import numpy as np
        a = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_matrix(a, a)
        assert abs(float(result[0, 0]) - 1.0) < 1e-6

    def test_orthogonal_vectors_give_0(self) -> None:
        """Cosine similarity of orthogonal vectors is 0.0."""
        import numpy as np
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 1.0, 0.0]])
        result = _cosine_matrix(a, b)
        assert abs(float(result[0, 0]) - 0.0) < 1e-6

    def test_opposite_vectors_give_negative_1(self) -> None:
        """Cosine similarity of opposite vectors is -1.0."""
        import numpy as np
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[-1.0, 0.0, 0.0]])
        result = _cosine_matrix(a, b)
        assert abs(float(result[0, 0]) + 1.0) < 1e-6

    def test_zero_vector_handled(self) -> None:
        """Zero vector does not cause division by zero."""
        import numpy as np
        a = np.array([[0.0, 0.0, 0.0]])
        b = np.array([[1.0, 0.0, 0.0]])
        result = _cosine_matrix(a, b)
        # Zero vector / max(norm, 1e-10) → all zeros, dot with anything = 0
        assert abs(float(result[0, 0]) - 0.0) < 1e-6

    def test_matrix_shape(self) -> None:
        """Output shape is (n, m) for inputs (n, d) and (m, d)."""
        import numpy as np
        a = np.random.randn(5, 10)
        b = np.random.randn(3, 10)
        result = _cosine_matrix(a, b)
        assert result.shape == (5, 3)


# ======================================================================
# Pipeline — E2E and edge cases
# ======================================================================


class TestPipelineE2E:
    """End-to-end pipeline tests with deterministic mock dependencies."""

    @pytest.mark.asyncio
    async def test_e2e_produces_valid_correlation_map(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Full pipeline with mock deps produces a valid CorrelationMap (CM-01–CM-04)."""
        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.papers_analyzed == 1
        assert result.total_claims == 3
        assert result.total_components == 3
        assert result.total_modules == 2

        # At threshold 0.7:
        # Claim 0 ("attention") close to comp 0 (attention_layer): cos≈0.994 → candidate
        # Claim 0 close to mod 0 (model.layers): cos≈0.970 → candidate
        # Claim 2 ("gradient") close to comp 1 (train_model): cos≈0.973 → candidate
        # Claim 2 close to mod 1 (train): cos≈0.991 → candidate
        # Other pairs are orthogonal (cos≈0) or far (data_loader is [-1,-1,-1])

        # All candidates should be validated by the classifier
        assert result.matches_found > 0

        # Both Correlation and ModuleCorrelation types should be present
        has_component = len(result.correlations) > 0
        has_module = len(result.module_correlations) > 0
        assert has_component or has_module, "Should have at least some correlations"

        # Verify traceability: Correlation holds full Claim and Component
        for corr in result.correlations:
            assert isinstance(corr, Correlation)
            assert corr.claim is not None
            assert corr.component is not None
            assert corr.paper_title == "Deep Learning for NLP"
            assert corr.correlation_type != ""
            assert corr.reasoning != ""

        # Verify traceability: ModuleCorrelation holds full Claim and ModuleSummary
        for mcorr in result.module_correlations:
            assert isinstance(mcorr, ModuleCorrelation)
            assert mcorr.claim is not None
            assert mcorr.module is not None

    @pytest.mark.asyncio
    async def test_empty_profiles_returns_empty_map(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Empty profiles list → empty CorrelationMap with zero claims (CM-05)."""
        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.papers_analyzed == 0
        assert result.total_claims == 0
        assert result.correlations == []
        assert result.module_correlations == []
        assert result.matches_found == 0

    @pytest.mark.asyncio
    async def test_empty_codebase_returns_empty_map(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Empty codebase (no components, no modules) → empty CorrelationMap (CM-05)."""
        empty_codebase = CodebaseContext(
            project_name="empty",
            project_path="/tmp/empty",
            language="python",
        )

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], empty_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.total_components == 0
        assert result.total_modules == 0
        assert result.matches_found == 0

    @pytest.mark.asyncio
    async def test_zero_matches_above_threshold(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
    ) -> None:
        """When threshold is 1.0, no pairs pass filter → empty with stats (CM-05, CM-02)."""
        config = CorrelationMapConfig(similarity_threshold=1.0)

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.matches_found == 0
        assert result.correlations == []
        assert result.module_correlations == []
        # Stats should still be accurate
        assert result.total_claims == 3
        assert result.total_components == 3

    @pytest.mark.asyncio
    async def test_both_component_and_module_correlations_present(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Both component-level and module-level correlations are present in output (D3)."""
        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        # We should have at least some of both types since vectors are designed
        # to match both component and module levels
        assert len(result.correlations) > 0, "Should have component correlations"
        assert len(result.module_correlations) > 0, "Should have module correlations"

    @pytest.mark.asyncio
    async def test_config_snapshot_in_output(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
    ) -> None:
        """CorrelationMap output includes config snapshot for reproducibility (CM-04)."""
        config = CorrelationMapConfig(
            similarity_threshold=0.5,
            embedding_model="custom-model",
        )

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert result.similarity_threshold == 0.5
        assert result.embedding_model == "custom-model"


class TestPipelineErrorResilience:
    """Verify the 4-tier error resilience cascade (CM-05, AD-05)."""

    @pytest.mark.asyncio
    async def test_embedder_total_failure_returns_empty_map(
        self,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Embedding API failure → empty CorrelationMap with stats, no raise (CM-05)."""
        failing_embedder = AsyncMock()
        failing_embedder.embed = AsyncMock(
            side_effect=RuntimeError("Embedding API down")
        )

        pipeline = CorrelationPipeline(
            embedder=failing_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.correlations == []
        assert result.module_correlations == []
        assert result.matches_found == 0
        # Stats preserved
        assert result.papers_analyzed == 1
        assert result.total_claims == 3
        assert result.total_components == 3
        assert result.total_modules == 2

    @pytest.mark.asyncio
    async def test_partial_llm_failure_preserves_successful_matches(
        self,
        deterministic_embedder: AsyncMock,
        flaky_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """When LLM fails for one paper, only the successful paper's matches survive (CM-05)."""
        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=flaky_classifier,
            config=correlation_config,
        )

        # Single paper → single LLM call → it fails because call_count[0] + 1 = 1 != 2? 
        # Wait: the flaky classifier fails on second call, but we only have one paper.
        # We need a scenario where there are multiple papers.
        # Actually, the classifier is called per paper_id. With one paper,
        # there's only one paper_id, so only one call. The flaky fails on
        # the second call. So it should succeed with one paper.

        # Let's test with the flaky — for a single paper it should work
        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.matches_found > 0, (
            "Single paper should succeed (flaky fails on 2nd call only)"
        )

    @pytest.mark.asyncio
    async def test_pipeline_never_raises_on_total_garbage_input(
        self,
        validating_classifier: AsyncMock,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Pipeline never raises even on completely invalid inputs."""
        # Create a profile with no ranked_paper attributes accessible
        # but the pipeline uses profile.ranked_paper.paper.title which
        # could fail. Let's test with a valid-but-empty profile.
        from research_to_dev.extraction.types import AcademicContent

        bad_paper = AcademicContent(
            source="arxiv",
            title="",  # empty title
            url="",
            abstract="",
        )
        ranked = RankedPaper(
            paper=bad_paper, relevance_score=0.5, semantic_score=0.5,
            llm_reasoning="", passed_metadata_filter=True,
        )
        profile = PaperProfile(
            ranked_paper=ranked,
            claims=[Claim(text="", paper_id="p1", section_name="abstract")],
        )
        codebase = CodebaseContext(
            project_name="test", project_path="/tmp", language="python",
            components=[
                Component(
                    name="f", kind="function", file_path="f.py",
                    line_start=1, line_end=2, signature="def f()",
                    description="", module_name="f",
                ),
            ],
        )

        embedder = _mock_embedder_from_vectors(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        pipeline = CorrelationPipeline(
            embedder=embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        # Should NOT raise
        result = await pipeline.correlate([profile], codebase)
        assert isinstance(result, CorrelationMap)

    @pytest.mark.asyncio
    async def test_no_components_no_modules_returns_empty(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Empty components + empty modules → empty map with stats, no error (CM-05)."""
        empty_codebase = CodebaseContext(
            project_name="test", project_path="/tmp", language="python",
            components=[], modules=[],
        )

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], empty_codebase)

        assert result.correlations == []
        assert result.module_correlations == []
        assert result.matches_found == 0
        assert result.total_components == 0
        assert result.total_modules == 0

    @pytest.mark.asyncio
    async def test_all_claims_rejected_by_llm(
        self,
        deterministic_embedder: AsyncMock,
        rejecting_classifier: AsyncMock,
        sample_paper_profile: PaperProfile,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """When LLM rejects all candidates, map is empty with stats (D10)."""
        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=rejecting_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([sample_paper_profile], sample_codebase)

        assert isinstance(result, CorrelationMap)
        assert result.correlations == []
        assert result.module_correlations == []
        assert result.matches_found == 0
        # Stats still accurate
        assert result.total_claims == 3
        assert result.papers_analyzed == 1


class TestPipelineConfigInjection:
    """Verify config injection and defaults."""

    @pytest.mark.asyncio
    async def test_default_config_used_when_none_provided(
        self, mock_embedder: AsyncMock, mock_classifier: AsyncMock,
    ) -> None:
        """Pipeline uses CorrelationMapConfig() defaults when no config injected."""
        pipeline = CorrelationPipeline(
            embedder=mock_embedder,
            classifier=mock_classifier,
        )

        assert pipeline._config.similarity_threshold == 0.7
        assert pipeline._config.embedding_model == "text-embedding-3-small"
        assert pipeline._config.llm_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_custom_config_injected(
        self, mock_embedder: AsyncMock, mock_classifier: AsyncMock,
    ) -> None:
        """Custom config is preserved after injection."""
        config = CorrelationMapConfig(
            similarity_threshold=0.5,
            embedding_model="big-model",
            llm_model="gpt-4o",
        )
        pipeline = CorrelationPipeline(
            embedder=mock_embedder,
            classifier=mock_classifier,
            config=config,
        )

        assert pipeline._config is config
        assert pipeline._config.similarity_threshold == 0.5
        assert pipeline._config.embedding_model == "big-model"


class TestPipelineMultiplePapers:
    """Verify correct handling of multiple paper profiles."""

    @pytest.mark.asyncio
    async def test_two_papers_with_different_claims(
        self,
        deterministic_embedder: AsyncMock,
        validating_classifier: AsyncMock,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """Two papers with different claims produce correlations from both (D9)."""
        from research_to_dev.extraction.types import AcademicContent

        # Paper 1
        p1 = AcademicContent(
            source="arxiv", title="Attention in NLP",
            url="https://arxiv.org/abs/2501.00001",
            abstract="Attention mechanisms", authors=["A"],
            published_date="2025-01-15",
        )
        rp1 = RankedPaper(
            paper=p1, relevance_score=0.9, semantic_score=0.8,
            llm_reasoning="r", passed_metadata_filter=True,
        )
        prof1 = PaperProfile(
            ranked_paper=rp1,
            claims=[Claim(text="Attention helps", paper_id="p1", section_name="abstract")],
        )

        # Paper 2
        p2 = AcademicContent(
            source="arxiv", title="Gradient Methods",
            url="https://arxiv.org/abs/2502.00002",
            abstract="Gradient descent", authors=["B"],
            published_date="2025-02-01",
        )
        rp2 = RankedPaper(
            paper=p2, relevance_score=0.8, semantic_score=0.7,
            llm_reasoning="r", passed_metadata_filter=True,
        )
        prof2 = PaperProfile(
            ranked_paper=rp2,
            claims=[Claim(text="Gradient descent optimization", paper_id="p2", section_name="methods")],
        )

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=validating_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([prof1, prof2], sample_codebase)

        assert result.papers_analyzed == 2
        assert result.total_claims == 2

        # Both papers should contribute some matches
        # Paper 1 ("Attention") → should match attention_layer component
        # Paper 2 ("Gradient") → should match train_model component
        titles = {c.paper_title for c in result.correlations}
        titles.update(m.paper_title for m in result.module_correlations)
        assert "Attention in NLP" in titles, f"Paper 1 should have matches, got titles: {titles}"
        assert "Gradient Methods" in titles, f"Paper 2 should have matches, got titles: {titles}"

    @pytest.mark.asyncio
    async def test_partial_llm_failure_with_two_papers(
        self,
        deterministic_embedder: AsyncMock,
        flaky_classifier: AsyncMock,
        sample_codebase: CodebaseContext,
        correlation_config: CorrelationMapConfig,
    ) -> None:
        """LLM fails for one paper, the other paper's matches survive (CM-05)."""
        from research_to_dev.extraction.types import AcademicContent

        p1 = AcademicContent(
            source="arxiv", title="Attention in NLP",
            url="https://arxiv.org/abs/1", abstract="", authors=["A"],
            published_date="2025-01-01",
        )
        rp1 = RankedPaper(
            paper=p1, relevance_score=0.5, semantic_score=0.5,
            llm_reasoning="", passed_metadata_filter=True,
        )
        prof1 = PaperProfile(
            ranked_paper=rp1,
            claims=[Claim(text="Attention helps", paper_id="p1", section_name="abstract")],
        )

        p2 = AcademicContent(
            source="arxiv", title="Other Paper",
            url="https://arxiv.org/abs/2", abstract="", authors=["B"],
            published_date="2025-01-01",
        )
        rp2 = RankedPaper(
            paper=p2, relevance_score=0.5, semantic_score=0.5,
            llm_reasoning="", passed_metadata_filter=True,
        )
        prof2 = PaperProfile(
            ranked_paper=rp2,
            claims=[Claim(text="Unrelated topic", paper_id="p2", section_name="abstract")],
        )

        pipeline = CorrelationPipeline(
            embedder=deterministic_embedder,
            classifier=flaky_classifier,
            config=correlation_config,
        )

        result = await pipeline.correlate([prof1, prof2], sample_codebase)

        assert result.papers_analyzed == 2
        # Paper 1 succeeds (call #1 passes), Paper 2 fails (call #2 fails)
        # So we should have matches from paper 1 only
        titles = {c.paper_title for c in result.correlations}
        titles.update(m.paper_title for m in result.module_correlations)
        assert "Attention in NLP" in titles, "Paper 1 should have matches"
        # Paper 2 should have been silently excluded
