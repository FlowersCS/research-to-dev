"""Pipeline integration and unit tests for HypothesisPipeline.

All tests use mocked ``HypothesisGenerator`` and ``Embedder`` —
no real API calls. Covers E2E, reflection loop, deduplication,
convergence detection, grounding, and error resilience (HG-06).
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import numpy as np
import pytest

from research_to_dev.codebase.types import Component, ModuleSummary
from research_to_dev.correlation.types import Correlation, CorrelationMap, ModuleCorrelation
from research_to_dev.extraction.types import GeneralContent
from research_to_dev.hypothesis.pipeline import HypothesisPipeline
from research_to_dev.hypothesis.types import (
    Hypothesis,
    HypothesisError,
    HypothesisResult,
    HypothesisScores,
)
from research_to_dev.paper_profile.types import Claim
from research_to_dev.shared.config import HypothesisConfig


# ======================================================================
# Helpers
# ======================================================================


def _make_claim(text: str, paper_id: str) -> Claim:
    return Claim(text=text, paper_id=paper_id, section_name="abstract")


def _make_component(name: str, desc: str, module: str) -> Component:
    return Component(
        name=name, kind="class", file_path="f.py",
        line_start=1, line_end=10, signature=f"class {name}",
        description=desc, module_name=module,
    )


def _make_module(name: str, summary: str) -> ModuleSummary:
    return ModuleSummary(
        module_name=name, file_path="f.py", summary=summary,
        key_responsibilities=["test"],
    )


def _build_correlation_map(
    claim: Claim, component: Component, paper_title: str,
    mod: ModuleSummary | None = None,
    include_module: bool = True,
) -> CorrelationMap:
    """Build a minimal CorrelationMap with one claim↔component correlation."""
    corr = Correlation(
        claim=claim, paper_title=paper_title,
        component=component, module_name=component.module_name,
        similarity_score=0.85, correlation_type="direct_solution",
        reasoning="Test",
    )
    mod_corrs: list[ModuleCorrelation] = []
    if include_module and mod is not None:
        mod_corrs.append(
            ModuleCorrelation(
                claim=claim, paper_title=paper_title,
                module=mod, similarity_score=0.80,
                correlation_type="direct_solution",
                reasoning="Test",
            )
        )
    return CorrelationMap(
        correlations=[corr], module_correlations=mod_corrs,
        papers_analyzed=1, total_claims=1,
        total_components=1, total_modules=1 if mod_corrs else 0,
        matches_found=1 + len(mod_corrs),
    )


def _make_mock_gen_with_hypothesis(
    hypothesis_title: str = "Test Hypothesis",
    relevance: int = 8,
    score_dict: dict | None = None,
) -> AsyncMock:
    """Create a mock generator that returns a single hypothesis.

    The returned hypothesis will reference ALL correlation IDs from the
    input so grounding check passes.
    """
    gen = AsyncMock()

    async def _generate(
        title: str, paper_claims: str,
        correlations: list[dict], query: str, round_num: int = 0,
        general_content: str | None = None,
    ) -> list[dict]:
        s = score_dict or {"relevance": relevance, "feasibility": 7, "evidence": 6}
        return [{
            "title": hypothesis_title,
            "description": f"Description for {hypothesis_title}",
            "approach": "Implementation approach",
            "supporting_papers": [title],
            "target_metric": "accuracy",
            "expected_improvement": "10%",
            "code_changes": "code.py",
            "scores": s,
            "correlations": [c["id"] for c in correlations],
            "success_criteria": "metric improves",
        }]

    gen.generate = AsyncMock(side_effect=_generate)

    async def _critique(*args, **kwargs) -> dict:
        return {"critiques": [], "new_hypotheses": []}

    gen.critique_and_expand = AsyncMock(side_effect=_critique)
    return gen


# ======================================================================
# E2E Pipeline Tests
# ======================================================================


class TestHypothesisPipelineE2E:
    """End-to-end pipeline tests using mock generator and embedder."""

    @pytest.mark.asyncio
    async def test_e2e_basic_flow(
        self, sample_correlation_map, mock_generator,
        mock_embedder, hypothesis_config,
    ) -> None:
        """Full pipeline produces HypothesisResult with correct metadata."""
        pipeline = HypothesisPipeline(
            generator=mock_generator,
            embedder=mock_embedder,
            config=hypothesis_config,
        )

        result = await pipeline.run(
            correlation_map=sample_correlation_map,
            query="How can I improve my model?",
        )

        assert isinstance(result, HypothesisResult)
        assert len(result.hypotheses) > 0
        assert result.total_input_papers == 2
        assert result.total_correlations == 5  # 3 component + 2 module

        # Each hypothesis should be grounded
        for h in result.hypotheses:
            assert len(h.correlations) > 0
            assert h.scores.relevance >= 7  # relevance gate

    @pytest.mark.asyncio
    async def test_e2e_single_paper(
        self, single_paper_correlation_map, mock_generator,
        mock_embedder, hypothesis_config,
    ) -> None:
        """Pipeline with single paper works correctly."""
        pipeline = HypothesisPipeline(
            generator=mock_generator,
            embedder=mock_embedder,
            config=hypothesis_config,
        )

        result = await pipeline.run(
            correlation_map=single_paper_correlation_map,
            query="query",
        )

        assert result.total_input_papers == 1
        assert len(result.hypotheses) > 0

    @pytest.mark.asyncio
    async def test_e2e_empty_correlation_map(
        self, mock_generator, mock_embedder, hypothesis_config,
    ) -> None:
        """Empty CorrelationMap produces empty HypothesisResult."""
        pipeline = HypothesisPipeline(
            generator=mock_generator,
            embedder=mock_embedder,
            config=hypothesis_config,
        )

        empty_map = CorrelationMap()
        result = await pipeline.run(
            correlation_map=empty_map, query="query",
        )

        assert isinstance(result, HypothesisResult)
        assert result.hypotheses == []
        assert result.total_input_papers == 0
        assert result.total_correlations == 0

    @pytest.mark.asyncio
    async def test_e2e_hypotheses_have_correct_ids(
        self, sample_correlation_map, mock_generator,
        mock_embedder, hypothesis_config,
    ) -> None:
        """Generated hypotheses have deterministic SHA-256 IDs (AD-02)."""
        pipeline = HypothesisPipeline(
            generator=mock_generator,
            embedder=mock_embedder,
            config=hypothesis_config,
        )

        result = await pipeline.run(
            correlation_map=sample_correlation_map,
            query="query",
        )

        for h in result.hypotheses:
            expected_id = hashlib.sha256(
                f"{h.title}{h.description}".encode()
            ).hexdigest()[:12]
            assert h.id == expected_id
            assert len(h.id) == 12

    @pytest.mark.asyncio
    async def test_e2e_relevance_gate_filters_low_scores(
        self, tmp_path, mock_embedder, hypothesis_config,
    ) -> None:
        """Hypotheses with relevance < 7 are filtered out (relevance gate)."""
        claim = _make_claim("Test claim", "paper-1")
        comp = _make_component("test_func", "test desc", "test.module")
        mod = _make_module("test.module", "test summary")
        cmap = _build_correlation_map(
            claim, comp, "Test Paper", mod, include_module=True,
        )

        # Generator that produces one high-relevance and one low-relevance
        gen = AsyncMock()

        async def _generate(
            title: str, paper_claims: str,
            correlations: list[dict], query: str, round_num: int = 0,
            general_content: str | None = None,
        ) -> list[dict]:
            return [
                {
                    "title": "High relevance", "description": "Good hypothesis",
                    "approach": "", "supporting_papers": [],
                    "target_metric": "", "expected_improvement": "",
                    "code_changes": "",
                    "scores": {"relevance": 9, "feasibility": 8, "evidence": 7},
                    "correlations": [c["id"] for c in correlations],
                    "success_criteria": "",
                },
                {
                    "title": "Low relevance", "description": "Weak hypothesis",
                    "approach": "", "supporting_papers": [],
                    "target_metric": "", "expected_improvement": "",
                    "code_changes": "",
                    "scores": {"relevance": 3, "feasibility": 8, "evidence": 7},
                    "correlations": [c["id"] for c in correlations],
                    "success_criteria": "",
                },
            ]

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=hypothesis_config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        titles = {h.title for h in result.hypotheses}
        assert "High relevance" in titles
        assert "Low relevance" not in titles


# ======================================================================
# Deduplication Tests
# ======================================================================


class TestDeduplication:
    """Tests for _deduplicate() with cosine similarity threshold."""

    @pytest.mark.asyncio
    async def test_identical_hypotheses_merged(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Two identical hypotheses (same title+desc → high cosine sim)
        are merged, keeping the higher composite."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=mock_embedder,
            config=hypothesis_config,
        )

        # Same text → same embedding vector → cosine = 1.0 > threshold
        h1 = Hypothesis(
            id="id1", title="Same Title", description="Same description",
            scores=HypothesisScores(relevance=8, feasibility=8, evidence=8),
            composite=8.0, correlations=["cid1"],
        )
        h2 = Hypothesis(
            id="id2", title="Same Title", description="Same description",
            scores=HypothesisScores(relevance=5, feasibility=5, evidence=5),
            composite=5.0, correlations=["cid1"],
        )

        result = await pipeline._deduplicate([h1, h2])

        assert len(result) == 1
        assert result[0].id == "id1"  # higher composite kept
        assert result[0].composite == 8.0

    @pytest.mark.asyncio
    async def test_different_hypotheses_not_merged(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Different hypotheses (very different text) are not merged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=mock_embedder,
            config=hypothesis_config,
        )

        h1 = Hypothesis(
            id="id1",
            title="Completely different hypothesis A",
            description="This one is about attention mechanisms in NLP models",
            scores=HypothesisScores(relevance=8, feasibility=7, evidence=6),
            composite=7.0, correlations=["cid1"],
        )
        h2 = Hypothesis(
            id="id2",
            title="Totally unrelated hypothesis B",
            description="This one discusses gradient optimization strategies",
            scores=HypothesisScores(relevance=8, feasibility=7, evidence=6),
            composite=7.0, correlations=["cid2"],
        )

        result = await pipeline._deduplicate([h1, h2])

        # With content-based mock embedder, very different texts
        # should produce different vectors → low cosine sim
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_single_hypothesis_unchanged(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Single hypothesis passes through dedup unchanged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=mock_embedder,
            config=hypothesis_config,
        )

        h = Hypothesis(
            id="id1", title="Title", description="Description",
            scores=HypothesisScores(relevance=8, feasibility=7, evidence=6),
            composite=7.0, correlations=["cid1"],
        )

        result = await pipeline._deduplicate([h])
        assert result == [h]

    @pytest.mark.asyncio
    async def test_empty_dedup_returns_empty(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Empty list returns empty list."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=mock_embedder,
            config=hypothesis_config,
        )

        result = await pipeline._deduplicate([])
        assert result == []

    @pytest.mark.asyncio
    async def test_dedup_embedder_failure_returns_all(
        self, hypothesis_config,
    ) -> None:
        """T3: When embedder fails, all hypotheses are returned (skip dedup)."""
        embedder = AsyncMock()
        embedder.embed.side_effect = RuntimeError("API error")

        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=embedder,
            config=hypothesis_config,
        )

        h1 = Hypothesis(
            id="id1", title="T1", description="D1",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["c1"],
        )
        h2 = Hypothesis(
            id="id2", title="T2", description="D2",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["c2"],
        )

        result = await pipeline._deduplicate([h1, h2])
        assert len(result) == 2  # Both kept — dedup skipped
        assert {h.id for h in result} == {"id1", "id2"}


# ======================================================================
# Convergence Detection Tests
# ======================================================================


class TestConvergenceDetection:
    """Tests for _detect_convergence() (AD-05)."""

    def test_new_titles_not_converged(self, hypothesis_config) -> None:
        """New titles in 'after' set → not converged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        before = [
            Hypothesis(id="a", title="Old", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
        ]
        after = [
            Hypothesis(id="a", title="Old", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
            Hypothesis(id="b", title="New!", description="D2",
                       scores=HypothesisScores(7, 7, 7), composite=7.0),
        ]

        assert pipeline._detect_convergence(before, after) is False

    def test_title_set_same_but_score_delta_too_high(
        self, hypothesis_config,
    ) -> None:
        """Same titles but large composite score change → not converged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        before = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
        ]
        after = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(10, 10, 10), composite=10.0),
        ]

        # Delta = |10 - 7| = 3.0 >= convergence_score_delta (0.5)
        assert pipeline._detect_convergence(before, after) is False

    def test_stable_titles_and_scores_is_converged(
        self, hypothesis_config,
    ) -> None:
        """Same titles and small score delta → converged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        before = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
        ]
        after = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.1),
        ]

        # Delta = |7.1 - 7.0| = 0.1 < 0.5
        assert pipeline._detect_convergence(before, after) is True

    def test_small_delta_converged(self, hypothesis_config) -> None:
        """Score delta just under threshold → converged."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        before = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
        ]
        after = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.4),
        ]

        # Delta = 0.4 < 0.5 → converged
        assert pipeline._detect_convergence(before, after) is True

    def test_custom_epsilon(self) -> None:
        """Convergence uses config.convergence_score_delta."""
        strict_config = HypothesisConfig(convergence_score_delta=0.01)
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=strict_config,
        )

        before = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.0),
        ]
        after = [
            Hypothesis(id="a", title="T1", description="D",
                       scores=HypothesisScores(8, 7, 6), composite=7.1),
        ]

        # Delta = 0.1 >= 0.01 → NOT converged with strict config
        assert pipeline._detect_convergence(before, after) is False

    def test_empty_sets_converged(self, hypothesis_config) -> None:
        """Empty before and after → converged (degenerate case)."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        assert pipeline._detect_convergence([], []) is True


# ======================================================================
# Grounding Tests
# ======================================================================


class TestGrounding:
    """Tests for _check_grounding() (HG-05)."""

    def test_grounded_kept(self, hypothesis_config) -> None:
        """Hypothesis with valid correlation ID is kept."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        valid_ids = {"a1b2c3d4e5f6", "b2c3d4e5f6a7"}

        h = Hypothesis(
            id="id1", title="T", description="D",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["a1b2c3d4e5f6"],
        )

        result = pipeline._check_grounding([h], valid_ids)
        assert len(result) == 1
        assert result[0] is h

    def test_ungrounded_discarded(self, hypothesis_config) -> None:
        """Hypothesis without valid correlation IDs is discarded."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        valid_ids = {"a1b2c3d4e5f6"}

        h1 = Hypothesis(
            id="id1", title="Grounded", description="D",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["a1b2c3d4e5f6"],
        )
        h2 = Hypothesis(
            id="id2", title="Hallucinated", description="D",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=[],  # no valid IDs
        )

        result = pipeline._check_grounding([h1, h2], valid_ids)
        assert len(result) == 1
        assert result[0].title == "Grounded"

    def test_all_hallucinated_returns_empty(self, hypothesis_config) -> None:
        """All ungrounded → empty list."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        valid_ids = {"real_id_only"}

        h = Hypothesis(
            id="id1", title="T", description="D",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["fake_id"],
        )

        result = pipeline._check_grounding([h], valid_ids)
        assert result == []

    def test_partial_valid_correlation_ids(self, hypothesis_config) -> None:
        """Hypothesis with at least one valid ID among many is kept."""
        pipeline = HypothesisPipeline(
            generator=AsyncMock(), embedder=AsyncMock(),
            config=hypothesis_config,
        )

        valid_ids = {"real1"}

        h = Hypothesis(
            id="id1", title="T", description="D",
            scores=HypothesisScores(8, 7, 6), composite=7.0,
            correlations=["fake1", "real1", "fake2"],
        )

        result = pipeline._check_grounding([h], valid_ids)
        assert len(result) == 1


# ======================================================================
# Error Resilience Tests (HG-06)
# ======================================================================


class TestErrorResilience:
    """Tests for error resilience tiers T1-T4 (AD-07)."""

    @pytest.mark.asyncio
    async def test_t1_generator_failure_returns_partial(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """T1: When one paper fails, others still produce hypotheses."""
        claim1 = _make_claim("Claim 1", "paper-1")
        claim2 = _make_claim("Claim 2", "paper-2")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")

        cmap = CorrelationMap(
            correlations=[
                Correlation(
                    claim=claim1, paper_title="Paper 1",
                    component=comp, module_name="mod",
                    similarity_score=0.85,
                    correlation_type="direct_solution",
                    reasoning="OK",
                ),
                Correlation(
                    claim=claim2, paper_title="Paper 2",
                    component=comp, module_name="mod",
                    similarity_score=0.80,
                    correlation_type="direct_solution",
                    reasoning="OK",
                ),
            ],
            module_correlations=[
                ModuleCorrelation(
                    claim=claim1, paper_title="Paper 1",
                    module=mod, similarity_score=0.8,
                    correlation_type="direct_solution",
                    reasoning="OK",
                ),
            ],
            papers_analyzed=2, total_claims=2,
            total_components=1, total_modules=1, matches_found=3,
        )

        gen = AsyncMock()

        call_count = [0]

        async def _generate(
            title: str, paper_claims: str,
            correlations: list[dict], query: str, round_num: int = 0,
            general_content: str | None = None,
        ) -> list[dict]:
            call_count[0] += 1
            if "Paper 1" in title:
                raise RuntimeError("Paper 1 failed")
            return [{
                "title": f"From {title}",
                "description": "desc",
                "approach": "", "supporting_papers": [],
                "target_metric": "", "expected_improvement": "",
                "code_changes": "",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [c["id"] for c in correlations],
                "success_criteria": "",
            }]

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=hypothesis_config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        # Paper 1 failed, Paper 2 succeeded
        assert len(result.hypotheses) >= 1
        assert len(result.errors) >= 1
        assert any("Paper 1" in e.message for e in result.errors)

    @pytest.mark.asyncio
    async def test_t2_reflection_failure_still_returns_hypotheses(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """T2: When reflection round fails, prior hypotheses are returned."""
        claim = _make_claim("Claim", "paper-1")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")
        cmap = _build_correlation_map(claim, comp, "Test Paper", mod)

        gen = AsyncMock()

        async def _generate(*args, **kwargs) -> list[dict]:
            return [{
                "title": "Initial",
                "description": "Initial hypothesis",
                "approach": "", "supporting_papers": [],
                "target_metric": "", "expected_improvement": "",
                "code_changes": "",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [
                    c["id"] for c in kwargs.get("correlations", [])
                    if isinstance(c, dict)
                ],
                "success_criteria": "",
            }]

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand.side_effect = RuntimeError(
            "Reflection failed",
        )

        config_with_reflection = HypothesisConfig(
            max_reflection_rounds=1, relevance_gate=7,
        )
        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder,
            config=config_with_reflection,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        assert len(result.hypotheses) >= 1
        assert len(result.errors) >= 1

    @pytest.mark.asyncio
    async def test_t3_embedder_failure_continues(
        self, mock_generator, hypothesis_config,
    ) -> None:
        """T3: When embedder fails during dedup, pipeline continues."""
        embedder = AsyncMock()
        embedder.embed.side_effect = RuntimeError("Embedder down")

        claim = _make_claim("Claim", "paper-1")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")
        cmap = _build_correlation_map(claim, comp, "Test", mod)

        pipeline = HypothesisPipeline(
            generator=mock_generator, embedder=embedder,
            config=hypothesis_config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        # Should still produce hypotheses — dedup just skips
        assert isinstance(result, HypothesisResult)
        assert len(result.hypotheses) >= 1

    @pytest.mark.asyncio
    async def test_t4_malformed_json_returns_partial(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """T4: Malformed LLM JSON skips that paper, others continue."""
        claim1 = _make_claim("Claim 1", "paper-1")
        claim2 = _make_claim("Claim 2", "paper-2")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")

        cmap = CorrelationMap(
            correlations=[
                Correlation(
                    claim=claim1, paper_title="Paper 1",
                    component=comp, module_name="mod",
                    similarity_score=0.85, correlation_type="direct_solution",
                    reasoning="OK",
                ),
                Correlation(
                    claim=claim2, paper_title="Paper 2",
                    component=comp, module_name="mod",
                    similarity_score=0.80, correlation_type="direct_solution",
                    reasoning="OK",
                ),
            ],
            module_correlations=[],
            papers_analyzed=2, total_claims=2,
            total_components=1, total_modules=1, matches_found=2,
        )

        gen = AsyncMock()

        async def _generate(
            title: str, paper_claims: str,
            correlations: list[dict], query: str, round_num: int = 0,
            general_content: str | None = None,
        ) -> list[dict]:
            if "Paper 1" in title:
                # Return malformed entry
                return [{"not_a_title": True}]
            return [{
                "title": "Good",
                "description": "Valid hypothesis",
                "approach": "", "supporting_papers": [],
                "target_metric": "", "expected_improvement": "",
                "code_changes": "",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [c["id"] for c in correlations],
                "success_criteria": "",
            }]

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=hypothesis_config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        # Paper 2 hypothesis should be present
        titles = {h.title for h in result.hypotheses}
        assert "Good" in titles


# ======================================================================
# Edge Cases
# ======================================================================


class TestEdgeCases:
    """Edge case tests for HypothesisPipeline."""

    @pytest.mark.asyncio
    async def test_zero_correlations(
        self, mock_generator, mock_embedder, hypothesis_config,
    ) -> None:
        """Empty CorrelationMap produces empty result."""
        pipeline = HypothesisPipeline(
            generator=mock_generator, embedder=mock_embedder,
            config=hypothesis_config,
        )

        empty = CorrelationMap()
        result = await pipeline.run(correlation_map=empty, query="q")

        assert result.hypotheses == []
        assert result.total_input_papers == 0

    @pytest.mark.asyncio
    async def test_max_round_exhaustion(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Pipeline stops after max_reflection_rounds even without
        convergence. Each reflection round adds genuinely NEW
        hypotheses with different titles and significantly different
        scores so convergence never triggers."""
        claim = _make_claim("Claim", "paper-1")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")
        cmap = _build_correlation_map(claim, comp, "Test", mod)

        gen = AsyncMock()
        round_counter = [0]

        async def _generate(*args, **kwargs) -> list[dict]:
            return [{
                "title": "Initial Hypothesis",
                "description": "Initial description",
                "approach": "", "supporting_papers": [],
                "target_metric": "", "expected_improvement": "",
                "code_changes": "",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [
                    c["id"] for c in kwargs.get("correlations", [])
                    if isinstance(c, dict)
                ],
                "success_criteria": "",
            }]

        gen.generate = AsyncMock(side_effect=_generate)

        # Each round adds hypotheses with NEW title + different scores → never converges
        async def _critique(*args, **kwargs) -> dict:
            round_counter[0] += 1
            r = round_counter[0]
            return {
                "critiques": [],
                "new_hypotheses": [{
                    "title": f"Reflection Round {r} Hypothesis",
                    "description": f"Generated in round {r}",
                    "approach": "", "supporting_papers": [],
                    "target_metric": "", "expected_improvement": "",
                    "code_changes": "",
                    "scores": {
                        "relevance": 8, "feasibility": 8,
                        "evidence": 7,
                    },
                    "correlations": [
                        c["id"] for c in kwargs.get("correlations", [])
                        if isinstance(c, dict)
                    ],
                    "success_criteria": "",
                }],
            }

        gen.critique_and_expand = AsyncMock(side_effect=_critique)

        config = HypothesisConfig(
            max_reflection_rounds=2,
            relevance_gate=7,
            dedup_threshold=0.99,  # high threshold → no dedup removes anything
        )
        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")

        # Should have completed 2 rounds
        assert result.reflection_rounds_completed == 2
        assert not result.converged  # New hypotheses each round → no convergence

    @pytest.mark.asyncio
    async def test_reflection_rounds_counted_correctly(
        self, sample_correlation_map, mock_embedder, hypothesis_config,
    ) -> None:
        """reflection_rounds_completed tracks actual rounds run."""
        gen = _make_mock_gen_with_hypothesis("H")
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        config = HypothesisConfig(
            max_reflection_rounds=3, relevance_gate=7,
        )
        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=config,
        )

        result = await pipeline.run(
            correlation_map=sample_correlation_map, query="q",
        )

        # Since no new hypotheses are added in reflection,
        # it short-circuits after round 1
        assert result.reflection_rounds_completed in (1, 2)

    @pytest.mark.asyncio
    async def test_pipeline_never_raises(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Pipeline never raises even with completely broken generator."""
        gen = AsyncMock()
        gen.generate.side_effect = RuntimeError("Broken")
        gen.critique_and_expand.side_effect = RuntimeError("Broken")

        claim = _make_claim("C", "paper-1")
        comp = _make_component("f", "d", "m")
        mod = _make_module("m", "s")
        cmap = _build_correlation_map(claim, comp, "T", mod)

        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=hypothesis_config,
        )

        # This MUST NOT raise
        result = await pipeline.run(correlation_map=cmap, query="q")
        assert isinstance(result, HypothesisResult)
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_max_total_hypotheses_cap(
        self, mock_embedder,
    ) -> None:
        """Total hypotheses are capped at max_total_hypotheses (AD-09)."""
        claim = _make_claim("Claim", "paper-1")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")
        cmap = _build_correlation_map(claim, comp, "Test", mod)

        gen = AsyncMock()

        # Generate many hypotheses per paper
        async def _generate(*args, **kwargs) -> list[dict]:
            hypotheses = []
            for i in range(10):  # 10 per paper
                hypotheses.append({
                    "title": f"H{i}",
                    "description": f"Description {i}",
                    "approach": "", "supporting_papers": [],
                    "target_metric": "", "expected_improvement": "",
                    "code_changes": "",
                    "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                    "correlations": [
                        c["id"] for c in kwargs.get("correlations", [])
                        if isinstance(c, dict)
                    ],
                    "success_criteria": "",
                })
            return hypotheses

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        # Very low cap
        config = HypothesisConfig(
            max_total_hypotheses=3, relevance_gate=7,
            max_hypotheses_per_paper=10,
        )
        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=config,
        )

        result = await pipeline.run(correlation_map=cmap, query="q")
        assert len(result.hypotheses) <= 3

    @pytest.mark.asyncio
    async def test_hypotheses_have_composite_scores(
        self, sample_correlation_map, mock_generator,
        mock_embedder, hypothesis_config,
    ) -> None:
        """All returned hypotheses have valid composite scores."""
        pipeline = HypothesisPipeline(
            generator=mock_generator, embedder=mock_embedder,
            config=hypothesis_config,
        )

        result = await pipeline.run(
            correlation_map=sample_correlation_map, query="q",
        )

        for h in result.hypotheses:
            assert 1.0 <= h.composite <= 10.0
            assert isinstance(h.composite, float)
            assert isinstance(h.scores, HypothesisScores)
            assert 1 <= h.scores.relevance <= 10
            assert 1 <= h.scores.feasibility <= 10
            assert 1 <= h.scores.evidence <= 10

    @pytest.mark.asyncio
    async def test_general_content_reaches_adapter(
        self, mock_embedder, hypothesis_config,
    ) -> None:
        """Pipeline passes formatted general_content string to generator."""
        claim = _make_claim("Claim", "paper-1")
        comp = _make_component("func", "desc", "mod")
        mod = _make_module("mod", "summary")
        cmap = _build_correlation_map(claim, comp, "Test Paper", mod)

        gen = AsyncMock()

        async def _generate(
            title: str, paper_claims: str,
            correlations: list[dict], query: str, round_num: int = 0,
            general_content: str | None = None,
        ) -> list[dict]:
            return [{
                "title": "H from GC",
                "description": "Generated with general content context",
                "approach": "", "supporting_papers": [],
                "target_metric": "", "expected_improvement": "",
                "code_changes": "",
                "scores": {"relevance": 8, "feasibility": 7, "evidence": 6},
                "correlations": [c["id"] for c in correlations],
                "success_criteria": "",
            }]

        gen.generate = AsyncMock(side_effect=_generate)
        gen.critique_and_expand = AsyncMock(
            return_value={"critiques": [], "new_hypotheses": []},
        )

        pipeline = HypothesisPipeline(
            generator=gen, embedder=mock_embedder, config=hypothesis_config,
        )

        gc_items = [
            GeneralContent(
                source="tavily",
                title="Stable Training Guide",
                url="https://example.com/stable-training",
                body="Use gradient clipping and layer normalization for stability.",
            ),
        ]

        result = await pipeline.run(
            correlation_map=cmap, query="q", general_content=gc_items,
        )

        # Verify generate() was called with general_content keyword
        assert gen.generate.called
        call_kwargs = gen.generate.call_args.kwargs
        assert "general_content" in call_kwargs
        assert call_kwargs["general_content"] is not None
        assert "Stable Training Guide" in call_kwargs["general_content"]
        assert "gradient clipping" in call_kwargs["general_content"]

        # Verify pipeline still produces hypotheses
        assert len(result.hypotheses) >= 1

    @pytest.mark.asyncio
    async def test_general_content_none_backward_compatible(
        self, mock_generator, sample_correlation_map,
        mock_embedder, hypothesis_config,
    ) -> None:
        """Pipeline works correctly when general_content is None (default)."""
        pipeline = HypothesisPipeline(
            generator=mock_generator,
            embedder=mock_embedder,
            config=hypothesis_config,
        )

        # Run WITHOUT general_content (default None)
        result = await pipeline.run(
            correlation_map=sample_correlation_map,
            query="How can I improve my model?",
        )

        assert isinstance(result, HypothesisResult)
        assert len(result.hypotheses) > 0
        assert result.total_input_papers == 2

        # Verify generate() was called without general_content
        assert mock_generator.generate.called
        call_kwargs = mock_generator.generate.call_args.kwargs
        assert call_kwargs.get("general_content") is None
