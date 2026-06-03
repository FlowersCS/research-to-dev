"""Tests for correlation data contracts (types.py)."""

from __future__ import annotations

import pytest

from research_to_dev.codebase.types import Component, ModuleSummary
from research_to_dev.correlation.types import (
    Correlation,
    CorrelationError,
    CorrelationMap,
    ModuleCorrelation,
)
from research_to_dev.paper_profile.types import Claim
from research_to_dev.shared.config import CorrelationMapConfig


# -------------------------------------------------------------------
# Claim + Component + ModuleSummary helpers for test construction
# -------------------------------------------------------------------


def _make_claim() -> Claim:
    return Claim(
        text="Attention mechanisms improve NLP accuracy",
        paper_id="paper-1",
        section_name="abstract",
    )


def _make_component() -> Component:
    return Component(
        name="attention_layer",
        kind="class",
        file_path="model/layers.py",
        line_start=10,
        line_end=50,
        signature="class AttentionLayer",
        description="Implements attention mechanism",
        module_name="model.layers",
    )


def _make_module_summary() -> ModuleSummary:
    return ModuleSummary(
        module_name="model.layers",
        file_path="model/layers.py",
        summary="Attention-based neural network layers",
        key_responsibilities=["attention", "normalization"],
    )


# ===================================================================
# CorrelationMapConfig
# ===================================================================


class TestCorrelationMapConfig:
    """Tests for CorrelationMapConfig in shared/config.py (CM-CFG-01)."""

    def test_default_construction(self) -> None:
        """Default values match the spec."""
        c = CorrelationMapConfig()
        assert c.similarity_threshold == 0.7
        assert c.embedding_model == "text-embedding-3-small"
        assert c.llm_model == "gpt-4o-mini"

    def test_custom_threshold_override(self) -> None:
        """Custom threshold overrides default, others keep defaults."""
        c = CorrelationMapConfig(similarity_threshold=0.65)
        assert c.similarity_threshold == 0.65
        assert c.embedding_model == "text-embedding-3-small"
        assert c.llm_model == "gpt-4o-mini"

    def test_custom_embedding_model(self) -> None:
        """Custom embedding model is set."""
        c = CorrelationMapConfig(embedding_model="text-embedding-3-large")
        assert c.embedding_model == "text-embedding-3-large"

    def test_custom_llm_model(self) -> None:
        """Custom LLM model is set."""
        c = CorrelationMapConfig(llm_model="gpt-4o")
        assert c.llm_model == "gpt-4o"

    def test_full_custom(self) -> None:
        """All fields can be customized."""
        c = CorrelationMapConfig(
            similarity_threshold=0.5,
            embedding_model="custom-embed",
            llm_model="custom-llm",
        )
        assert c.similarity_threshold == 0.5
        assert c.embedding_model == "custom-embed"
        assert c.llm_model == "custom-llm"

    def test_field_types(self) -> None:
        """Config fields have correct types."""
        c = CorrelationMapConfig()
        assert isinstance(c.similarity_threshold, float)
        assert isinstance(c.embedding_model, str)
        assert isinstance(c.llm_model, str)


# ===================================================================
# Correlation
# ===================================================================


class TestCorrelation:
    """Tests for the Correlation dataclass (CM-04, D5)."""

    def test_full_instantiation(self) -> None:
        """Correlation can be created with all fields."""
        claim = _make_claim()
        component = _make_component()
        corr = Correlation(
            claim=claim,
            paper_title="Deep Learning for NLP",
            component=component,
            module_name="model.layers",
            similarity_score=0.85,
            correlation_type="direct_solution",
            reasoning="The attention layer directly implements the attention mechanism.",
        )
        assert corr.claim is claim
        assert corr.paper_title == "Deep Learning for NLP"
        assert corr.component is component
        assert corr.module_name == "model.layers"
        assert corr.similarity_score == 0.85
        assert corr.correlation_type == "direct_solution"
        assert corr.reasoning == "The attention layer directly implements the attention mechanism."

    def test_field_types(self) -> None:
        """Correlation fields have correct types."""
        corr = Correlation(
            claim=_make_claim(),
            paper_title="Test",
            component=_make_component(),
            module_name="mod",
            similarity_score=0.75,
            correlation_type="related_technique",
            reasoning="Related",
        )
        assert isinstance(corr.claim, Claim)
        assert isinstance(corr.paper_title, str)
        assert isinstance(corr.component, Component)
        assert isinstance(corr.module_name, str)
        assert isinstance(corr.similarity_score, float)
        assert isinstance(corr.correlation_type, str)
        assert isinstance(corr.reasoning, str)

    def test_correlation_type_values(self) -> None:
        """All four correlation types are accepted."""
        claim = _make_claim()
        component = _make_component()
        for ctype in ("direct_solution", "related_technique", "contradicts", "prerequisite"):
            corr = Correlation(
                claim=claim,
                paper_title="Test",
                component=component,
                module_name="mod",
                similarity_score=0.8,
                correlation_type=ctype,
                reasoning=f"Type: {ctype}",
            )
            assert corr.correlation_type == ctype


# ===================================================================
# ModuleCorrelation
# ===================================================================


class TestModuleCorrelation:
    """Tests for the ModuleCorrelation dataclass (CM-04, D5)."""

    def test_full_instantiation(self) -> None:
        """ModuleCorrelation can be created with all fields."""
        claim = _make_claim()
        mod = _make_module_summary()
        mcorr = ModuleCorrelation(
            claim=claim,
            paper_title="Deep Learning for NLP",
            module=mod,
            similarity_score=0.82,
            correlation_type="related_technique",
            reasoning="The module provides foundational attention infrastructure.",
        )
        assert mcorr.claim is claim
        assert mcorr.paper_title == "Deep Learning for NLP"
        assert mcorr.module is mod
        assert mcorr.similarity_score == 0.82
        assert mcorr.correlation_type == "related_technique"
        assert mcorr.reasoning == "The module provides foundational attention infrastructure."

    def test_field_types(self) -> None:
        """ModuleCorrelation fields have correct types."""
        mcorr = ModuleCorrelation(
            claim=_make_claim(),
            paper_title="Test",
            module=_make_module_summary(),
            similarity_score=0.9,
            correlation_type="prerequisite",
            reasoning="Prerequisite work",
        )
        assert isinstance(mcorr.claim, Claim)
        assert isinstance(mcorr.paper_title, str)
        assert isinstance(mcorr.module, ModuleSummary)
        assert isinstance(mcorr.similarity_score, float)
        assert isinstance(mcorr.correlation_type, str)
        assert isinstance(mcorr.reasoning, str)


# ===================================================================
# CorrelationMap
# ===================================================================


class TestCorrelationMap:
    """Tests for the CorrelationMap dataclass (CM-04, D5)."""

    def test_default_construction(self) -> None:
        """CorrelationMap defaults to empty with zero stats."""
        cmap = CorrelationMap()
        assert cmap.correlations == []
        assert cmap.module_correlations == []
        assert cmap.papers_analyzed == 0
        assert cmap.total_claims == 0
        assert cmap.total_components == 0
        assert cmap.total_modules == 0
        assert cmap.matches_found == 0
        assert cmap.similarity_threshold == 0.7
        assert cmap.embedding_model == "text-embedding-3-small"

    def test_config_snapshot_preserved(self) -> None:
        """Config values are snapshotted in CorrelationMap."""
        cmap = CorrelationMap(
            similarity_threshold=0.55,
            embedding_model="custom-model",
        )
        assert cmap.similarity_threshold == 0.55
        assert cmap.embedding_model == "custom-model"

    def test_with_matches(self) -> None:
        """CorrelationMap tracks matches correctly."""
        claim = _make_claim()
        comp = _make_component()
        mod = _make_module_summary()
        corr = Correlation(
            claim=claim, paper_title="T", component=comp,
            module_name="m", similarity_score=0.8,
            correlation_type="direct_solution", reasoning="R",
        )
        mcorr = ModuleCorrelation(
            claim=claim, paper_title="T", module=mod,
            similarity_score=0.9, correlation_type="related_technique", reasoning="R",
        )
        cmap = CorrelationMap(
            correlations=[corr],
            module_correlations=[mcorr],
            papers_analyzed=5,
            total_claims=15,
            total_components=10,
            total_modules=3,
            matches_found=2,
        )
        assert len(cmap.correlations) == 1
        assert len(cmap.module_correlations) == 1
        assert cmap.matches_found == 2
        assert cmap.papers_analyzed == 5

    def test_field_types(self) -> None:
        """CorrelationMap fields have correct types."""
        cmap = CorrelationMap()
        assert isinstance(cmap.correlations, list)
        assert isinstance(cmap.module_correlations, list)
        assert isinstance(cmap.papers_analyzed, int)
        assert isinstance(cmap.total_claims, int)
        assert isinstance(cmap.total_components, int)
        assert isinstance(cmap.total_modules, int)
        assert isinstance(cmap.matches_found, int)
        assert isinstance(cmap.similarity_threshold, float)
        assert isinstance(cmap.embedding_model, str)


# ===================================================================
# CorrelationError
# ===================================================================


class TestCorrelationError:
    """Tests for the CorrelationError dataclass."""

    def test_full_instantiation(self) -> None:
        """CorrelationError can be created with message and exception_type."""
        err = CorrelationError(
            message="Embedding API timeout",
            exception_type="TimeoutError",
        )
        assert err.message == "Embedding API timeout"
        assert err.exception_type == "TimeoutError"

    def test_default_exception_type(self) -> None:
        """Exception type defaults to None."""
        err = CorrelationError(message="Something went wrong")
        assert err.exception_type is None

    def test_field_types(self) -> None:
        """CorrelationError fields have correct types."""
        err = CorrelationError(message="err")
        assert isinstance(err.message, str)
        assert err.exception_type is None or isinstance(err.exception_type, str)
