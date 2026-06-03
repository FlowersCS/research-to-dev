"""Data contracts for the correlation pipeline.

Every ``Correlation`` and ``ModuleCorrelation`` retains full references
to its source ``Claim`` and target ``Component`` / ``ModuleSummary``
(CM-04, D5). The ``CorrelationMap`` captures a config snapshot for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from research_to_dev.codebase.types import Component, ModuleSummary
from research_to_dev.paper_profile.types import Claim


@dataclass
class Correlation:
    """A validated claim↔component correlation with full traceability (D5)."""

    claim: Claim
    """Full ``Claim`` — paper_id, section_name, provenance preserved."""

    paper_title: str
    """Convenience field for display and grouping."""

    component: Component
    """Full ``Component`` — file_path, module_name, signature preserved."""

    module_name: str
    """Dotted module name for grouping (same as ``component.module_name``)."""

    similarity_score: float
    """Raw cosine similarity score from the embedding pre-filter stage."""

    correlation_type: str
    """LLM-classified type: direct_solution | related_technique | contradicts | prerequisite."""

    reasoning: str
    """Natural language explanation from the LLM classifier."""


@dataclass
class ModuleCorrelation:
    """A validated claim↔module correlation at the architectural level (D5)."""

    claim: Claim
    """Full ``Claim`` reference."""

    paper_title: str
    """Convenience field for display and grouping."""

    module: ModuleSummary
    """Full ``ModuleSummary`` — summary, responsibilities preserved."""

    similarity_score: float
    """Raw cosine similarity score from the embedding pre-filter stage."""

    correlation_type: str
    """LLM-classified type."""

    reasoning: str
    """Natural language explanation from the LLM classifier."""


@dataclass
class CorrelationMap:
    """Top-level output of the correlation pipeline (D5, CM-04).

    Carries a config snapshot (``similarity_threshold``, ``embedding_model``)
    so results are reproducible.
    """

    correlations: list[Correlation] = field(default_factory=list)
    """Validated claim↔component matches."""

    module_correlations: list[ModuleCorrelation] = field(default_factory=list)
    """Validated claim↔module matches."""

    papers_analyzed: int = 0
    """Number of input ``PaperProfile`` objects processed."""

    total_claims: int = 0
    """Total number of claims across all papers."""

    total_components: int = 0
    """Total number of ``Component`` objects from the codebase."""

    total_modules: int = 0
    """Total number of ``ModuleSummary`` objects from the codebase."""

    matches_found: int = 0
    """Number of LLM-validated matches (component + module combined)."""

    similarity_threshold: float = 0.7
    """Cosine similarity threshold snapshot from config (CM-04)."""

    embedding_model: str = "text-embedding-3-small"
    """Embedding model snapshot from config (CM-04)."""


@dataclass
class CorrelationError:
    """Metadata about a failure during the correlation pipeline.

    Follows the same pattern as ``RankingError``, ``ProfilingError``,
    and ``CodebaseError``.
    """

    message: str
    """Human-readable error description."""

    exception_type: str | None = None
    """Python exception class name for debugging."""
